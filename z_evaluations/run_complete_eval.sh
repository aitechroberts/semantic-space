#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Master Evaluation Runner
# =============================================================================
# This script runs the complete evaluation pipeline:
# 1. Downloads data from S3 (ablation outputs)
# 2. Downloads Space3D-Bench (VQA questions)
# 3. Runs CLIP/VLM ablation comparison
# 4. Runs Space3D-Bench VQA evaluation
# 5. Runs Complex Queries evaluation (affordance/negation)
# 6. Generates comprehensive reports
#
# NOTE: Designed for 16GB GPU (RTX 4090 Laptop)
#       Models are loaded sequentially with GPU cleanup between runs
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AWS S3 Settings - Override with environment variables
S3_BUCKET="${S3_BUCKET:-data-finished-585780419748-us-east-1}"
S3_PREFIX="${S3_PREFIX:-}"  # Optional prefix within bucket (e.g., "experiments/v2")
AWS_PROFILE="${AWS_PROFILE:-}"

# OpenAI API (for GPT-4o-mini)
# Set via: export OPENAI_API_KEY="sk-your-key-here"

# Local paths
BASE_DIR="${BASE_DIR:-${HOME}/ablation_eval}"
DATA_DIR="${BASE_DIR}/data"
SPACE3D_DIR="${BASE_DIR}/Space3D-Bench"
RESULTS_DIR="${BASE_DIR}/results"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configs and scenes
CONFIGS=("oracle" "qwen" "paligemma")
SCENES=("room0" "room1" "office2" "office3")

# =============================================================================
# CONFIG -> MODEL MAPPING
# =============================================================================
# Each config folder was generated with a specific VLM + CLIP pairing:
#
#   oracle/     -> GPT-4o-mini       + MobileCLIP2-S3 (dfndr2b)
#   qwen/       -> Qwen3-VL-2B       + TinyCLIP-ViT-8M
#   paligemma/  -> PaliGemma2-3b     + PE-Core-T-16-384
#
# The embeddings are already baked into the pkl.gz files.
# For VQA/queries, we use the VLM that matches each config.
# =============================================================================

# Associative arrays for config -> model mapping
declare -A CONFIG_TO_VLM=(
    ["oracle"]="gpt4"
    ["qwen"]="qwen"
    ["paligemma"]="paligemma"
)

declare -A CONFIG_TO_CLIP=(
    ["oracle"]="mobileclip"
    ["qwen"]="tinyclip"
    ["paligemma"]="pecore"
)

# Options
SKIP_S3_SYNC="${SKIP_S3_SYNC:-false}"
SKIP_SPACE3D_DOWNLOAD="${SKIP_SPACE3D_DOWNLOAD:-false}"
SKIP_ABLATION_EVAL="${SKIP_ABLATION_EVAL:-false}"
SKIP_VQA_EVAL="${SKIP_VQA_EVAL:-false}"
SKIP_COMPLEX_EVAL="${SKIP_COMPLEX_EVAL:-false}"
SKIP_SPICE="${SKIP_SPICE:-false}"
DEVICE="${DEVICE:-cuda}"

# Python
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Complex queries file (should be in SCRIPTS_DIR)
COMPLEX_QUERIES_PATH="${SCRIPTS_DIR}/complex_queries.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log_section() { echo ""; echo "========== $* =========="; echo ""; }
log_error() { echo "[ERROR] $*" >&2; }

# ---------------------------------------------------------------------------
# GPU Memory Management (for 16GB RTX 4090 Laptop)
# ---------------------------------------------------------------------------

clear_gpu_memory() {
    log "Clearing GPU memory..."
    ${PYTHON_BIN} -c "
import gc
gc.collect()
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f'  GPU: {used:.2f}/{total:.2f} GB used')
except ImportError:
    pass
" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Prerequisites Check
# ---------------------------------------------------------------------------

check_prerequisites() {
    log_section "Checking Prerequisites"
    
    # Python
    if ! command -v "${PYTHON_BIN}" &>/dev/null; then
        log_error "Python not found: ${PYTHON_BIN}"
        exit 1
    fi
    log "Python: $(${PYTHON_BIN} --version)"
    
    # AWS CLI
    if ! command -v aws &>/dev/null; then
        log_error "AWS CLI not found"
        exit 1
    fi
    log "AWS CLI: $(aws --version | head -1)"
    
    # Python packages
    log "Checking Python packages..."
    ${PYTHON_BIN} -c "import torch; print(f'  PyTorch: {torch.__version__}')" || {
        log_error "PyTorch not installed"
        exit 1
    }
    
    ${PYTHON_BIN} -c "import numpy; print(f'  NumPy: {numpy.__version__}')" || {
        log_error "NumPy not installed"
        exit 1
    }
    
    ${PYTHON_BIN} -c "import open_clip; print('  OpenCLIP: OK')" || {
        log "Warning: open_clip not installed. CLIP eval may fail."
    }
    
    ${PYTHON_BIN} -c "from pycocoevalcap.cider.cider import Cider; print('  pycocoevalcap: OK')" || {
        log "Warning: pycocoevalcap not installed. Text metrics may fail."
    }
    
    log "Prerequisites OK"
}

# ---------------------------------------------------------------------------
# Step 1: Sync from S3
# ---------------------------------------------------------------------------

sync_s3_data() {
    log_section "Step 1: Syncing from S3"
    
    if [[ "${SKIP_S3_SYNC}" == "true" ]]; then
        log "Skipping S3 sync (SKIP_S3_SYNC=true)"
        return
    fi
    
    mkdir -p "${DATA_DIR}"
    
    AWS_OPTS=()
    if [[ -n "${AWS_PROFILE}" ]]; then
        AWS_OPTS+=("--profile" "${AWS_PROFILE}")
    fi
    
    # Build base S3 URI
    if [[ -n "${S3_PREFIX}" ]]; then
        S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"
    else
        S3_BASE="s3://${S3_BUCKET}"
    fi
    
    log "S3 Base URI: ${S3_BASE}"
    
    for config in "${CONFIGS[@]}"; do
        for scene in "${SCENES[@]}"; do
            local_path="${DATA_DIR}/${config}/${scene}"
            s3_path="${S3_BASE}/${config}/${scene}/"
            
            log "Syncing ${config}/${scene}..."
            mkdir -p "${local_path}"
            
            aws s3 sync "${s3_path}" "${local_path}" "${AWS_OPTS[@]}" \
                --exclude "*" \
                --include "*.json" \
                --include "*.pkl.gz" \
                2>/dev/null || log "  Warning: Some files may not have synced"
        done
    done
    
    log "S3 sync complete"
}

# ---------------------------------------------------------------------------
# Step 2: Download Space3D-Bench
# ---------------------------------------------------------------------------

download_space3d_bench() {
    log_section "Step 2: Downloading Space3D-Bench"
    
    if [[ "${SKIP_SPACE3D_DOWNLOAD}" == "true" ]]; then
        log "Skipping Space3D-Bench download (SKIP_SPACE3D_DOWNLOAD=true)"
        return
    fi
    
    if [[ -d "${SPACE3D_DIR}/data" ]]; then
        log "Space3D-Bench already exists at ${SPACE3D_DIR}"
        return
    fi
    
    mkdir -p "${SPACE3D_DIR}"
    cd "${SPACE3D_DIR}"
    
    log "Downloading data.zip..."
    wget -q --show-progress \
        "https://github.com/Space3D-Bench/Space3D-Bench/releases/download/v0.0.2/data.zip" \
        -O data.zip
    
    log "Extracting..."
    unzip -q data.zip
    rm data.zip
    
    log "Space3D-Bench downloaded to ${SPACE3D_DIR}"
    
    # Show question counts
    log "Question counts per scene:"
    for scene in room_0 room_1 office_2 office_3; do
        if [[ -f "data/${scene}/questions.json" ]]; then
            count=$(${PYTHON_BIN} -c "import json; print(len(json.load(open('data/${scene}/questions.json'))))")
            log "  ${scene}: ${count} questions"
        fi
    done
}

# ---------------------------------------------------------------------------
# Step 3: Run Ablation Evaluation (CLIP + VLM Comparison)
# ---------------------------------------------------------------------------

run_ablation_eval() {
    log_section "Step 3: Running Ablation Evaluation"
    
    if [[ "${SKIP_ABLATION_EVAL}" == "true" ]]; then
        log "Skipping ablation eval (SKIP_ABLATION_EVAL=true)"
        return
    fi
    
    mkdir -p "${RESULTS_DIR}/ablation"
    
    EVAL_CMD=("${PYTHON_BIN}" "${SCRIPTS_DIR}/run_ablation_eval.py")
    EVAL_CMD+=("--data_root" "${DATA_DIR}")
    EVAL_CMD+=("--output_dir" "${RESULTS_DIR}/ablation")
    EVAL_CMD+=("--scenes" "${SCENES[@]}")
    EVAL_CMD+=("--configs" "qwen" "paligemma")
    
    if [[ "${SKIP_SPICE}" == "true" ]]; then
        EVAL_CMD+=("--skip_spice")
    fi
    
    log "Running: ${EVAL_CMD[*]}"
    "${EVAL_CMD[@]}"
    
    log "Ablation evaluation complete"
}

# ---------------------------------------------------------------------------
# Step 4: Run VQA Evaluation
# ---------------------------------------------------------------------------

run_vqa_eval() {
    log_section "Step 4: Running Space3D-Bench VQA Evaluation"
    
    if [[ "${SKIP_VQA_EVAL}" == "true" ]]; then
        log "Skipping VQA eval (SKIP_VQA_EVAL=true)"
        return
    fi
    
    if [[ ! -d "${SPACE3D_DIR}/data" ]]; then
        log_error "Space3D-Bench not found. Run download first."
        return
    fi
    
    mkdir -p "${RESULTS_DIR}/vqa"
    
    # Run for each config with its matched VLM model
    for config in "${CONFIGS[@]}"; do
        local vlm_model="${CONFIG_TO_VLM[$config]}"
        local clip_model="${CONFIG_TO_CLIP[$config]}"
        
        log "Evaluating ${config} config..."
        log "  VLM: ${vlm_model}, CLIP: ${clip_model}"
        
        # Skip GPT-4 if no API key
        if [[ "${vlm_model}" == "gpt4" && -z "${OPENAI_API_KEY:-}" ]]; then
            log "  Skipping ${config}: OPENAI_API_KEY not set"
            continue
        fi
        
        EVAL_CMD=("${PYTHON_BIN}" "${SCRIPTS_DIR}/run_vqa_eval.py")
        EVAL_CMD+=("--scene_graphs_root" "${DATA_DIR}")
        EVAL_CMD+=("--space3d_root" "${SPACE3D_DIR}/data")
        EVAL_CMD+=("--output_dir" "${RESULTS_DIR}/vqa/${config}")
        EVAL_CMD+=("--config" "${config}")
        EVAL_CMD+=("--scenes" "${SCENES[@]}")
        EVAL_CMD+=("--vlm_models" "${vlm_model}")
        EVAL_CMD+=("--clip_models" "${clip_model}")
        EVAL_CMD+=("--device" "${DEVICE}")
        
        log "Running: ${EVAL_CMD[*]}"
        "${EVAL_CMD[@]}" || log "Warning: VQA eval for ${config} may have had errors"
        
        # Clear GPU memory between config runs
        clear_gpu_memory
    done
    
    log "VQA evaluation complete"
}

# ---------------------------------------------------------------------------
# Step 5: Run Complex Queries Evaluation (Affordance/Negation)
# ---------------------------------------------------------------------------

run_complex_queries_eval() {
    log_section "Step 5: Running Complex Queries Evaluation"
    
    if [[ "${SKIP_COMPLEX_EVAL}" == "true" ]]; then
        log "Skipping complex queries eval (SKIP_COMPLEX_EVAL=true)"
        return
    fi
    
    if [[ ! -f "${COMPLEX_QUERIES_PATH}" ]]; then
        log_error "Complex queries file not found: ${COMPLEX_QUERIES_PATH}"
        log "Please ensure complex_queries.json is in the scripts directory."
        return
    fi
    
    mkdir -p "${RESULTS_DIR}/complex_queries"
    
    # Run for each config with its matched VLM model
    for config in "${CONFIGS[@]}"; do
        local vlm_model="${CONFIG_TO_VLM[$config]}"
        local clip_model="${CONFIG_TO_CLIP[$config]}"
        
        log "Evaluating ${config} config on complex queries..."
        log "  VLM: ${vlm_model}, CLIP: ${clip_model}"
        
        # Skip GPT-4 if no API key
        if [[ "${vlm_model}" == "gpt4" && -z "${OPENAI_API_KEY:-}" ]]; then
            log "  Skipping ${config}: OPENAI_API_KEY not set"
            continue
        fi
        
        EVAL_CMD=("${PYTHON_BIN}" "${SCRIPTS_DIR}/run_complex_queries_eval.py")
        EVAL_CMD+=("--scene_graphs_root" "${DATA_DIR}")
        EVAL_CMD+=("--queries_path" "${COMPLEX_QUERIES_PATH}")
        EVAL_CMD+=("--output_dir" "${RESULTS_DIR}/complex_queries/${config}")
        EVAL_CMD+=("--config" "${config}")
        EVAL_CMD+=("--scenes" "${SCENES[@]}")
        EVAL_CMD+=("--vlm_models" "${vlm_model}")
        EVAL_CMD+=("--clip_models" "${clip_model}")
        EVAL_CMD+=("--device" "${DEVICE}")
        
        log "Running: ${EVAL_CMD[*]}"
        "${EVAL_CMD[@]}" || log "Warning: Complex queries eval for ${config} may have had errors"
        
        # Clear GPU memory between config runs
        clear_gpu_memory
    done
    
    log "Complex queries evaluation complete"
}

# ---------------------------------------------------------------------------
# Step 6: Generate Final Report
# ---------------------------------------------------------------------------

generate_final_report() {
    log_section "Step 6: Generating Final Report"
    
    REPORT_PATH="${RESULTS_DIR}/FINAL_REPORT.md"
    
    cat > "${REPORT_PATH}" << 'EOF'
# Complete Ablation Study Report

## Executive Summary

This report contains the complete evaluation results comparing different VLM and CLIP configurations for 3D scene understanding.

### Evaluation Datasets
1. **Scene Graph Quality** - CLIP embeddings + VLM caption comparison
2. **Space3D-Bench VQA** - Spatial question answering benchmark
3. **Complex Queries** - Affordance and negation reasoning

---

EOF

    # Append ablation results
    if [[ -f "${RESULTS_DIR}/ablation/ABLATION_RESULTS.md" ]]; then
        echo "## Part 1: Scene Graph Quality (CLIP + VLM Comparison)" >> "${REPORT_PATH}"
        echo "" >> "${REPORT_PATH}"
        tail -n +3 "${RESULTS_DIR}/ablation/ABLATION_RESULTS.md" >> "${REPORT_PATH}"
        echo "" >> "${REPORT_PATH}"
        echo "---" >> "${REPORT_PATH}"
        echo "" >> "${REPORT_PATH}"
    fi
    
    # Append VQA results
    for config in "${CONFIGS[@]}"; do
        if [[ -f "${RESULTS_DIR}/vqa/${config}/VQA_RESULTS.md" ]]; then
            echo "## Part 2: Space3D-Bench VQA (${config} config)" >> "${REPORT_PATH}"
            echo "" >> "${REPORT_PATH}"
            tail -n +3 "${RESULTS_DIR}/vqa/${config}/VQA_RESULTS.md" >> "${REPORT_PATH}"
            echo "" >> "${REPORT_PATH}"
            echo "---" >> "${REPORT_PATH}"
            echo "" >> "${REPORT_PATH}"
        fi
    done
    
    # Append Complex Queries results
    for config in "${CONFIGS[@]}"; do
        if [[ -f "${RESULTS_DIR}/complex_queries/${config}/COMPLEX_QUERIES_RESULTS.md" ]]; then
            echo "## Part 3: Complex Queries (${config} config)" >> "${REPORT_PATH}"
            echo "" >> "${REPORT_PATH}"
            tail -n +3 "${RESULTS_DIR}/complex_queries/${config}/COMPLEX_QUERIES_RESULTS.md" >> "${REPORT_PATH}"
            echo "" >> "${REPORT_PATH}"
            echo "---" >> "${REPORT_PATH}"
            echo "" >> "${REPORT_PATH}"
        fi
    done
    
    log "Final report saved to: ${REPORT_PATH}"
    
    # Print summary
    log_section "EVALUATION COMPLETE"
    log ""
    log "Results saved to: ${RESULTS_DIR}/"
    log ""
    log "Key files:"
    log "  - ${RESULTS_DIR}/FINAL_REPORT.md"
    log "  - ${RESULTS_DIR}/ablation/ABLATION_RESULTS.md"
    
    for config in "${CONFIGS[@]}"; do
        if [[ -f "${RESULTS_DIR}/vqa/${config}/VQA_RESULTS.md" ]]; then
            log "  - ${RESULTS_DIR}/vqa/${config}/VQA_RESULTS.md"
        fi
        if [[ -f "${RESULTS_DIR}/complex_queries/${config}/COMPLEX_QUERIES_RESULTS.md" ]]; then
            log "  - ${RESULTS_DIR}/complex_queries/${config}/COMPLEX_QUERIES_RESULTS.md"
        fi
    done
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo "============================================================"
    echo "   COMPLETE ABLATION EVALUATION PIPELINE"
    echo "   (Designed for 16GB GPU - Sequential Model Loading)"
    echo "============================================================"
    echo ""
    echo "S3 Configuration:"
    echo "  S3 Bucket:     ${S3_BUCKET}"
    if [[ -n "${S3_PREFIX}" ]]; then
    echo "  S3 Prefix:     ${S3_PREFIX}"
    fi
    echo ""
    echo "Local Paths:"
    echo "  Data Dir:      ${DATA_DIR}"
    echo "  Space3D Dir:   ${SPACE3D_DIR}"
    echo "  Results Dir:   ${RESULTS_DIR}"
    echo ""
    echo "Scenes:          ${SCENES[*]}"
    echo "Device:          ${DEVICE}"
    echo ""
    echo "Config -> Model Mapping:"
    echo "  oracle/     -> GPT-4o-mini     + MobileCLIP2-S3"
    echo "  qwen/       -> Qwen3-VL-2B     + TinyCLIP-ViT-8M"
    echo "  paligemma/  -> PaliGemma2-3b   + PE-Core-T-16-384"
    echo ""
    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "OpenAI API:      ✓ Key set (oracle config will run)"
    else
    echo "OpenAI API:      ✗ Not set (oracle config will be skipped)"
    fi
    echo ""
    echo "============================================================"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --s3-prefix)
                S3_PREFIX="$2"
                shift 2
                ;;
            --s3-uri)
                # Parse full S3 URI like s3://bucket/prefix/path
                S3_URI="$2"
                # Extract bucket and prefix
                S3_BUCKET=$(echo "${S3_URI}" | sed 's|s3://||' | cut -d'/' -f1)
                S3_PREFIX=$(echo "${S3_URI}" | sed 's|s3://||' | cut -d'/' -f2-)
                shift 2
                ;;
            --skip-s3)
                SKIP_S3_SYNC="true"
                shift
                ;;
            --skip-space3d)
                SKIP_SPACE3D_DOWNLOAD="true"
                shift
                ;;
            --skip-ablation)
                SKIP_ABLATION_EVAL="true"
                shift
                ;;
            --skip-vqa)
                SKIP_VQA_EVAL="true"
                shift
                ;;
            --skip-complex)
                SKIP_COMPLEX_EVAL="true"
                shift
                ;;
            --skip-spice)
                SKIP_SPICE="true"
                shift
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "This script evaluates 3 config pipelines, each with its own VLM + CLIP:"
                echo ""
                echo "  Config        VLM Model           CLIP Model"
                echo "  ------        ---------           ----------"
                echo "  oracle/       GPT-4o-mini         MobileCLIP2-S3"
                echo "  qwen/         Qwen3-VL-2B         TinyCLIP-ViT-8M"
                echo "  paligemma/    PaliGemma2-3b       PE-Core-T-16-384"
                echo ""
                echo "S3 Configuration:"
                echo "  --s3-bucket BUCKET   Set S3 bucket name"
                echo "  --s3-prefix PREFIX   Set S3 prefix path within bucket"
                echo "  --s3-uri URI         Set full S3 URI (e.g., s3://bucket/path)"
                echo ""
                echo "Skip Options:"
                echo "  --skip-s3            Skip S3 data sync"
                echo "  --skip-space3d       Skip Space3D-Bench download"
                echo "  --skip-ablation      Skip ablation evaluation"
                echo "  --skip-vqa           Skip Space3D-Bench VQA evaluation"
                echo "  --skip-complex       Skip complex queries evaluation"
                echo "  --skip-spice         Skip SPICE metric (faster)"
                echo ""
                echo "Other Options:"
                echo "  --device DEVICE      Set device (cuda/cpu)"
                echo ""
                echo "Environment Variables:"
                echo "  OPENAI_API_KEY       Required for oracle config (GPT-4o-mini)"
                echo "  AWS_PROFILE          AWS profile to use"
                echo "  S3_BUCKET            S3 bucket name"
                echo "  S3_PREFIX            S3 path prefix"
                echo "  BASE_DIR             Base directory (default: ~/ablation_eval)"
                echo ""
                echo "Examples:"
                echo "  # Run all (requires OPENAI_API_KEY for oracle)"
                echo "  export OPENAI_API_KEY='sk-...'"
                echo "  $0"
                echo ""
                echo "  # Run with custom S3 path"
                echo "  $0 --s3-uri s3://my-bucket/experiments/v1"
                echo ""
                echo "  # Skip oracle (no OpenAI key), skip SPICE (faster)"
                echo "  $0 --skip-spice"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    check_prerequisites
    sync_s3_data
    download_space3d_bench
    run_ablation_eval
    run_vqa_eval
    run_complex_queries_eval
    generate_final_report
}

main "$@"
