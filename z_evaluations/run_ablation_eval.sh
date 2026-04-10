#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Ablation Evaluation Pipeline
# =============================================================================
# This script:
# 1. Syncs evaluation data from S3
# 2. Runs CLIP embedding + VLM text evaluations
# 3. Generates markdown report tables
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration - EDIT THESE
# ---------------------------------------------------------------------------

# S3 bucket containing evaluation outputs
S3_BUCKET="data-finished-585780419748-us-east-1"

# Configurations to evaluate (folders in S3)
CONFIGS=("oracle" "qwen" "paligemma")

# Scenes to process
SCENES=("room0" "room1" "office2" "office3")

# Local directory for downloaded data
DATA_ROOT="${HOME}/ablation_eval_data"

# Output directory for evaluation results
OUTPUT_DIR="${HOME}/ablation_eval_results"

# AWS Profile (optional)
AWS_PROFILE="${AWS_PROFILE:-}"

# Skip SPICE metric (set to "true" for faster evaluation)
SKIP_SPICE="false"

# Python interpreter
PYTHON_BIN="python3"

# Path to evaluation script (relative to this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/run_ablation_eval.py"

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v aws >/dev/null 2>&1; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        log_error "Python not found at: ${PYTHON_BIN}"
        exit 1
    fi
    
    # Check Python packages
    "${PYTHON_BIN}" -c "import torch" 2>/dev/null || {
        log_error "PyTorch not installed. Run: pip install torch"
        exit 1
    }
    
    "${PYTHON_BIN}" -c "import numpy" 2>/dev/null || {
        log_error "NumPy not installed. Run: pip install numpy"
        exit 1
    }
    
    # Check optional packages
    "${PYTHON_BIN}" -c "from pycocoevalcap.cider.cider import Cider" 2>/dev/null || {
        log_info "pycocoevalcap not found. Text metrics will be skipped."
        log_info "To enable: pip install pycocoevalcap"
    }
    
    log_info "Dependencies OK"
}

# ---------------------------------------------------------------------------
# S3 Sync
# ---------------------------------------------------------------------------

sync_from_s3() {
    log_info "Syncing data from S3..."
    
    mkdir -p "${DATA_ROOT}"
    
    # Build AWS CLI options
    AWS_OPTS=()
    if [[ -n "${AWS_PROFILE}" ]]; then
        AWS_OPTS+=("--profile" "${AWS_PROFILE}")
    fi
    
    for config in "${CONFIGS[@]}"; do
        log_info "Syncing ${config}..."
        
        for scene in "${SCENES[@]}"; do
            local_path="${DATA_ROOT}/${config}/${scene}"
            s3_path="s3://${S3_BUCKET}/${config}/${scene}/"
            
            mkdir -p "${local_path}"
            
            log_info "  ${s3_path} -> ${local_path}"
            aws s3 sync "${s3_path}" "${local_path}" "${AWS_OPTS[@]}" \
                --exclude "*" \
                --include "*.json" \
                --include "*.pkl.gz" \
                || log_error "Failed to sync ${config}/${scene}"
        done
    done
    
    log_info "S3 sync complete"
}

# ---------------------------------------------------------------------------
# Run Evaluation
# ---------------------------------------------------------------------------

run_evaluation() {
    log_info "Running evaluation..."
    
    mkdir -p "${OUTPUT_DIR}"
    
    # Build evaluation command
    EVAL_CMD=("${PYTHON_BIN}" "${EVAL_SCRIPT}")
    EVAL_CMD+=("--data_root" "${DATA_ROOT}")
    EVAL_CMD+=("--output_dir" "${OUTPUT_DIR}")
    EVAL_CMD+=("--scenes" "${SCENES[@]}")
    EVAL_CMD+=("--configs" "qwen" "paligemma")
    
    if [[ "${SKIP_SPICE}" == "true" ]]; then
        EVAL_CMD+=("--skip_spice")
    fi
    
    log_info "Command: ${EVAL_CMD[*]}"
    "${EVAL_CMD[@]}"
    
    log_info "Evaluation complete"
}

# ---------------------------------------------------------------------------
# Print Results Summary
# ---------------------------------------------------------------------------

print_summary() {
    log_info "=" 
    log_info "EVALUATION COMPLETE"
    log_info "="
    
    if [[ -f "${OUTPUT_DIR}/ABLATION_RESULTS.md" ]]; then
        echo ""
        echo "===== RESULTS SUMMARY ====="
        cat "${OUTPUT_DIR}/ABLATION_RESULTS.md"
        echo ""
    fi
    
    log_info "Results saved to:"
    log_info "  - JSON: ${OUTPUT_DIR}/ablation_results.json"
    log_info "  - Markdown: ${OUTPUT_DIR}/ABLATION_RESULTS.md"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    echo "============================================================"
    echo "   ABLATION EVALUATION PIPELINE"
    echo "============================================================"
    echo "S3 Bucket: ${S3_BUCKET}"
    echo "Configs:   ${CONFIGS[*]}"
    echo "Scenes:    ${SCENES[*]}"
    echo "Output:    ${OUTPUT_DIR}"
    echo "============================================================"
    echo ""
    
    check_dependencies
    
    # Parse arguments
    SKIP_SYNC="false"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-sync)
                SKIP_SYNC="true"
                shift
                ;;
            --skip-spice)
                SKIP_SPICE="true"
                shift
                ;;
            --data-root)
                DATA_ROOT="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
    
    if [[ "${SKIP_SYNC}" != "true" ]]; then
        sync_from_s3
    else
        log_info "Skipping S3 sync (--skip-sync)"
    fi
    
    run_evaluation
    print_summary
}

main "$@"
