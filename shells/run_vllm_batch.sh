#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# vLLM Native Serve Batch Processing Script
#
# Runs one VLM model at a time via native vllm serve (subprocess), processes
# scenes via batch_vlm_mapping_api.py, then tears down the vLLM server.
#
# Usage:
#   ./shells/run_vllm_batch.sh
#   VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" ./shells/run_vllm_batch.sh
#   VLM_MODEL="OpenGVLab/InternVL3-2B" PROMPT_CONFIG="prompts_compact" ./shells/run_vllm_batch.sh
# =============================================================================

# ---------------------------------------------------------------------------
# Configuration (override any of these via environment variables)
# ---------------------------------------------------------------------------

# HuggingFace model ID -- the single variable to change between experiments
VLM_MODEL="${VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct-AWQ}"

# vLLM serve settings
VLLM_CMD="${VLLM_CMD:-uv run vllm serve}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"

# Prompt config: "prompts_standard" for capable VLMs, "prompts_compact" for smaller ones
PROMPT_CONFIG="${PROMPT_CONFIG:-prompts_standard}"

# Vision encoder embedding extraction
EXTRACT_ENCODER="${EXTRACT_ENCODER:-true}"

# Paths
REPO_ROOT="${REPO_ROOT:-$HOME/cmu-grad/neuro-nav}"
DATA_ROOT="${DATA_ROOT:-$HOME/cmu-grad/neuro-data}"
REPLICA_ROOT="${REPLICA_ROOT:-${DATA_ROOT}/Replica}"
SCANNET_ROOT="${SCANNET_ROOT:-${DATA_ROOT}/ScanNet/scans}"
CKPT_DIR="${CKPT_DIR:-${DATA_ROOT}/checkpoints}"

# Dataset configs (resolved from REPO_ROOT)
REPLICA_DATASET_CONFIG="${REPO_ROOT}/conceptgraph/dataset/dataconfigs/replica/replica.yaml"
SCANNET_DATASET_CONFIG="${REPO_ROOT}/conceptgraph/dataset/dataconfigs/scannet/base.yaml"

# Scenes to process (set to empty string to skip a dataset)
# room1 office2 office3
SCENES="${SCENES:-room1 office2 }"
# scene0046_00  scene0222_00 scene0389_00 scene0435_00
SCANNET_SCENES="${SCANNET_SCENES:-}"

# Experiment labels
EXP_SUFFIX="${EXP_SUFFIX:-batch_api}"
DET_EXP_SUFFIX="${DET_EXP_SUFFIX:-s_detections_api}"

# Pipeline mode: "all-in-one" (default) or "staged"
PIPELINE_MODE="${PIPELINE_MODE:-all-in-one}"

# Mapping script (used for all-in-one mode; staged mode uses run_staged_pipeline.sh)
PY_SCRIPT="conceptgraph/slam/vlm_run/batch_vlm_mapping_api.py"

# Mapping controls
DEVICE="${DEVICE:-cuda}"
START="${START:-0}"
END="${END:--1}"
STRIDE="${STRIDE:-10}"
MAKE_EDGES="${MAKE_EDGES:-true}"
FORCE_DET="${FORCE_DET:-true}"
SAVE_JSON="${SAVE_JSON:-true}"
SAVE_PCD="${SAVE_PCD:-true}"
SAVE_SEMANTIC_SNAPSHOT="${SAVE_SEMANTIC_SNAPSHOT:-true}"
VIS_RENDER="${VIS_RENDER:-false}"
USE_WANDB="${USE_WANDB:-false}"

# Server health check
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
HEALTH_INTERVAL="${HEALTH_INTERVAL:-5}"

# S3 upload (set to empty string to skip)
S3_OUTPUT_URI="${S3_OUTPUT_URI:-}"
AWS_PROFILE="${AWS_PROFILE:-acct2}"

# Python interpreter (use the project venv directly)
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# HuggingFace cache (used via HF_HOME for model downloads)
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"

# ---------------------------------------------------------------------------
# Sanity Checks
# ---------------------------------------------------------------------------

if [[ "${VLLM_CMD}" == *"uv run"* ]]; then
    if ! uv run vllm --help >/dev/null 2>&1; then
        echo "[vllm-batch] ERROR: vllm not found. Run 'uv add vllm' or ensure vllm is in the project venv."
        exit 1
    fi
else
    if ! command -v vllm >/dev/null 2>&1; then
        echo "[vllm-batch] ERROR: vllm not found in PATH. Set VLLM_CMD='uv run vllm serve' to use project venv."
        exit 1
    fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[vllm-batch] ERROR: ${PYTHON_BIN} not found in PATH."
    exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "[vllm-batch] ERROR: REPO_ROOT does not exist: ${REPO_ROOT}"
    exit 1
fi

if [[ -n "${SCENES}" && ! -d "${REPLICA_ROOT}" ]]; then
    echo "[vllm-batch] ERROR: REPLICA_ROOT does not exist: ${REPLICA_ROOT}"
    exit 1
fi

if [[ -n "${SCANNET_SCENES}" && ! -d "${SCANNET_ROOT}" ]]; then
    echo "[vllm-batch] ERROR: SCANNET_ROOT does not exist: ${SCANNET_ROOT}"
    exit 1
fi

echo "=================================================================="
echo "[vllm-batch] Configuration"
echo "=================================================================="
echo "  Model:             ${VLM_MODEL}"
echo "  vLLM command:      ${VLLM_CMD}"
echo "  GPU Mem Util:      ${GPU_MEM_UTIL}"
echo "  Max Model Len:     ${MAX_MODEL_LEN}"
echo "  Port:              ${VLLM_PORT}"
echo "  Pipeline Mode:     ${PIPELINE_MODE}"
echo "  Prompt Config:     ${PROMPT_CONFIG}"
echo "  Extract Encoder:   ${EXTRACT_ENCODER}"
echo "  Replica Scenes:    ${SCENES:-<none>}"
echo "  ScanNet Scenes:    ${SCANNET_SCENES:-<none>}"
echo "=================================================================="

# ---------------------------------------------------------------------------
# 1. vLLM Server Management
# ---------------------------------------------------------------------------

VLLM_PID=""
VLLM_LOG=""
VLLM_MANAGED=false

cleanup_vllm() {
    if [[ "${VLLM_MANAGED}" != "true" ]]; then
        return 0
    fi
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[vllm-batch] Stopping vLLM serve (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
    [[ -n "${VLLM_LOG:-}" && -f "${VLLM_LOG}" ]] && rm -f "${VLLM_LOG}"
}
trap cleanup_vllm EXIT INT TERM

mkdir -p "${HF_CACHE}"
export HF_HOME="${HF_CACHE}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"

cd "${REPO_ROOT}"

echo "[vllm-batch] Ensuring model is cached locally: ${VLM_MODEL}..."
uv run hf download "${VLM_MODEL}"
echo "[vllm-batch] Model ready."

HEALTH_URL="http://localhost:${VLLM_PORT}/health"

start_vllm() {
    VLLM_MANAGED=true
    [[ -z "${VLLM_LOG}" ]] && VLLM_LOG=$(mktemp)
    echo "[vllm-batch] Starting vLLM serve for ${VLM_MODEL}..."
    ${VLLM_CMD} "${VLM_MODEL}" --port "${VLLM_PORT}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --trust-remote-code \
        --dtype auto \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    echo "[vllm-batch] vLLM serve started (PID ${VLLM_PID}). Waiting for health check..."
    echo "[vllm-batch] Log file: ${VLLM_LOG}"

    local ELAPSED=0
    while [[ ${ELAPSED} -lt ${HEALTH_TIMEOUT} ]]; do
        if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
            echo "[vllm-batch] Server is ready after ${ELAPSED}s."
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[vllm-batch] ERROR: vLLM process died during startup."
            echo "[vllm-batch] Last 50 lines of vLLM output:"
            tail -50 "${VLLM_LOG}"
            return 1
        fi
        sleep "${HEALTH_INTERVAL}"
        ELAPSED=$((ELAPSED + HEALTH_INTERVAL))
        echo "[vllm-batch] Waiting... (${ELAPSED}/${HEALTH_TIMEOUT}s)"
    done

    echo "[vllm-batch] ERROR: Server did not become healthy within ${HEALTH_TIMEOUT}s."
    echo "[vllm-batch] Last 50 lines of vLLM output:"
    tail -50 "${VLLM_LOG}"
    return 1
}

ensure_vllm_alive() {
    if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
        return 0
    fi
    if [[ "${VLLM_MANAGED}" != "true" ]]; then
        echo "[vllm-batch] ERROR: External vLLM server is no longer responding."
        echo "[vllm-batch] Please check your vLLM terminal and restart it."
        return 1
    fi
    echo "[vllm-batch] Managed vLLM server is down. Restarting..."
    cleanup_vllm
    sleep 2
    VLLM_LOG=$(mktemp)
    start_vllm
}

# ---------------------------------------------------------------------------
# 2. Detect or Start vLLM Server
# ---------------------------------------------------------------------------

if curl -sf "${HEALTH_URL}" >/dev/null 2>&1; then
    echo "[vllm-batch] Found existing vLLM server on port ${VLLM_PORT}. Using it."
    echo "[vllm-batch] (Server lifecycle is NOT managed by this script.)"
    VLLM_MANAGED=false
else
    if ! start_vllm; then
        cleanup_vllm
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 3. Run Scenes
# ---------------------------------------------------------------------------

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export REPO_ROOT DATA_ROOT CKPT_DIR
export VLM_API_URL="http://localhost:${VLLM_PORT}/v1"
export VLM_MODEL_NAME="${VLM_MODEL}"
export EXTRACT_VLM_ENCODER_FEATS="${EXTRACT_ENCODER}"
export MAKE_EDGES SAVE_JSON SAVE_PCD SAVE_SEMANTIC_SNAPSHOT VIS_RENDER USE_WANDB

if [[ -n "${S3_OUTPUT_URI}" ]]; then
    export AWS_PROFILE
fi

# ---------------------------------------------------------------------------
# run_scene <scene_id> <dataset_root> <dataset_config> <scene_num> <total>
# ---------------------------------------------------------------------------
run_scene() {
    local SCENE="$1"
    local DS_ROOT="$2"
    local DS_CONFIG="$3"
    local SCENE_NUM="$4"
    local TOTAL="$5"
    local SCENE_DIR="${DS_ROOT}/${SCENE}"

    if [[ ! -d "${SCENE_DIR}" ]]; then
        echo "[vllm-batch][skip] Scene directory not found: ${SCENE_DIR}"
        return 0
    fi

    ensure_vllm_alive

    echo ""
    echo "=================================================================="
    echo "[vllm-batch] Scene ${SCENE_NUM}/${TOTAL}: ${SCENE}"
    echo "=================================================================="

    export OUTPUT_ROOT="${DS_ROOT}"

    OVERRIDES=()
    OVERRIDES+=("repo_root=${REPO_ROOT}")
    OVERRIDES+=("data_root=${DATA_ROOT}")
    OVERRIDES+=("dataset_root=${DS_ROOT}")
    OVERRIDES+=("dataset_config=${DS_CONFIG}")
    OVERRIDES+=("scene_id=${SCENE}")
    OVERRIDES+=("exp_suffix=${EXP_SUFFIX}")
    OVERRIDES+=("detections_exp_suffix=${DET_EXP_SUFFIX}")
    OVERRIDES+=("device=${DEVICE}")
    OVERRIDES+=("start=${START}")
    OVERRIDES+=("end=${END}")
    OVERRIDES+=("stride=${STRIDE}")
    OVERRIDES+=("make_edges=${MAKE_EDGES}")
    OVERRIDES+=("force_detection=${FORCE_DET}")
    OVERRIDES+=("save_json=${SAVE_JSON}")
    OVERRIDES+=("save_pcd=${SAVE_PCD}")
    OVERRIDES+=("save_semantic_snapshot=${SAVE_SEMANTIC_SNAPSHOT}")
    OVERRIDES+=("vis_render=${VIS_RENDER}")
    OVERRIDES+=("use_wandb=${USE_WANDB}")
    OVERRIDES+=("vlm_api_url=${VLM_API_URL}")
    OVERRIDES+=("vlm_model_name=${VLM_MODEL}")

    if [[ "${PIPELINE_MODE}" == "staged" ]]; then
        echo "[vllm-batch] Launching staged pipeline for ${SCENE}..."
        set +e
        bash "${REPO_ROOT}/shells/run_staged_pipeline.sh" "${OVERRIDES[@]}"
        local RET=$?
        set -e
    else
        echo "[vllm-batch] Launching: ${PYTHON_BIN} ${PY_SCRIPT} for ${SCENE}..."
        set +e
        "${PYTHON_BIN}" "${PY_SCRIPT}" "${OVERRIDES[@]}"
        local RET=$?
        set -e
    fi

    if [[ ${RET} -ne 0 ]]; then
        echo "[vllm-batch][error] Mapping failed for ${SCENE} (exit code ${RET}). Skipping upload."
        return 0
    fi

    echo "[vllm-batch] Mapping succeeded for ${SCENE}."

    if [[ -n "${S3_OUTPUT_URI}" ]]; then
        local SCENE_OUTPUT_DIR="${DS_ROOT}/${SCENE}/exps/${EXP_SUFFIX}"
        if [[ -d "${SCENE_OUTPUT_DIR}" ]]; then
            local S3_SCENE_URI="${S3_OUTPUT_URI%/}/${SCENE}"
            echo "[vllm-batch] Uploading to ${S3_SCENE_URI}..."
            aws s3 sync "${SCENE_OUTPUT_DIR}" "${S3_SCENE_URI}"
        fi
    fi

    echo "[vllm-batch] Done with scene ${SCENE}."
}

# ---------------------------------------------------------------------------
# 3a. Replica Scenes
# ---------------------------------------------------------------------------

REPLICA_ARRAY=(${SCENES})
SCANNET_ARRAY=(${SCANNET_SCENES})
TOTAL_SCENES=$(( ${#REPLICA_ARRAY[@]} + ${#SCANNET_ARRAY[@]} ))
SCENE_NUM=0

if [[ ${#REPLICA_ARRAY[@]} -gt 0 ]]; then
    echo ""
    echo "[vllm-batch] === Replica (${#REPLICA_ARRAY[@]} scenes) ==="
    for SCENE in "${REPLICA_ARRAY[@]}"; do
        SCENE_NUM=$((SCENE_NUM + 1))
        run_scene "${SCENE}" "${REPLICA_ROOT}" "${REPLICA_DATASET_CONFIG}" "${SCENE_NUM}" "${TOTAL_SCENES}"
    done
fi

# ---------------------------------------------------------------------------
# 3b. ScanNet Scenes
# ---------------------------------------------------------------------------

if [[ ${#SCANNET_ARRAY[@]} -gt 0 ]]; then
    echo ""
    echo "[vllm-batch] === ScanNet (${#SCANNET_ARRAY[@]} scenes) ==="
    for SCENE in "${SCANNET_ARRAY[@]}"; do
        SCENE_NUM=$((SCENE_NUM + 1))
        run_scene "${SCENE}" "${SCANNET_ROOT}" "${SCANNET_DATASET_CONFIG}" "${SCENE_NUM}" "${TOTAL_SCENES}"
    done
fi

# ---------------------------------------------------------------------------
# 4. Teardown
# ---------------------------------------------------------------------------

echo ""
echo "[vllm-batch] All scenes processed. Stopping vLLM serve..."
# Trap handles cleanup on exit

echo "[vllm-batch] Complete. Model: ${VLM_MODEL}"
echo "[vllm-batch]   Replica: ${SCENES:-<none>}"
echo "[vllm-batch]   ScanNet: ${SCANNET_SCENES:-<none>}"
