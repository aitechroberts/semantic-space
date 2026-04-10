#!/usr/bin/env bash
set -euo pipefail
OVERRIDES=("$@")

# =============================================================================
# Phase A — Oracle scene construction (run once per scene)
# Geometry only, no language models.
# =============================================================================

echo "[staged] Phase A — Stage 1/4: Detection + 3D lifting + 1.5x crops"
python -m conceptgraph.stages.detect "${OVERRIDES[@]}"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[staged] detect.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[staged] Phase A — Stage 2/4: Oracle encoder feature extraction"
python -m conceptgraph.stages.embed "${OVERRIDES[@]}"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[staged] embed.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[staged] Phase A — Stage 3/4: Map building (matching + merging)"
python -m conceptgraph.stages.build_map "${OVERRIDES[@]}"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[staged] build_map.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[staged] Phase A — Stage 4/4: Oracle finalization (MST edges + HPSG planes)"
python -m conceptgraph.stages.oracle_finalize "${OVERRIDES[@]}"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[staged] oracle_finalize.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[staged] Phase A complete. Oracle scene saved."

# =============================================================================
# Phase B — Semantic evaluation (run per encoder x VLM combination)
# User sets ENCODER and VLM env vars, or passes Hydra overrides.
# =============================================================================

ENCODER="${ENCODER:-openai/clip-vit-large-patch14}"
VLM="${VLM:-Qwen/Qwen3-VL-2B-Instruct}"

echo ""
echo "[staged] Phase B — encoder=${ENCODER}, vlm=${VLM}"

echo "[staged] Phase B — Step 1/4: Re-embed with evaluation encoder"
python -m conceptgraph.stages.embed "${OVERRIDES[@]}" \
    embed.mode=re_embed \
    "embed.encoder_name=${ENCODER}"

echo "[staged] Phase B — Step 2/4: Per-object VLM captioning"
python -m conceptgraph.stages.caption "${OVERRIDES[@]}" \
    "caption.vlm_name=${VLM}"

SAFE_ENC="${ENCODER//\//_}"
SAFE_VLM="${VLM//\//_}"

echo "[staged] Phase B — Step 3/4: Semantic assembly"
python -m conceptgraph.stages.semantic_assemble "${OVERRIDES[@]}" \
    "assemble.encoder=${SAFE_ENC}" \
    "assemble.vlm=${SAFE_VLM}"

echo "[staged] Phase B — Step 4/4: Evaluation"
python -m conceptgraph.stages.eval "${OVERRIDES[@]}" \
    "eval.encoder=${SAFE_ENC}" \
    "eval.vlm=${SAFE_VLM}"

echo "[staged] Pipeline complete."
