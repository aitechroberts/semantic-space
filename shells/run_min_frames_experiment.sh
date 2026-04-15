#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Minimum-frames ablation experiment
#
# Runs a full-scene oracle (stride=1), creates subsets at various frame
# counts via stride and fps_pose selectors, builds maps on each subset,
# and compares the resulting oracles geometrically.
# =============================================================================

# --- Configuration (override via environment) ---
SCENE="${SCENE:-office0}"
DATASET_ROOT="${DATASET_ROOT:-/home/jrob/cmu-grad/neuro-data/Replica}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jrob/cmu-grad/neuro-experiments/min-frames}"
SEG_BACKEND="${SEG_BACKEND:-sam3_auto}"

ORACLE_DIR="$OUTPUT_ROOT"
ABLATION_ROOT="$OUTPUT_ROOT/ablations"
RESULTS_DIR="$OUTPUT_ROOT/results"

COMMON_OVERRIDES=(
    "scene_id=$SCENE"
    "dataset_root=$DATASET_ROOT"
    "exp_suffix=min_frames"
)

# =============================================================================
# Step 1: Oracle detection (stride=1, run once — expensive)
# =============================================================================

echo "[min-frames] Step 1: Oracle detection (stride=1)"
python -m semgraph.stages.detect \
    "${COMMON_OVERRIDES[@]}" \
    "output_root=$ORACLE_DIR" \
    "segmentation_backend=$SEG_BACKEND" \
    stride=1
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[min-frames] detect.py failed (exit $RET). Aborting."
    exit $RET
fi

# =============================================================================
# Step 2: Full Phase A on stride=1 (embed, build_map, oracle_finalize)
# =============================================================================

echo "[min-frames] Step 2a: Embed (oracle encoder features)"
python -m semgraph.stages.embed \
    "${COMMON_OVERRIDES[@]}" \
    "output_root=$ORACLE_DIR"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[min-frames] embed.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[min-frames] Step 2b: Build map"
python -m semgraph.stages.build_map \
    "${COMMON_OVERRIDES[@]}" \
    "output_root=$ORACLE_DIR"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[min-frames] build_map.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[min-frames] Step 2c: Oracle finalize"
python -m semgraph.stages.oracle_finalize \
    "${COMMON_OVERRIDES[@]}" \
    "output_root=$ORACLE_DIR"
RET=$?
if [[ $RET -ne 0 ]]; then
    echo "[min-frames] oracle_finalize.py failed (exit $RET). Aborting."
    exit $RET
fi

echo "[min-frames] Reference oracle complete."
echo ""

# =============================================================================
# Step 3a: Create subsets (initial bracket)
# =============================================================================

ORACLE_FD="$ORACLE_DIR/$SCENE/stages/frame_data"
BRACKET_COUNTS=(200 50 10)

echo "[min-frames] Step 3a: Creating subsets ..."
for N in "${BRACKET_COUNTS[@]}"; do
    for METHOD in stride fps_pose; do
        NAME="${METHOD}_${N}"
        SUBSET_ROOT="$ABLATION_ROOT/$NAME"
        echo "  Creating subset: $NAME"
        python -m semgraph.scripts.create_subset \
            --source "$ORACLE_FD" \
            --dest "$SUBSET_ROOT/$SCENE/stages/frame_data" \
            --method "$METHOD" --n_frames "$N"
    done
done
echo ""

# =============================================================================
# Step 3a continued: build_map + oracle_finalize on each subset
# (embed is skipped — geometric comparison only)
# =============================================================================

echo "[min-frames] Step 3a: Building maps on subsets ..."
for SUBSET_ROOT in "$ABLATION_ROOT"/*/; do
    NAME=$(basename "$SUBSET_ROOT")
    echo "  === $NAME ==="

    python -m semgraph.stages.build_map \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$SUBSET_ROOT"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames] build_map failed for $NAME (exit $RET). Skipping."
        continue
    fi

    python -m semgraph.stages.oracle_finalize \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$SUBSET_ROOT"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames] oracle_finalize failed for $NAME (exit $RET). Skipping."
        continue
    fi
done
echo ""

# =============================================================================
# Step 3a continued: compare each subset oracle against the reference
# =============================================================================

REF_ORACLE="$ORACLE_DIR/$SCENE/stages/oracle"
mkdir -p "$RESULTS_DIR"

echo "[min-frames] Step 3a: Comparing oracles ..."
for SUBSET_ROOT in "$ABLATION_ROOT"/*/; do
    NAME=$(basename "$SUBSET_ROOT")
    echo "  Comparing: $NAME"
    python -m semgraph.scripts.compare_oracles \
        --reference "$REF_ORACLE" \
        --subset "$SUBSET_ROOT/$SCENE/stages/oracle" \
        --output "$RESULTS_DIR/${NAME}.json"
done
echo ""

# =============================================================================
# Summary table
# =============================================================================

echo "[min-frames] ========== RESULTS =========="
printf "%-20s %8s %8s %8s %8s %8s\n" "Subset" "RefObjs" "SubObjs" "Recall" "Prec" "F1"
printf "%-20s %8s %8s %8s %8s %8s\n" "--------------------" "--------" "--------" "--------" "--------" "--------"

for RESULT_FILE in "$RESULTS_DIR"/*.json; do
    NAME=$(basename "$RESULT_FILE" .json)
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, sys
with open('$RESULT_FILE') as f:
    r = json.load(f)
print(f\"{'$NAME':<20s} {r['ref_n_objects']:>8d} {r['subset_n_objects']:>8d} {r['recall']:>8.3f} {r['precision']:>8.3f} {r['f1']:>8.3f}\")
"
    fi
done

echo ""
echo "[min-frames] Done. Results in $RESULTS_DIR/"

# =============================================================================
# Step 3b: Binary search (manual / iterative)
#
# After reviewing the bracket results above, run additional iterations:
#
#   python -m semgraph.scripts.create_subset \
#       --source "$ORACLE_FD" \
#       --dest "$ABLATION_ROOT/fps_pose_<N>/$SCENE/stages/frame_data" \
#       --method fps_pose --n_frames <N>
#
#   python -m semgraph.stages.build_map \
#       "${COMMON_OVERRIDES[@]}" output_root="$ABLATION_ROOT/fps_pose_<N>"
#
#   python -m semgraph.stages.oracle_finalize \
#       "${COMMON_OVERRIDES[@]}" output_root="$ABLATION_ROOT/fps_pose_<N>"
#
#   python -m semgraph.scripts.compare_oracles \
#       --reference "$REF_ORACLE" \
#       --subset "$ABLATION_ROOT/fps_pose_<N>/$SCENE/stages/oracle" \
#       --output "$RESULTS_DIR/fps_pose_<N>.json"
# =============================================================================
