#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Minimum-frames ablation: detect_sam3 + GroundingDINO
#
# Runs a full-scene oracle (stride=1), creates subsets at various frame
# counts via stride and fps_pose selectors, builds maps on each subset,
# and compares the resulting oracles geometrically.
#
# Detector vocabulary comes from scannet200_classes.txt via classes.yaml.
#
# Override output location with DETECT_SAM3_OUTPUT (not OUTPUT_ROOT, to
# avoid collision with other ablation scripts sharing the same env).
# =============================================================================

# --- Configuration ---
SCENE="${SCENE:-office0}"
DATASET_ROOT="${DATASET_ROOT:-/home/jrob/cmu-grad/neuro-data/Replica}"
OUTPUT_ROOT="${DETECT_SAM3_OUTPUT:-/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3}"
DETECTOR_TYPE="${DETECTOR_TYPE:-gdino}"

ORACLE_DIR="$OUTPUT_ROOT"
ABLATION_ROOT="$OUTPUT_ROOT/ablations"
RESULTS_DIR="$OUTPUT_ROOT/results"

COMMON_OVERRIDES=(
    "scene_id=$SCENE"
    "dataset_root=$DATASET_ROOT"
    "exp_suffix=min_frames"
    "segmentation_backend=detect_sam3"
    "detector_type=$DETECTOR_TYPE"
    "downsample_voxel_size=0.025"
)

# =============================================================================
# Derived paths for skip guards
# =============================================================================

ORACLE_FD="$ORACLE_DIR/$SCENE/stages/frame_data"
EMBED_STAMP="$ORACLE_DIR/$SCENE/stages/.embed_done"
ORACLE_MAP="$ORACLE_DIR/$SCENE/stages/map/oracle_map.pkl.gz"
ORACLE_SCENE="$ORACLE_DIR/$SCENE/stages/oracle/oracle_scene.npz"

# =============================================================================
# Step 1: Oracle detection (stride=1, run once — expensive)
# =============================================================================

if [[ -d "$ORACLE_FD" ]] && [[ $(find "$ORACLE_FD" -name '*.npz' 2>/dev/null | head -1) ]]; then
    echo "[min-frames-detect] Step 1: frame_data already exists, skipping detect"
else
    echo "[min-frames-detect] Step 1: Oracle detection (stride=1, detect_sam3 + $DETECTOR_TYPE)"
    python -m semgraph.stages.detect \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$ORACLE_DIR" \
        stride=1
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames-detect] detect.py failed (exit $RET). Aborting."
        exit $RET
    fi
fi

# =============================================================================
# Step 2: Full Phase A on stride=1 (embed, build_map, oracle_finalize)
# =============================================================================

if [[ -f "$EMBED_STAMP" ]]; then
    echo "[min-frames-detect] Step 2a: embed already done, skipping"
else
    echo "[min-frames-detect] Step 2a: Embed (oracle encoder features)"
    python -m semgraph.stages.embed \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$ORACLE_DIR"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames-detect] embed.py failed (exit $RET). Aborting."
        exit $RET
    fi
    touch "$EMBED_STAMP"
fi

if [[ -f "$ORACLE_MAP" ]]; then
    echo "[min-frames-detect] Step 2b: build_map already done, skipping"
else
    echo "[min-frames-detect] Step 2b: Build map"
    python -m semgraph.stages.build_map \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$ORACLE_DIR"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames-detect] build_map.py failed (exit $RET). Aborting."
        exit $RET
    fi
fi

if [[ -f "$ORACLE_SCENE" ]]; then
    echo "[min-frames-detect] Step 2c: oracle already exists, skipping"
else
    echo "[min-frames-detect] Step 2c: Oracle finalize"
    python -m semgraph.stages.oracle_finalize \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$ORACLE_DIR"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames-detect] oracle_finalize.py failed (exit $RET). Aborting."
        exit $RET
    fi
fi

echo "[min-frames-detect] Reference oracle complete."
echo ""

# =============================================================================
# Step 3a: Create subsets (initial bracket)
# =============================================================================

BRACKET_COUNTS=(200 50 10)

echo "[min-frames-detect] Step 3a: Creating subsets ..."
for N in "${BRACKET_COUNTS[@]}"; do
    for METHOD in stride fps_pose; do
        NAME="${METHOD}_${N}"
        SUBSET_ROOT="$ABLATION_ROOT/$NAME"
        DEST="$SUBSET_ROOT/$SCENE/stages/frame_data"
        if [[ -d "$DEST" ]]; then
            echo "  $NAME: already exists, skipping create_subset"
            continue
        fi
        echo "  Creating subset: $NAME"
        python -m semgraph.scripts.create_subset \
            --source "$ORACLE_FD" \
            --dest "$DEST" \
            --method "$METHOD" --n_frames "$N"
    done
done
echo ""

# =============================================================================
# Step 3a continued: build_map + oracle_finalize on each subset
# =============================================================================

echo "[min-frames-detect] Step 3a: Building maps on subsets ..."
for SUBSET_ROOT in "$ABLATION_ROOT"/*/; do
    NAME=$(basename "$SUBSET_ROOT")
    SUBSET_ORACLE="$SUBSET_ROOT/$SCENE/stages/oracle/oracle_scene.npz"

    if [[ -f "$SUBSET_ORACLE" ]]; then
        echo "  $NAME: oracle already exists, skipping"
        continue
    fi

    echo "  === $NAME ==="

    SUBSET_MAP="$SUBSET_ROOT/$SCENE/stages/map/oracle_map.pkl.gz"

    if [[ -f "$SUBSET_MAP" ]]; then
        echo "  $NAME: build_map already done, skipping"
    else
        python -m semgraph.stages.build_map \
            "${COMMON_OVERRIDES[@]}" \
            "output_root=$SUBSET_ROOT"
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "[min-frames-detect] build_map failed for $NAME (exit $RET). Skipping."
            continue
        fi
    fi

    python -m semgraph.stages.oracle_finalize \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$SUBSET_ROOT"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[min-frames-detect] oracle_finalize failed for $NAME (exit $RET). Skipping."
        continue
    fi
done
echo ""

# =============================================================================
# Step 3a continued: compare each subset oracle against the reference
# =============================================================================

REF_ORACLE="$ORACLE_DIR/$SCENE/stages/oracle"
mkdir -p "$RESULTS_DIR"

echo "[min-frames-detect] Step 3a: Comparing oracles ..."
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

echo "[min-frames-detect] ========== RESULTS =========="
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
echo "[min-frames-detect] Done. Results in $RESULTS_DIR/"

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
