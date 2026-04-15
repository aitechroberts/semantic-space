#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Build 6 ablation subsets from an existing stride=1 detect run.
# Creates symlinked frame_data, then runs embed + build_map + oracle_finalize
# on each subset independently.
#
# Prerequisites: detect.py stride=1 must have already completed.
#
# Usage:
#   SCENE=office0 bash shells/build_subsets.sh
# =============================================================================

SCENE="${SCENE:-office0}"
DATASET_ROOT="${DATASET_ROOT:-/home/jrob/cmu-grad/neuro-data/Replica}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/jrob/cmu-grad/neuro-experiments/min-frames-sam-auto}"

ORACLE_FD="$OUTPUT_ROOT/$SCENE/stages/frame_data"
ABLATION_ROOT="$OUTPUT_ROOT/ablations"

COMMON=(
    "scene_id=$SCENE"
    "dataset_root=$DATASET_ROOT"
    "exp_suffix=min_frames"
)

# --- Verify detect output exists ---
if [[ ! -d "$ORACLE_FD" ]]; then
    echo "Error: frame_data not found at $ORACLE_FD"
    echo "Run detect.py with stride=1 first."
    exit 1
fi

# --- Create 6 subsets (symlinks only, no overwrites) ---
BRACKET_COUNTS=(50 10)

echo "[subsets] Creating subsets from $ORACLE_FD ..."
for N in "${BRACKET_COUNTS[@]}"; do
    for METHOD in stride fps_pose; do
        NAME="${METHOD}_${N}"
        DEST="$ABLATION_ROOT/$NAME/$SCENE/stages/frame_data"
        if [[ -d "$DEST" ]]; then
            echo "  $NAME: already exists, skipping create_subset"
            continue
        fi
        echo "  $NAME: creating"
        python -m semgraph.scripts.create_subset \
            --source "$ORACLE_FD" \
            --dest "$DEST" \
            --method "$METHOD" --n_frames "$N"
    done
done
echo ""

# --- Embed + build_map + oracle_finalize on each subset ---
for SUBSET_DIR in "$ABLATION_ROOT"/*/; do
    NAME=$(basename "$SUBSET_DIR")
    ORACLE_OUT="$SUBSET_DIR/$SCENE/stages/oracle"

    if [[ -f "$ORACLE_OUT/oracle_scene.npz" ]]; then
        echo "  $NAME: oracle already exists, skipping"
        continue
    fi

    echo "=== $NAME ==="

    MAP_OUT="$SUBSET_DIR/$SCENE/stages/map/oracle_map.pkl.gz"

    # --- embed (skip if build_map output already exists) ---
    if [[ -f "$MAP_OUT" ]]; then
        echo "  $NAME: embed already done (map exists), skipping"
    else
        python -m semgraph.stages.embed \
            "${COMMON[@]}" "output_root=$SUBSET_DIR"
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "  embed failed for $NAME (exit $RET). Skipping subset."
            continue
        fi
    fi

    # --- build_map (skip if map output already exists) ---
    if [[ -f "$MAP_OUT" ]]; then
        echo "  $NAME: build_map already done, skipping"
    else
        python -m semgraph.stages.build_map \
            "${COMMON[@]}" "output_root=$SUBSET_DIR"
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "  build_map failed for $NAME (exit $RET). Skipping subset."
            continue
        fi
    fi

    # --- oracle_finalize ---
    python -m semgraph.stages.oracle_finalize \
        "${COMMON[@]}" "output_root=$SUBSET_DIR"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "  oracle_finalize failed for $NAME (exit $RET). Skipping subset."
        continue
    fi
done
echo ""

echo "[subsets] Done. Results in $ABLATION_ROOT/"
ls -1d "$ABLATION_ROOT"/*/