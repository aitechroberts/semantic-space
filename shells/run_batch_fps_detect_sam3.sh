#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Batch-mode re-run for fps_pose subsets (detect_sam3 + GroundingDINO)
#
# Reuses existing frame_data from the incremental ablation but runs
# build_map in batch matching mode, which finds global connected components
# instead of sequential frame-by-frame matching.  This is the correct mode
# for FPS-selected diverse viewpoints where frame-to-frame overlap is low.
#
# Prerequisite: run_min_frames_detect_sam3.sh must have completed at least
# through subset creation (Step 3a) so that frame_data exists.
# =============================================================================

SCENE="${SCENE:-office0}"
OUTPUT_ROOT="${DETECT_SAM3_OUTPUT:-/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3}"

ORACLE_DIR="$OUTPUT_ROOT"
INCR_ABLATION_ROOT="$OUTPUT_ROOT/ablations"
BATCH_ABLATION_ROOT="$OUTPUT_ROOT/ablations_batch"
RESULTS_DIR="$OUTPUT_ROOT/results_batch"

COMMON_OVERRIDES=(
    "scene_id=$SCENE"
    "dataset_root=${DATASET_ROOT:-/home/jrob/cmu-grad/neuro-data/Replica}"
    "exp_suffix=min_frames"
    "segmentation_backend=detect_sam3"
    "detector_type=${DETECTOR_TYPE:-gdino}"
    "downsample_voxel_size=0.025"
)

REF_ORACLE="$ORACLE_DIR/$SCENE/stages/oracle"

# Verify reference oracle exists
if [[ ! -f "$REF_ORACLE/oracle_scene.npz" ]]; then
    echo "[batch-fps] ERROR: Reference oracle not found at $REF_ORACLE"
    echo "[batch-fps] Run run_min_frames_detect_sam3.sh first."
    exit 1
fi

# =============================================================================
# For each fps_pose subset: copy frame_data, run batch build_map, finalize
# =============================================================================

for N in 200 50; do
    NAME="fps_pose_${N}"
    SRC_FD="$INCR_ABLATION_ROOT/$NAME/$SCENE/stages/frame_data"
    DEST_ROOT="$BATCH_ABLATION_ROOT/$NAME"
    DEST_FD="$DEST_ROOT/$SCENE/stages/frame_data"
    DEST_MAP="$DEST_ROOT/$SCENE/stages/map/oracle_map.pkl.gz"
    DEST_ORACLE="$DEST_ROOT/$SCENE/stages/oracle/oracle_scene.npz"

    # Check source frame_data exists
    if [[ ! -d "$SRC_FD" ]]; then
        echo "[batch-fps] ERROR: frame_data not found for $NAME at $SRC_FD"
        echo "[batch-fps] Run run_min_frames_detect_sam3.sh first to create subsets."
        continue
    fi

    echo "[batch-fps] ========== $NAME (batch mode) =========="

    # Symlink frame_data instead of copying
    if [[ ! -d "$DEST_FD" ]]; then
        mkdir -p "$(dirname "$DEST_FD")"
        ln -sfn "$SRC_FD" "$DEST_FD"
        echo "  Linked frame_data from incremental run"
    fi

    # build_map (batch mode)
    if [[ -f "$DEST_MAP" ]]; then
        echo "  build_map already done, skipping"
    else
        echo "  Running build_map (matching_mode=batch) ..."
        python -m semgraph.stages.build_map \
            "${COMMON_OVERRIDES[@]}" \
            "output_root=$DEST_ROOT" \
            "build_map.matching_mode=batch"
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "[batch-fps] build_map failed for $NAME (exit $RET). Skipping."
            continue
        fi
    fi

    # oracle_finalize
    if [[ -f "$DEST_ORACLE" ]]; then
        echo "  oracle already exists, skipping"
    else
        echo "  Running oracle_finalize ..."
        python -m semgraph.stages.oracle_finalize \
            "${COMMON_OVERRIDES[@]}" \
            "output_root=$DEST_ROOT"
        RET=$?
        if [[ $RET -ne 0 ]]; then
            echo "[batch-fps] oracle_finalize failed for $NAME (exit $RET). Skipping."
            continue
        fi
    fi
done

echo ""

# =============================================================================
# Compare each batch oracle against the reference
# =============================================================================

mkdir -p "$RESULTS_DIR"

echo "[batch-fps] Comparing oracles ..."
for SUBSET_ROOT in "$BATCH_ABLATION_ROOT"/*/; do
    NAME=$(basename "$SUBSET_ROOT")
    SUBSET_ORACLE="$SUBSET_ROOT/$SCENE/stages/oracle"
    if [[ ! -f "$SUBSET_ORACLE/oracle_scene.npz" ]]; then
        echo "  $NAME: no oracle found, skipping comparison"
        continue
    fi
    echo "  Comparing: $NAME"
    python -m semgraph.scripts.compare_oracles \
        --reference "$REF_ORACLE" \
        --subset "$SUBSET_ORACLE" \
        --output "$RESULTS_DIR/${NAME}.json"
done

echo ""

# =============================================================================
# Summary: batch results alongside incremental for direct comparison
# =============================================================================

echo "[batch-fps] ========== RESULTS (batch vs incremental) =========="
printf "%-30s %8s %8s %8s %8s %8s %5s\n" "Subset" "RefObjs" "SubObjs" "Recall" "Prec" "F1" "Frag"
printf "%-30s %8s %8s %8s %8s %8s %5s\n" "------------------------------" "--------" "--------" "--------" "--------" "--------" "-----"

# Print incremental results first (from original run)
for NAME in fps_pose_200 fps_pose_50; do
    INCR_FILE="$OUTPUT_ROOT/results/${NAME}.json"
    if [[ -f "$INCR_FILE" ]]; then
        python3 -c "
import json
with open('$INCR_FILE') as f:
    r = json.load(f)
print(f\"{'${NAME} (incremental)':<30s} {r['ref_n_objects']:>8d} {r['subset_n_objects']:>8d} {r['recall']:>8.3f} {r['precision']:>8.3f} {r['f1']:>8.3f} {r.get('fragmentation_count',0):>5d}\")
"
    fi
done

# Print batch results
for NAME in fps_pose_200 fps_pose_50; do
    BATCH_FILE="$RESULTS_DIR/${NAME}.json"
    if [[ -f "$BATCH_FILE" ]]; then
        python3 -c "
import json
with open('$BATCH_FILE') as f:
    r = json.load(f)
print(f\"{'${NAME} (batch)':<30s} {r['ref_n_objects']:>8d} {r['subset_n_objects']:>8d} {r['recall']:>8.3f} {r['precision']:>8.3f} {r['f1']:>8.3f} {r.get('fragmentation_count',0):>5d}\")
"
    fi
done

echo ""
echo "[batch-fps] Done. Results in $RESULTS_DIR/"