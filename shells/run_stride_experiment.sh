#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Tunable stride experiment runner for detect_sam3 + GroundingDINO
#
# Reuses the oracle frame_data from run_min_frames_detect_sam3.sh.
# Each parameter combination gets its own output folder under ablations/.
#
# Usage:
#   SCENE=room0 STRIDE=10 bash shells/run_stride_experiment.sh
#   SCENE=room0 STRIDE=17 SIM_THRESH=1.0 PHYS_BIAS=0.3 bash shells/run_stride_experiment.sh
# =============================================================================

# --- Scene / paths ---
SCENE="${SCENE:-room0}"
DATASET_ROOT="${DATASET_ROOT:-/home/jrob/cmu-grad/neuro-data/Replica}"
OUTPUT_ROOT="${DETECT_SAM3_OUTPUT:-/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3}"

# --- Frame selection ---
STRIDE="${STRIDE:-10}"

# --- Matching mode ---
MATCHING_MODE="${MATCHING_MODE:-incremental}"   # "incremental" or "batch"

# --- Stage 1: Detection-to-object matching ---
SIM_THRESH="${SIM_THRESH:-1.2}"
PHYS_BIAS="${PHYS_BIAS:-0.0}"
IOU_MERGE_KAPPA="${IOU_MERGE_KAPPA:-0.25}"

# --- Stage 2: Post-hoc overlap merging ---
MERGE_OVERLAP_THRESH="${MERGE_OVERLAP_THRESH:-0.7}"
MERGE_VISUAL_SIM_THRESH="${MERGE_VISUAL_SIM_THRESH:-0.7}"
MERGE_TEXT_SIM_THRESH="${MERGE_TEXT_SIM_THRESH:-0.7}"
MERGE_INTERVAL="${MERGE_INTERVAL:-5}"

# --- Stage 3: Object filtering ---
OBJ_MIN_DETECTIONS="${OBJ_MIN_DETECTIONS:-1}"
OBJ_MIN_POINTS="${OBJ_MIN_POINTS:-0}"
AUTO_FILTER_PRESET="${AUTO_FILTER_PRESET:-null}"

# --- DBSCAN ---
DBSCAN_EPS="${DBSCAN_EPS:-0.1}"

# =============================================================================
# Derived names — each unique config gets its own folder
# =============================================================================

# Compute effective frame count: total frames / stride
ORACLE_FD="$OUTPUT_ROOT/$SCENE/stages/frame_data"
if [[ -d "$ORACLE_FD" ]]; then
    TOTAL_FRAMES=$(find "$ORACLE_FD" -name '*.npz' 2>/dev/null | wc -l)
else
    echo "[experiment] ERROR: Oracle frame_data not found at $ORACLE_FD"
    echo "[experiment] Run run_min_frames_detect_sam3.sh first."
    exit 1
fi
N_FRAMES=$(( TOTAL_FRAMES / STRIDE ))

# Build a human-readable experiment tag from non-default parameters
TAG="stride_${STRIDE}"
[[ "$SIM_THRESH"           != "1.2"  ]] && TAG="${TAG}_st${SIM_THRESH}"
[[ "$PHYS_BIAS"            != "0.0"  ]] && TAG="${TAG}_pb${PHYS_BIAS}"
[[ "$IOU_MERGE_KAPPA"      != "0.25" ]] && TAG="${TAG}_imk${IOU_MERGE_KAPPA}"
[[ "$MERGE_OVERLAP_THRESH" != "0.7"  ]] && TAG="${TAG}_mot${MERGE_OVERLAP_THRESH}"
[[ "$MERGE_VISUAL_SIM_THRESH" != "0.7" ]] && TAG="${TAG}_mvs${MERGE_VISUAL_SIM_THRESH}"
[[ "$MERGE_INTERVAL"       != "5"    ]] && TAG="${TAG}_mi${MERGE_INTERVAL}"
[[ "$OBJ_MIN_DETECTIONS"   != "1"    ]] && TAG="${TAG}_md${OBJ_MIN_DETECTIONS}"
[[ "$DBSCAN_EPS"           != "0.1"  ]] && TAG="${TAG}_eps${DBSCAN_EPS}"
[[ "$MATCHING_MODE"        != "incremental" ]] && TAG="${TAG}_${MATCHING_MODE}"
[[ "$AUTO_FILTER_PRESET"   != "null" ]] && TAG="${TAG}_${AUTO_FILTER_PRESET}"

ABLATION_ROOT="$OUTPUT_ROOT/ablations"
SUBSET_ROOT="$ABLATION_ROOT/$TAG"
DEST_FD="$SUBSET_ROOT/$SCENE/stages/frame_data"
DEST_MAP="$SUBSET_ROOT/$SCENE/stages/map/oracle_map.pkl.gz"
DEST_ORACLE="$SUBSET_ROOT/$SCENE/stages/oracle/oracle_scene.npz"

REF_ORACLE="$OUTPUT_ROOT/$SCENE/stages/oracle"
RESULTS_DIR="$OUTPUT_ROOT/results"

# =============================================================================
# Print config summary
# =============================================================================

echo "============================================================"
echo " Experiment: $TAG"
echo "============================================================"
echo " Scene:              $SCENE"
echo " Stride:             $STRIDE  (~$N_FRAMES frames from $TOTAL_FRAMES)"
echo " Matching mode:      $MATCHING_MODE"
echo ""
echo " Stage 1 (matching):"
echo "   sim_threshold:    $SIM_THRESH"
echo "   phys_bias:        $PHYS_BIAS"
echo "   iou_merge_kappa:  $IOU_MERGE_KAPPA"
echo ""
echo " Stage 2 (overlap merge):"
echo "   merge_overlap:    $MERGE_OVERLAP_THRESH"
echo "   merge_visual_sim: $MERGE_VISUAL_SIM_THRESH"
echo "   merge_text_sim:   $MERGE_TEXT_SIM_THRESH"
echo "   merge_interval:   $MERGE_INTERVAL"
echo ""
echo " Stage 3 (filtering):"
echo "   obj_min_det:      $OBJ_MIN_DETECTIONS"
echo "   obj_min_points:   $OBJ_MIN_POINTS"
echo "   auto_preset:      $AUTO_FILTER_PRESET"
echo "   dbscan_eps:       $DBSCAN_EPS"
echo ""
echo " Output: $SUBSET_ROOT"
echo "============================================================"

# =============================================================================
# Common Hydra overrides
# =============================================================================

COMMON_OVERRIDES=(
    "scene_id=$SCENE"
    "dataset_root=$DATASET_ROOT"
    "exp_suffix=min_frames"
    "segmentation_backend=detect_sam3"
    "detector_type=${DETECTOR_TYPE:-gdino}"
    "downsample_voxel_size=0.025"
    "sim_threshold=$SIM_THRESH"
    "phys_bias=$PHYS_BIAS"
    "iou_merge_kappa=$IOU_MERGE_KAPPA"
    "merge_overlap_thresh=$MERGE_OVERLAP_THRESH"
    "merge_visual_sim_thresh=$MERGE_VISUAL_SIM_THRESH"
    "merge_text_sim_thresh=$MERGE_TEXT_SIM_THRESH"
    "merge_interval=$MERGE_INTERVAL"
    "obj_min_detections=$OBJ_MIN_DETECTIONS"
    "obj_min_points=$OBJ_MIN_POINTS"
    "dbscan_eps=$DBSCAN_EPS"
    "build_map.matching_mode=$MATCHING_MODE"
)

[[ "$AUTO_FILTER_PRESET" != "null" ]] && COMMON_OVERRIDES+=("auto_filter_preset=$AUTO_FILTER_PRESET")

# =============================================================================
# Step 1: Create subset (stride-based, reuses oracle frame_data)
# =============================================================================

if [[ -d "$DEST_FD" ]] && [[ $(find "$DEST_FD" -name '*.npz' 2>/dev/null | head -1) ]]; then
    echo "[experiment] Subset frame_data exists, skipping create_subset"
else
    echo "[experiment] Creating stride_${STRIDE} subset (~$N_FRAMES frames) ..."
    python -m semgraph.scripts.create_subset \
        --source "$ORACLE_FD" \
        --dest "$DEST_FD" \
        --method stride --n_frames "$N_FRAMES"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[experiment] create_subset failed (exit $RET). Aborting."
        exit $RET
    fi
fi

# =============================================================================
# Step 2: build_map
# =============================================================================

if [[ -f "$DEST_MAP" ]]; then
    echo "[experiment] build_map already done, skipping"
else
    echo "[experiment] Running build_map ($MATCHING_MODE) ..."
    python -m semgraph.stages.build_map \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$SUBSET_ROOT"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[experiment] build_map failed (exit $RET). Aborting."
        exit $RET
    fi
fi

# =============================================================================
# Step 3: oracle_finalize
# =============================================================================

if [[ -f "$DEST_ORACLE" ]]; then
    echo "[experiment] oracle already exists, skipping"
else
    echo "[experiment] Running oracle_finalize ..."
    python -m semgraph.stages.oracle_finalize \
        "${COMMON_OVERRIDES[@]}" \
        "output_root=$SUBSET_ROOT"
    RET=$?
    if [[ $RET -ne 0 ]]; then
        echo "[experiment] oracle_finalize failed (exit $RET). Aborting."
        exit $RET
    fi
fi

# =============================================================================
# Step 4: Compare against reference oracle
# =============================================================================

if [[ ! -f "$REF_ORACLE/oracle_scene.npz" ]]; then
    echo "[experiment] WARNING: Reference oracle not found, skipping comparison"
else
    mkdir -p "$RESULTS_DIR"
    echo "[experiment] Comparing against reference oracle ..."
    python -m semgraph.scripts.compare_oracles \
        --reference "$REF_ORACLE" \
        --subset "$SUBSET_ROOT/$SCENE/stages/oracle" \
        --output "$RESULTS_DIR/${TAG}.json"
fi

echo ""
echo "[experiment] Done: $TAG"