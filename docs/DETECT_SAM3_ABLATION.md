# Detect-SAM3 + GroundingDINO Minimum-Frames Ablation — Runbook

> **Last updated:** 2026-04-14
>
> Step-by-step commands for running the minimum-frames ablation experiment
> with `detect_sam3` + GroundingDINO on Replica scenes.  For background on
> the tools used here, see [BATCH_MATCHING.md](BATCH_MATCHING.md) and
> [STAGE_DETECTION.md](STAGE_DETECTION.md).

---

## Prerequisites

- Replica dataset at `$DATASET_ROOT` (default: `/home/jrob/cmu-grad/neuro-data/Replica`)
- GroundingDINO weights (`IDEA-Research/grounding-dino-base`) — auto-downloaded from HuggingFace on first run
- SAM 3 weights (`sam3.pt`) reachable via `$CKPT_DIR` — must be manually downloaded from [HuggingFace](https://huggingface.co/facebook/sam3)
- GPU with enough VRAM for SAM 3 (~3.5 GB) + GroundingDINO (~900 MB) + CLIP (~400 MB)
- All commands run from the repo root (`neuro-nav/`)

```bash
export DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica
export OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3
export SCENE=office0
```

### Key differences from the SAM-Auto ablation

| | SAM-Auto | Detect-SAM3 + GDino |
|---|---|---|
| `segmentation_backend` | `sam_auto` | `detect_sam3` |
| `detector_type` | N/A | `gdino` |
| Segmenter | SAM 2.1 auto mode | SAM 3 box-prompted |
| Detector | None | GroundingDINO (scannet200 vocab) |
| VRAM (detect) | ~162 MB | ~4.4 GB (SAM 3 + GDino) |
| Class labels | None (class-agnostic) | From scannet200_classes.txt |
| Majority-vote relabeling | No | Yes (build_map.py) |

---

## Option A: Run Everything at Once

The orchestration script runs all steps end-to-end for one scene:

```bash
SCENE=office0 \
DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica \
OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3 \
  bash shells/run_min_frames_detect_sam3.sh
```

This will:
1. Run stride=1 detection with `detect_sam3` + `gdino` + embed + build_map + oracle_finalize (reference)
2. Create 6 subsets (200/50/10 frames x stride/fps_pose)
3. Build maps and oracle-finalize each subset
4. Compare each subset against the reference
5. Print a summary table

To override the output location or detector:

```bash
# Custom output directory (uses DETECT_SAM3_OUTPUT, not OUTPUT_ROOT,
# so a stale OUTPUT_ROOT from another ablation can't collide)
DETECT_SAM3_OUTPUT=/tmp/my-test SCENE=office0 \
  bash shells/run_min_frames_detect_sam3.sh

# Test YOLOE instead of GDino
DETECTOR_TYPE=yoloe SCENE=office0 \
  bash shells/run_min_frames_detect_sam3.sh
```

To run multiple scenes, loop:

```bash
for SCENE in office0 office1 office2 office3 office4 room0 room1 room2; do
  SCENE=$SCENE \
  DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica \
  OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3 \
    bash shells/run_min_frames_detect_sam3.sh
done
```

---

## Option B: Run Steps Manually

Use this when you want to resume from a checkpoint or run individual
steps.

### Environment

```bash
export DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica
export OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3
export SCENE=office0

ORACLE_DIR="$OUTPUT_ROOT"
ABLATION_ROOT="$OUTPUT_ROOT/ablations"
RESULTS_DIR="$OUTPUT_ROOT/results"

COMMON=(
    scene_id=$SCENE
    dataset_root=$DATASET_ROOT
    exp_suffix=min_frames
    segmentation_backend=detect_sam3
    detector_type=gdino
)
```

### Step 1: Reference detection (stride=1, expensive)

```bash
python -m semgraph.stages.detect \
    "${COMMON[@]}" \
    output_root=$ORACLE_DIR \
    stride=1
```

This runs GroundingDINO + SAM 3 on every frame.  The detector vocabulary
comes from `scannet200_classes.txt` (via `classes.yaml`).
`save_raw_detections` defaults to `false` — set `save_raw_detections=true`
to inspect pre-filter masks.

### Step 2: Embed (CLIP features for matching)

```bash
python -m semgraph.stages.embed \
    "${COMMON[@]}" \
    output_root=$ORACLE_DIR
```

### Step 3: Build map (incremental matching)

```bash
python -m semgraph.stages.build_map \
    "${COMMON[@]}" \
    output_root=$ORACLE_DIR
```

Because this is a detector-first backend (`detect_sam3`), `build_map.py`
runs majority-vote class relabeling — each object's `class_name` is
reassigned to the class that appeared most often across its detections.
This only works correctly with vocab-driven detectors (GroundingDINO,
YOLOE, YOLO-World) where `class_id` indexes into the global vocabulary.

To use batch matching instead (for sparse/DUSt3R data):

```bash
python -m semgraph.stages.build_map \
    "${COMMON[@]}" \
    output_root=$ORACLE_DIR \
    build_map.matching_mode=batch
```

### Step 4: Oracle finalize

```bash
python -m semgraph.stages.oracle_finalize \
    "${COMMON[@]}" \
    output_root=$ORACLE_DIR
```

The reference oracle is now at `$ORACLE_DIR/$SCENE/stages/oracle/`.

### Step 5: Create subsets

```bash
ORACLE_FD="$ORACLE_DIR/$SCENE/stages/frame_data"

# Initial bracket: 200, 50, 10 frames x stride and fps_pose
for N in 200 50 10; do
  for METHOD in stride fps_pose; do
    python -m semgraph.scripts.create_subset \
        --source "$ORACLE_FD" \
        --dest "$ABLATION_ROOT/${METHOD}_${N}/$SCENE/stages/frame_data" \
        --method "$METHOD" --n_frames "$N"
  done
done
```

Or use the subset-only script (skips detect, starts from existing frame_data):

```bash
SCENE=office0 \
OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3 \
  bash shells/build_subsets_detect_sam3.sh
```

Preview without creating files:

```bash
python -m semgraph.scripts.create_subset \
    --source "$ORACLE_FD" \
    --dest "$ABLATION_ROOT/fps_pose_30/$SCENE/stages/frame_data" \
    --method fps_pose --n_frames 30 --dry_run
```

### Step 6: Build map + oracle_finalize on each subset

```bash
for SUBSET_DIR in "$ABLATION_ROOT"/*/; do
  NAME=$(basename "$SUBSET_DIR")
  echo "=== $NAME ==="
  python -m semgraph.stages.build_map \
      "${COMMON[@]}" output_root="$SUBSET_DIR"
  python -m semgraph.stages.oracle_finalize \
      "${COMMON[@]}" output_root="$SUBSET_DIR"
done
```

### Step 7: Compare oracles

```bash
REF_ORACLE="$ORACLE_DIR/$SCENE/stages/oracle"
mkdir -p "$RESULTS_DIR"

for SUBSET_DIR in "$ABLATION_ROOT"/*/; do
  NAME=$(basename "$SUBSET_DIR")
  python -m semgraph.scripts.compare_oracles \
      --reference "$REF_ORACLE" \
      --subset "$SUBSET_DIR/$SCENE/stages/oracle" \
      --output "$RESULTS_DIR/${NAME}.json"
done
```

### Step 8: Review results

Each JSON in `$RESULTS_DIR/` contains: `recall`, `precision`, `f1`,
`mean_matched_iou`, `fragmentation_count`, `ref_n_objects`,
`subset_n_objects`.

Quick summary:

```bash
for f in "$RESULTS_DIR"/*.json; do
  NAME=$(basename "$f" .json)
  python3 -c "
import json
with open('$f') as fh:
    r = json.load(fh)
print(f'${NAME:<20s}  recall={r[\"recall\"]:.3f}  prec={r[\"precision\"]:.3f}  F1={r[\"f1\"]:.3f}  meanIoU={r[\"mean_matched_iou\"]:.3f}')
"
done
```

---

## Step 3b: Binary Search (Manual)

After reviewing the bracket results (200/50/10), narrow down:

```bash
# Example: test 100 frames with fps_pose
N=100
python -m semgraph.scripts.create_subset \
    --source "$ORACLE_FD" \
    --dest "$ABLATION_ROOT/fps_pose_${N}/$SCENE/stages/frame_data" \
    --method fps_pose --n_frames $N

python -m semgraph.stages.build_map \
    "${COMMON[@]}" output_root="$ABLATION_ROOT/fps_pose_${N}"

python -m semgraph.stages.oracle_finalize \
    "${COMMON[@]}" output_root="$ABLATION_ROOT/fps_pose_${N}"

python -m semgraph.scripts.compare_oracles \
    --reference "$REF_ORACLE" \
    --subset "$ABLATION_ROOT/fps_pose_${N}/$SCENE/stages/oracle" \
    --output "$RESULTS_DIR/fps_pose_${N}.json"
```

Repeat with different N values until you find the knee in the
recall-vs-frame-count curve.

---

## Directory Layout After Completion

```
$OUTPUT_ROOT/
├── $SCENE/stages/
│   ├── frame_data/          # stride=1 reference (all frames)
│   ├── crops/               # 1.5x projected crops
│   ├── map/                 # oracle_map.pkl.gz
│   └── oracle/              # oracle_scene.npz + .json
├── ablations/
│   ├── stride_200/$SCENE/stages/{frame_data,map,oracle}/
│   ├── fps_pose_200/$SCENE/stages/{frame_data,map,oracle}/
│   ├── stride_50/...
│   ├── fps_pose_50/...
│   ├── stride_10/...
│   └── fps_pose_10/...
└── results/
    ├── stride_200.json
    ├── fps_pose_200.json
    ├── stride_50.json
    ├── fps_pose_50.json
    ├── stride_10.json
    └── fps_pose_10.json
```

Subset `frame_data/` directories contain symlinks to the reference, not
copies.  The `_selection_metadata.json` in each records which frames were
selected and why.

---

## Comparing Against SAM-Auto Results

If you have already run the SAM-Auto ablation (see
[SAM_AUTO_ABLATION.md](SAM_AUTO_ABLATION.md)), you can cross-compare the
oracles directly:

```bash
SAM_AUTO_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-sam-auto
DETECT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-detect-sam3

# Compare detect_sam3 reference against sam_auto reference
python -m semgraph.scripts.compare_oracles \
    --reference "$SAM_AUTO_ROOT/$SCENE/stages/oracle" \
    --subset "$DETECT_ROOT/$SCENE/stages/oracle" \
    --output "$DETECT_ROOT/results/vs_sam_auto_reference.json"
```

This tells you how much the detector-first pipeline diverges from the
class-agnostic SAM-Auto pipeline on the same scene at full frame count,
before any subsampling.

---

## Key Config Parameters

| Parameter | Default | Where | Effect |
|-----------|---------|-------|--------|
| `segmentation_backend` | `detect_sam3` | CLI override | GroundingDINO + SAM 3 box-prompted |
| `detector_type` | `gdino` | CLI override / `base_mapping.yaml` | GroundingDINO detector |
| `detector_name` | `null` | `base_mapping.yaml` | Uses default `IDEA-Research/grounding-dino-base` |
| `stride` | `10` | `batch_vlm_mapping_api.yaml` | Frame stride for detect.py (set to 1 for reference) |
| `save_raw_detections` | `false` | `base_mapping.yaml` | Skip writing raw_detections/ to save disk |
| `build_map.matching_mode` | `incremental` | `batch_vlm_mapping_api.yaml` | `incremental` or `batch` |
| `sim_threshold` | `1.2` | `base_mapping.yaml` | Aggregated similarity threshold for matching |
| `iou_merge_kappa` | `0.25` | `base_mapping.yaml` | 3D bbox IoU fallback for merging |
| `mask_conf_threshold` | `0.25` | `base_mapping.yaml` | Confidence floor (GDino box_threshold in code is 0.3) |
| `--iou_threshold` | `0.25` | `compare_oracles.py` CLI | 3D bbox IoU threshold for evaluation matching |
