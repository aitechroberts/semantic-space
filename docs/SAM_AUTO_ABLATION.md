# SAM-Auto Minimum-Frames Ablation — Runbook

> **Last updated:** 2026-04-13
>
> Step-by-step commands for running the minimum-frames ablation experiment
> with `sam_auto` on Replica scenes.  For background on the tools used here,
> see [BATCH_MATCHING.md](BATCH_MATCHING.md) and
> [STAGE_DETECTION.md](STAGE_DETECTION.md).

---

## Prerequisites

- Replica dataset at `$DATASET_ROOT` (default: `/home/jrob/cmu-grad/neuro-data/Replica`)
- SAM 2.1 weights (`sam2.1_b.pt`) reachable via `$CKPT_DIR` or auto-download
- GPU with enough VRAM for SAM 2.1 (~162 MB) + CLIP (~400 MB)
- All commands run from the repo root (`neuro-nav/`)

```bash
export DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica
export OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-sam-auto
export SCENE=office0
```

---

## Option A: Run Everything at Once

The orchestration script runs all steps end-to-end for one scene.
Override `SEG_BACKEND` to use `sam_auto` instead of the default `sam3_auto`:

```bash
SEG_BACKEND=sam_auto \
SCENE=office0 \
DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica \
OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-sam-auto \
  bash shells/run_min_frames_experiment.sh
```

This will:
1. Run stride=1 detection + embed + build_map + oracle_finalize (reference)
2. Create 6 subsets (200/50/10 frames x stride/fps_pose)
3. Build maps and oracle-finalize each subset
4. Compare each subset against the reference
5. Print a summary table

To run multiple scenes, loop:

```bash
for SCENE in office0 office1 office2 office3 office4 room0 room1 room2; do
  SEG_BACKEND=sam_auto SCENE=$SCENE \
  DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica \
  OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-sam-auto \
    bash shells/run_min_frames_experiment.sh
done
```

---

## Option B: Run Steps Manually

Use this when you want to resume from a checkpoint or run individual
steps.

### Environment

```bash
export DATASET_ROOT=/home/jrob/cmu-grad/neuro-data/Replica
export OUTPUT_ROOT=/home/jrob/cmu-grad/neuro-experiments/min-frames-sam-auto
export SCENE=office0

ORACLE_DIR="$OUTPUT_ROOT"
ABLATION_ROOT="$OUTPUT_ROOT/ablations"
RESULTS_DIR="$OUTPUT_ROOT/results"

COMMON=(scene_id=$SCENE dataset_root=$DATASET_ROOT exp_suffix=min_frames)
```

### Step 1: Reference detection (stride=1, expensive)

```bash
python -m semgraph.stages.detect \
    "${COMMON[@]}" \
    output_root=$ORACLE_DIR \
    segmentation_backend=sam3_auto \
    stride=1
```

This runs SAM 3 on every frame.  `save_raw_detections` defaults to
`false` so no `raw_detections/` directory is created (saves disk).  Set
`save_raw_detections=true` if you want to inspect pre-filter masks.

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

Preview without creating files:

```bash
python -m semgraph.scripts.create_subset \
    --source "$ORACLE_FD" \
    --dest "$ABLATION_ROOT/fps_pose_30/$SCENE/stages/frame_data" \
    --method fps_pose --n_frames 30 --dry_run
```

### Step 6: Build map + oracle_finalize on each subset

Embed is skipped for subsets — we only need geometric comparison.

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

## Key Config Parameters

| Parameter | Default | Where | Effect |
|-----------|---------|-------|--------|
| `segmentation_backend` | `sam_auto` | `base_mapping.yaml` | Detection backend (use `sam_auto` for this experiment) |
| `stride` | `10` | `batch_vlm_mapping_api.yaml` | Frame stride for detect.py (set to 1 for reference) |
| `save_raw_detections` | `false` | `base_mapping.yaml` | Skip writing raw_detections/ to save disk |
| `build_map.matching_mode` | `incremental` | `batch_vlm_mapping_api.yaml` | `incremental` or `batch` |
| `sim_threshold` | `1.2` | `base_mapping.yaml` | Aggregated similarity threshold for matching |
| `iou_merge_kappa` | `0.25` | `base_mapping.yaml` | 3D bbox IoU fallback for merging |
| `--iou_threshold` | `0.25` | `compare_oracles.py` CLI | 3D bbox IoU threshold for evaluation matching |
