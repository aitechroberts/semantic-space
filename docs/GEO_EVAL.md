# Geometry Fidelity Evaluation (`geo_eval.py`)

Stage A5 — runs after `oracle_finalize.py`. Compares oracle scene objects
against ground-truth mesh instances. Purely geometric: no models, no GPU.

## Metrics

Four per-object metrics, each revealing a different failure mode:

| Metric | Formula | Diagnoses |
|--------|---------|-----------|
| **Volumetric IoU** | \|V_oracle ∩ V_gt\| / \|V_oracle ∪ V_gt\| | Overall geometric agreement |
| **Completeness** | \|V_oracle ∩ V_gt\| / \|V_gt\| | Missing geometry (low = sparse views missed parts) |
| **Accuracy** | \|V_oracle ∩ V_gt\| / \|V_oracle\| | Hallucinated geometry (low = noise/misplaced points) |
| **Chamfer Distance** | Mean nearest-neighbor distance | Surface precision (complements voxel metrics) |

Completeness and accuracy decompose in a useful way:

- **Low completeness + high accuracy**: the oracle captured only part of the
  object, but what it has is geometrically correct. Typical of sparse-view
  reconstructions with limited coverage.
- **Low accuracy + high completeness**: the oracle captured the full object
  but with geometric noise or misplaced points. Typical of depth sensor noise
  in the `trajectory` backend.

Chamfer distance is computed in both directions:
- **oracle→GT** mirrors accuracy (hallucinated geometry is far from GT surface)
- **GT→oracle** mirrors completeness (missing geometry is far from oracle surface)

## Matching Strategy

### General path (Hungarian)

For `sam_auto`, `sam3_auto`, and other non-GT segmentation backends, oracle
objects are matched to GT instances via Hungarian assignment on 3D AABB IoU:

1. Compute pairwise AABB IoU between all oracle objects and GT instances.
2. Run `scipy.optimize.linear_sum_assignment` on the negated IoU matrix.
3. Accept matches with IoU >= `match_iou_threshold` (default 0.2).
4. For borderline matches (IoU between threshold and threshold+0.1), apply a
   centroid distance sanity check — reject if centroids are >0.15m apart.

### `gt_instances` fast path

When `segmentation_backend=gt_instances`, `build_map` skips the merge loop
(`skip_matching=True`) and appends objects 1:1, preserving instance IDs. The
code detects this case and uses direct index-to-ID mapping. When
`validate_gt_instances: true`, it also runs Hungarian as a cross-check and
logs a warning if the assignments disagree (which signals an unexpected merge).

## ICP Alignment (sparse backend)

The `sparse` (DUSt3R) backend produces point clouds that may be in an
arbitrary scale/coordinate frame. Set `geo_eval.align_icp: true` to run ICP
registration before voxelization:

1. Concatenate all oracle PCDs and all GT PCDs into scene-level clouds.
2. Downsample both at 2x voxel_size for speed.
3. Run `open3d.pipelines.registration.registration_icp()`.
4. Apply the resulting rigid transform to each oracle object PCD.

The ICP fitness and RMSE are logged and included in `geometry.json`. For
`trajectory` and `gt_mesh` backends (which use ground-truth camera poses),
alignment is unnecessary and `align_icp` defaults to `false`.

## Usage

```bash
python -m semgraph.stages.geo_eval \
    scene_id=office0 \
    dataset_root=/path/to/Replica \
    mesh_path=/path/to/Replica/office0/mesh_semantic.ply \
    instance_class_map=/path/to/office0/instance_class_map.json
```

All standard Hydra overrides apply. The stage reads `oracle_scene.npz` from
the oracle directory and writes `geometry.json` to `stages/geo_eval/`.

### Configuration

In `batch_vlm_mapping_api.yaml`:

```yaml
geo_eval:
  voxel_size: 0.005              # 5mm for Replica-scale
  match_iou_threshold: 0.2       # minimum bbox IoU to accept a match
  centroid_max_dist: 0.15        # reject borderline matches (centroids > 0.15m)
  compute_chamfer: true          # directional Chamfer distance
  align_icp: false               # set true for sparse backend
  validate_gt_instances: true    # cross-check direct vs Hungarian
```

## Output Schema (`geometry.json`)

```json
{
  "scene_level": {
    "n_oracle": 45,
    "n_gt": 72,
    "n_matched": 40,
    "recall": 0.556,
    "precision": 0.889,
    "f1": 0.684,
    "vol_iou": {"mean": 0.72, "median": 0.78},
    "completeness": {"mean": 0.81, "median": 0.85},
    "accuracy": {"mean": 0.84, "median": 0.88},
    "chamfer_oracle_to_gt": {"mean": 0.012, "median": 0.008},
    "chamfer_gt_to_oracle": {"mean": 0.015, "median": 0.011}
  },
  "per_object": [
    {
      "oracle_idx": 0,
      "bbox_center": [1.2, 0.5, 0.8],
      "gt_instance_id": 14,
      "gt_class_name": "chair",
      "match_iou": 0.65,
      "vol_iou": 0.78,
      "completeness": 0.85,
      "accuracy": 0.88,
      "n_oracle_voxels": 1240,
      "n_gt_voxels": 1380,
      "chamfer_oracle_to_gt": 0.008,
      "chamfer_gt_to_oracle": 0.011
    }
  ],
  "unmatched_oracle": [12, 33],
  "unmatched_gt": [5, 19, 44],
  "matching": {"method": "hungarian"},
  "config": {
    "voxel_size": 0.005,
    "match_iou_threshold": 0.2,
    "pipeline_mode": "gt_mesh",
    "segmentation_backend": "sam_auto"
  }
}
```

The `bbox_center` field is a secondary join key for future stratified
evaluation. When `eval.py` consumes this file, it joins on `oracle_idx` and
cross-checks against `bbox_center` to guard against oracle reordering.

## Backend Benchmarking Workflow

Run Phase A three times with different geometry backends, then `geo_eval`
on each:

```bash
# 1. GT mesh backend (upper bound)
PIPELINE_MODE=gt_mesh python -m semgraph.stages.detect ...
# ... embed, build_map, oracle_finalize ...
python -m semgraph.stages.geo_eval ... pipeline_mode=gt_mesh

# 2. Trajectory backend (depth sensor)
PIPELINE_MODE=trajectory python -m semgraph.stages.detect ...
# ...
python -m semgraph.stages.geo_eval ... pipeline_mode=trajectory

# 3. Sparse backend (DUSt3R, no depth)
PIPELINE_MODE=sparse python -m semgraph.stages.detect ...
# ...
python -m semgraph.stages.geo_eval ... pipeline_mode=sparse geo_eval.align_icp=true
```

Compare the three `geometry.json` files. Expected fidelity ordering:

    gt_mesh > trajectory > sparse

If your results deviate from this ordering, that is a bug signal — either the
GT mesh loading is wrong, the depth sensor has systematic errors, or the
DUSt3R alignment failed.

The key thesis question this answers: if the `sparse` backend produces, say,
0.75 Volumetric IoU against `gt_mesh`, but downstream semantic classification
(Phase B) only drops 2 points of mIoU, you have demonstrated that the
semantic layer is robust to geometric imprecision.

## Future Work

### Stratified Phase B Evaluation (Item 3)

Load `geometry.json` in `eval.py`, partition objects by `vol_iou` threshold:

- "For objects with Vol IoU > 0.8, what is the classification mIoU?"
- "For objects with Vol IoU < 0.5, what is the classification mIoU?"

This separates geometric failure modes from semantic failure modes. The
`oracle_idx` and `bbox_center` join keys in the per-object schema are
designed for this purpose.

### Build_map Diagnostic Trace (Item 4)

Log per-object geometric quality during the incremental merge loop in
`build_map.py`. After each merge, compare the merged object's PCD against the
GT instance. This gives a trace showing how fidelity evolves with frame count
(e.g., "object 7 starts at IoU 0.3 after frame 10, reaches 0.85 by frame 50,
degrades to 0.7 by frame 80 due to a bad merge").

Not implemented because `build_map.py` currently has no GT mesh awareness.
Injecting GT context would couple Phase A construction to evaluation,
violating the modular design. Better as a separate diagnostic tool.

### Plane Validation (Item 5)

Voxelize the detected plane's inlier points and compare against GT mesh
structural surfaces. If the floor plane's voxels have low overlap with the
actual floor mesh faces, RANSAC parameters need tuning.

Both Replica (per-face `object_id` with structural instance classes) and
ScanNet++ (annotated mesh semantic labels) provide GT structural surface
segmentation, so the approach generalizes to both target datasets.
