# Architecture

## Pipeline Modes and Geometry Backends

Three geometry backends control how 2D detections are lifted to 3D:

| Backend | Input Requirements | 3D Lifting Method |
|---|---|---|
| `trajectory` (default) | RGBD depth + camera poses | Depth unprojection |
| `gt_mesh` | Ground-truth annotated mesh | Mesh vertex lookup |
| `sparse` | RGB images only | DUSt3R point maps |

### trajectory

Uses GradSLAMDataset to load RGBD frames with known camera intrinsics and poses. Each frame's depth map is unprojected to 3D using the camera parameters. This is the standard path for datasets like Replica, ScanNet, and R3DScanner.

### gt_mesh

Two sub-modes controlled by `segmentation_backend`:

- **gt_instances** (object-first): Iterates over GT mesh instances. Selects best camera views per instance. Skips 2D detection entirely — point clouds come directly from mesh vertices.
- **sam_auto / yolo_sam** (frame-first): Iterates over camera frames like trajectory. 2D masks are lifted to 3D by projecting mesh vertices into the frame and keeping those inside each mask.

### sparse

Runs DUSt3R on a set of RGB images (no depth, no poses required) to produce per-view 3D point maps and confidence masks. `complete` scene graph creates O(n^2) image pairs — impractical for more than ~20 images. Use `swin` or `logwin` for larger sets.

## Segmentation Backends

| Backend | Type | Description |
|---|---|---|
| `sam_auto` (default) | Class-agnostic | SAM automatic mask generation |
| `yolo_sam` | Closed-vocabulary | YOLO-World detection + SAM box-prompted |
| `gt_instances` | Ground-truth | Instance labels from mesh (`gt_mesh` mode only) |

## Error Contracts per Backend Method

### load()

- **Fatal:** missing dataset path, missing mesh file, DUSt3R model download failure.
- `SparseBackend.load()` can fail at pair construction or global alignment divergence. Fatal — no graceful degradation.
- `gt_mesh` `load_instance_mesh`: if `label_key` not found in PLY data, raises `ValueError` listing available keys.
- `gt_mesh` mesh loading: falls back from `plyfile` to `trimesh` if plyfile is unavailable.

### get_iterator()

- **Recoverable:** individual frame image load failure — skip frame, log warning.
- **Fatal:** zero frames/instances pass filters.

### lift_to_3d()

- **Recoverable:** zero valid depth points for a mask — returns empty PointCloud, detection gets filtered later by `min_points_threshold`.
- **Fatal:** none defined (all failures are per-detection).

### num_iterations()

Returns `int | None`. Currently all backends return `int`. `None` is reserved for future backends where the count cannot be cheaply pre-computed.

## The FrameContext.extra Dict

Each backend passes mode-specific data to `lift_to_3d()` via the `extra` dict on `FrameContext`:

| Backend | Keys in extra |
|---|---|
| `trajectory` | `depth_array` (H,W), `intrinsics_4x4` (4,4) |
| `gt_mesh` (frame-first) | `all_vertices` (V,3), `all_colors` (V,3) or None |
| `gt_mesh` (gt_instances) | `raw_gobs` (RawGobs), `instance_pcd` (o3d.PointCloud), `best_views` (list[dict]) |
| `sparse` | `pointmap` (H,W,3), `confidence` (H,W) |

**Why a dict:** `detect.py` treats `FrameContext` generically — it passes it to `backend.lift_to_3d()` without inspecting `extra`. The backend that created the context is the one that reads from it, so the type safety gap is contained.

**Cost:** No IDE autocompletion, `KeyError` at runtime on misspelled keys.

**Mitigation:** Each backend documents the exact keys it puts into `extra` in its `get_iterator` docstring, and the exact keys it reads in its `lift_to_3d` docstring.

**Future:** If stages other than `detect.py` start consuming `FrameContext.extra`, migrate to typed subclasses (`TrajectoryFrameContext`, `SparseFrameContext`, etc.).

## The RawGobs Schema

Defined in `conceptgraph/stages/paths.py::RawGobs` (TypedDict, 14 keys). See that file for the canonical field list with types and shapes.

**Producer/consumer mapping:**

| Key | Produced by | Consumed by |
|---|---|---|
| `xyxy`, `confidence`, `class_id`, `mask` | detect.py (segmentation) | detect.py (filter), caption.py (annotation) |
| `image_feats`, `text_feats` | detect.py (TinyCLIP) | build_map.py (visual similarity) |
| `vlm_vit_feats`, `vlm_proj_feats` | detect.py (VLM encoder, optional) | build_map.py (visual similarity) |
| `captions`, `edges`, `labels` | caption.py | build_map.py (caption merging, edge processing) |
| `image_crops`, `classes`, `detection_class_labels` | detect.py | caption.py (annotation) |

**Factory:** `make_empty_gobs()` in `paths.py` is the single source of truth for constructing RawGobs outside the detection pipeline (used by `gt_instances` and tests).

## Stage-Boundary Error Contracts

- **detect.py:** Writes one `raw_det` + one `frame_data` file per frame. If segmentation produces zero masks, frame is skipped (no files written).
- **caption.py:** Writes one `captions` file per frame. On VLM timeout, writes empty captions. Missing `raw_det` files are skipped with a warning.
- **build_map.py:** Processes frames in sorted order. Missing `frame_data` files are skipped. Missing `captions` files are treated as empty captions. **Must always run from frame 0** — the matching/merging loop is incremental; frame N depends on accumulated state from frames 0 through N-1. Cannot re-run a frame range.
- **postprocess.py:** Requires `map.pkl.gz`. Fails fast if missing.

## Performance Scaling Notes

- **FrameContext memory:** Holds a full `image_rgb` array (~1 MB at 640x480) plus `extra` data. Iterators are lazy — never materialize all frames into a list. For trajectory with 2000 frames + depth arrays, materializing would cost 4+ GB.
- **gt_mesh memory:** `load()` materializes the entire mesh as per-instance `o3d.PointCloud` objects. For Matterport with 10M+ vertices, expect 1-2 GB.
- **gt_mesh best-view selection:** O(instances x frames x vertices). For 100 instances, 900 frames, 100K vertices/instance, this is expensive. Future: precomputed visibility matrices.
- **DUSt3R:** `complete` scene graph creates O(n^2) pairs. Use `swin` or `logwin` for >20 images.
- **Staged pipeline disk I/O:** ~2 MB/frame x 90 frames = ~180 MB per scene. Negligible vs VLM latency.
- **VRAM in detect.py:** SAM (~300 MB) + TinyCLIP (~50 MB) + optional VLM encoder (several GB for 7B models). If VLM encoder is enabled, requires a large GPU.

## Config Hierarchy

All mode-specific keys live at the top level of `base_mapping.yaml`. This means `dust3r_niter` appears even for `trajectory` mode. Trade-off: simpler flat config, all overridable via env vars or CLI. Future improvement: migrate to Hydra config groups so mode-specific keys are absent when irrelevant.

## Logging and Observability

- **All-in-one mode:** `MappingTracker` (singleton) + `OptionalWandB` wrapper, initialized in the monolith. Tracks per-frame counts, logs to wandb if enabled.
- **Staged mode:** Each stage prints to stdout. No `MappingTracker`, no wandb. The shell orchestrator captures output. Rationale: 4 separate wandb runs per scene is not useful.
- **Future:** JSON-lines log per stage + post-hoc aggregation script if needed.

## Validation Strategy

See `z_evaluations/compare_pipeline_outputs.py` (created during validation phase). Compares all-in-one vs staged output on the same scene: object count (exact match), per-object CLIP cosine similarity (>0.999), bbox center distance (<1cm), edge count (exact match).
