# NeuroNav ConceptGraphs — Staged Pipeline Reference

> **Last updated:** 2026-04-02
>
> This document describes every stage of the current pipeline in detail: what
> each stage does, what data it produces, what data it consumes, what hard
> requirements cannot be changed by config alone, and where the known
> bottlenecks live.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Flow Diagram](#2-data-flow-diagram)
3. [Stage 0 — Dataset Loading](#3-stage-0--dataset-loading)
4. [Stage 1 — Detection & Segmentation](#4-stage-1--detection--segmentation)
5. [Stage 2 — Filtering & Mask Processing](#5-stage-2--filtering--mask-processing)
6. [Stage 3 — VLM Captioning & Edge Inference](#6-stage-3--vlm-captioning--edge-inference)
7. [Stage 4 — CLIP Feature Extraction](#7-stage-4--clip-feature-extraction)
8. [Stage 5 — Depth Unprojection & Point Cloud Construction](#8-stage-5--depth-unprojection--point-cloud-construction)
9. [Stage 6 — Object Matching & Merging (3D Map Building)](#9-stage-6--object-matching--merging-3d-map-building)
10. [Stage 7 — Periodic Maintenance (Denoise / Filter / Merge-Overlap)](#10-stage-7--periodic-maintenance-denoise--filter--merge-overlap)
11. [Stage 8 — Caption Consolidation](#11-stage-8--caption-consolidation)
12. [Stage 9 — Final Serialization & Output](#12-stage-9--final-serialization--output)
13. [Object Dictionary Schema](#13-object-dictionary-schema)
14. [Configuration Hierarchy](#14-configuration-hierarchy)
15. [Known Bottlenecks](#15-known-bottlenecks)
16. [Hard Requirements](#16-hard-requirements)
17. [Information Loss Points](#17-information-loss-points)
18. [Appendix A — PyTorch3D Removal](#appendix-a--pytorch3d-removal)

---

## 1. Architecture Overview

The pipeline is a **single-pass, frame-sequential loop** implemented in
`conceptgraph/slam/vlm_run/batch_vlm_mapping_api.py`. For each sampled frame
(controlled by `stride`), it runs segmentation, captioning, feature
extraction, and 3D merging before advancing to the next frame. All stages
execute within the same Python process and share GPU VRAM.

The pipeline supports two **segmentation backends**, controlled by the
`segmentation_backend` config parameter:

- **`sam_auto`** (default, recommended): Class-agnostic SAM automatic mask
  generation. Every maskable region is found without a class vocabulary.
  Labels come from the VLM downstream — the segmenter handles geometry,
  the VLM handles semantics. Best for open-vocabulary evaluation.
- **`yolo_sam`** (legacy): YOLO-World closed-vocabulary detection + SAM
  box-prompted masks. Restricted to the 200-class ScanNet vocabulary.
  Use for controlled benchmark reproduction.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Per-Frame Loop (main)                         │
│                                                                     │
│  sam_auto path:                                                     │
│  Dataset ──► SAM 2.1 auto ──► filter (area/NMS) ──►               │
│  mask_subtract_contained ──► depth unproject ──► PCD + BBox ──►    │
│  VLM caption/edges ──► CLIP features ──► match & merge ──►        │
│  periodic denoise/filter/merge-overlap                              │
│                                                                     │
│  yolo_sam path (legacy):                                            │
│  Dataset ──► YOLO-World ──► SAM 2.1 box ──► filter_gobs ──►       │
│  (same downstream as above)                                         │
│                                                                     │
│  After loop: caption consolidation ──► save PCD, JSON, edges       │
└─────────────────────────────────────────────────────────────────────┘
```

**Entry point:** `batch_vlm_mapping_api.py::main()` via Hydra.

**Config file:** `conceptgraph/hydra_configs/batch_vlm_mapping_api.yaml`,
which composes from `base.yaml`, `base_mapping.yaml`, `replica.yaml` (or
dataset override), `sam.yaml`, `classes.yaml`, `logging_level.yaml`, and
`prompts_standard.yaml`.

---

## 2. Data Flow Diagram

```
Frame N (RGB + Depth + Pose + Intrinsics)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ seg_backend == "sam_auto" (default):                         │
│   SAM 2.1 predict() [automatic mode, no prompts]            │
│   ──► masks, xyxy, conf (class-agnostic)                    │
│   ──► filter_sam_auto_masks (area, NMS)                     │
│   ──► class_id=0 for all, classes=["object"]                │
│                                                              │
│ seg_backend == "yolo_sam" (legacy):                          │
│   YOLO-World predict() ──► xyxy, conf, class_id             │
│   SAM 2.1 predict(bboxes=xyxy) ──► masks                    │
│   ──► classes from ScanNet200 vocabulary                     │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────┐
│ VLM API captions     │──► captions[], edges[]
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ TinyCLIP encode      │──► image_feats (N, D), text_feats (N, D)
└──────────────────────┘
    │
    ▼
   raw_gobs dict ──────────────────────────────────────────────────────┐
    │                                                                   │
    ▼                                                               [save_detections]
┌──────────────────────┐                                                │
│ resize_gobs()        │                                                ▼
│ filter_gobs()        │──► filtered_gobs                        det_exp_pkl_path/
│ mask_subtract_cont() │                                          └── ...
└──────────────────────┘
    │
    ▼
┌──────────────────────────────────┐
│ detections_to_obj_pcd_and_bbox() │──► [{pcd, bbox}, ...]
│ init_process_pcd()               │
│ get_bounding_box()               │
└──────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│ make_detection_list_from_pcd_and_gobs│──► DetectionList (list of dicts)
└──────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ compute_spatial_similarities()  │──► spatial_sim (M×N tensor)
│ compute_visual_similarities()   │──► visual_sim  (M×N tensor)
│ aggregate_similarities()        │──► agg_sim     (M×N tensor)
│ match_detections_to_objects()   │──► match_indices [int|None, ...]
│ merge_obj_matches()             │──► updated MapObjectList
└─────────────────────────────────┘
    │
    ▼
  MapObjectList (persistent across frames)
    │
    ▼ (periodically)
┌──────────────────────┐
│ denoise_objects()    │  DBSCAN per-object noise removal
│ filter_objects()     │  remove low-detection/low-point objects
│ merge_objects()      │  merge spatially overlapping objects
└──────────────────────┘
    │
    ▼ (after loop)
┌──────────────────────┐
│ consolidate_captions │  VLM API call per object
│ save_pointcloud()    │  .pkl.gz with serialized MapObjectList
│ save_obj_json()      │  scene graph JSON
│ save_edge_json()     │  edge/relation JSON
└──────────────────────┘
```

---

## 3. Stage 0 — Dataset Loading

**Code:** `conceptgraph/dataset/datasets_common.py::get_dataset()`

**What it does:**
Loads a NICE-SLAM / GradSLAM-format dataset from disk. Returns a PyTorch
dataset object that, when indexed, yields `(color_tensor, depth_tensor,
intrinsics, ...)`. Also exposes `dataset.color_paths[i]` and
`dataset.poses[i]`.

**Inputs:**
- `dataset_root`: path to the dataset (e.g., `/home/jrob/cmu-grad/neuro-data/replica`)
- `dataset_config`: YAML file describing the dataset format (intrinsics, depth scale, etc.)
- `scene_id`: subdirectory within the dataset root (e.g., `room0`)
- `start`, `end`, `stride`: frame sampling parameters

**Outputs (per index):**
| Field | Type | Shape |
|-------|------|-------|
| `color_tensor` | `torch.Tensor` | `(H, W, 3)` uint8 values |
| `depth_tensor` | `torch.Tensor` | `(H, W, 1)` float meters |
| `intrinsics` | `torch.Tensor` | `(4, 4)` camera matrix |
| `pose` (via `dataset.poses[i]`) | `torch.Tensor` | `(4, 4)` camera-to-world |

**Hard requirements:**
- The dataset must provide RGB, depth, camera intrinsics, and camera poses per frame.
- Depth must be metric (meters). If your depth source gives millimeters or a disparity map, the dataconfig YAML must specify the scale factor.
- Poses must be camera-to-world transformations in a consistent coordinate system across all frames.
- The dataset format must conform to one of the supported types: `Replica`, `ScanNet`, `TUM`, `Record3D`, or `AzureKinect` (see `datasets_common.py` for the registry).

**Stride interaction:**
The `stride` parameter directly controls which frames are sampled. With `stride=10` on a 900-frame Replica scene, ~90 frames are processed. With `stride=50`, ~18 frames. This is the single most impactful parameter for runtime vs. reconstruction quality.

---

## 4. Stage 1 — Detection & Segmentation

**Code:** `batch_vlm_mapping_api.py`, detection block within the frame loop.

This stage is controlled by the `segmentation_backend` config parameter.

### 4a. `sam_auto` backend (default — class-agnostic)

**Models loaded:**
| Model | Weights | VRAM | Purpose |
|-------|---------|------|---------|
| SAM 2.1 Base | `sam2.1_b.pt` | ~400 MB | Automatic mask generation (segment everything) |

**What it does:**

1. **SAM 2.1** runs `predict(color_path)` in automatic mode (no prompts).
   - Returns: all maskable regions in the frame — objects, surfaces, structures.
   - No class vocabulary needed. Every visible region gets a mask.

2. **`filter_sam_auto_masks()`** reduces raw SAM output to meaningful regions:
   - Drops masks with area < `sam_auto_min_mask_area_pixels` (default: 100px).
   - Drops masks covering > `sam_auto_max_mask_area_fraction` of the frame (default: 95%).
   - Applies NMS at `sam_auto_nms_iou_threshold` (default: 0.7) to remove heavy overlaps.

3. Results are packed into `sv.Detections` with `class_id=0` for all masks
   and `classes=["object"]`. Labels are generic (`"object 0"`, `"object 1"`, etc.).

**Why this is the default:** The class-agnostic approach maximizes information
reaching the VLM. No object is invisible because it wasn't on a predefined
list. The VLM handles semantics (what is it?) and the segmenter handles
geometry (where is it?). This is essential for evaluating VLM capabilities on
arbitrary real-world scenes (Record3D captures, novel environments, etc.).

### 4b. `yolo_sam` backend (legacy — closed vocabulary)

**Models loaded:**
| Model | Weights | VRAM | Purpose |
|-------|---------|------|---------|
| YOLO-World v2 Large | `yolov8l-worldv2.pt` | ~800 MB | Closed-vocabulary object detection |
| SAM 2.1 Base | `sam2.1_b.pt` | ~400 MB | Box-prompted instance segmentation |

**What it does:**

1. **YOLO-World** runs `predict(color_path, conf=0.1)` with a 200-class ScanNet vocabulary.
   - Only detects objects matching the vocabulary. Everything else is invisible.

2. **SAM 2.1** runs `predict(color_path, bboxes=xyxy_tensor)` using YOLO boxes as prompts.
   - Returns: binary masks for each detected box.

Use this mode for controlled benchmark reproduction where the class list is known.

### Output (both backends)

`sv.Detections` with fields:
| Field | Type | Shape |
|-------|------|-------|
| `xyxy` | `np.ndarray` | `(N, 4)` float32 |
| `confidence` | `np.ndarray` | `(N,)` float32 |
| `class_id` | `np.ndarray` | `(N,)` int (all 0 in sam_auto) |
| `mask` | `np.ndarray` | `(N, H, W)` bool |

**Hard requirements:**
- Both models run on CUDA. No CPU fallback is implemented.
- `yolo_sam` requires a class vocabulary file (`classes_file` in config).
- `sam_auto` has no vocabulary dependency.

**Bottleneck:** SAM automatic mode is ~100–200ms per frame (more masks to generate). YOLO+SAM box-prompted is ~50–100ms per frame. Neither is the dominant bottleneck (that's the VLM).

---

## 5. Stage 2 — Filtering & Mask Processing

**Code:** `conceptgraph/slam/utils.py::filter_gobs()` (lines 838–906), `resize_gobs()` (lines 909–936), `mask_subtract_contained()` in `conceptgraph/utils/ious.py` (line 453+)

**What it does:**

Three sequential operations on the raw detections:

### 5a. `resize_gobs()`
If the mask resolution doesn't match the RGB image resolution (can happen with certain SAM outputs or dataset rescaling), resizes all masks and recomputes xyxy coordinates to match the image dimensions. Uses nearest-neighbor interpolation.

### 5b. `filter_gobs()`
Iterates over every detection and applies four filters **in order**:

1. **Mask area threshold** (`mask_area_threshold: 25`): Any detection whose binary mask has fewer than 25 pixels is dropped. This removes noise fragments.

2. **Background class skip** (`skip_bg: False`, `bg_classes: ["wall", "floor", "ceiling"]`): If `skip_bg` is True, detections whose class name matches a background class are dropped. **Currently disabled** (`False` in `classes.yaml`), but the filter exists and would remove all wall/floor/ceiling detections if enabled.

3. **Large bounding box suppression** (`max_bbox_area_ratio: 0.9`): For non-background classes, if the bounding box area exceeds 90% of the image area, the detection is dropped. Background classes are exempt from this filter (the `if class_name not in BG_CLASSES` guard). In `sam_auto` mode where all masks are `class_name="object"`, this filter applies uniformly at the 90% threshold — permissive enough to keep walls, large furniture, and structural elements while still dropping full-frame noise masks.

4. **Confidence threshold** (`mask_conf_threshold: 0.25`): Detections below 25% confidence are dropped.

After filtering, the function also re-indexes `captions` and `detection_class_labels` to match the surviving detections.

### 5c. `mask_subtract_contained()`
Computes pairwise bounding box containment between all surviving detections. If box A is largely contained within box B (configurable thresholds `th1=0.8`, `th2=0.7`), the pixels of A's mask are subtracted from B's mask. This prevents double-counting — e.g., a pillow's mask pixels are removed from the couch's mask that contains it.

**Complexity:** O(N²) pairwise comparison on bounding boxes. At 15 detections this is negligible. At 100+ detections (e.g., with SAM automatic mode), this becomes measurable.

**Hard requirements:**
- `filter_gobs()` assumes `gobs['class_id']` indexes into `gobs['classes']` — the class vocabulary array. Any detection backend must provide a valid `class_id` → `classes` mapping.
- The `mask_subtract_contained` function operates on axis-aligned bounding boxes only. It does not use mask IoU for the containment check, only bbox IoU. This means two objects with overlapping bounding boxes but non-overlapping masks will still trigger subtraction.

**Bottleneck:** Negligible for typical detection counts (10–20 per frame). Would become significant at 100+ detections per frame.

---

## 6. Stage 3 — VLM Captioning & Edge Inference

**Code:** `batch_vlm_mapping_api.py::make_vlm_edges_and_captions()` (lines 91–139)

**Dependencies:** Requires a running vLLM server at `vlm_api_url` (default: `http://localhost:8000/v1`) serving a VLM model (default: `Qwen/Qwen3-VL-2B-Instruct`).

**What it does:**

1. **Annotation:** Draws bounding boxes and numeric IDs on a copy of the frame using `supervision` annotators.

2. **Caption API call:** Sends the annotated image + a list of `detection_class_labels` (e.g., `["chair 0", "table 1", "monitor 2"]`) to the VLM via `vlm_client.caption_objects_with_labels()`. The VLM returns structured JSON with per-object captions.

3. **Relation API call** (if `make_edges: True`): Sends the same annotated image to the VLM via `vlm_client.infer_relations_with_labels()`. Returns a list of `(id1, relationship, id2)` tuples.

**Output:**
| Field | Type | Description |
|-------|------|-------------|
| `labels` | `list[str]` | Per-detection label strings |
| `edges` | `list[tuple]` | Spatial relation triples |
| `captions` | `list[str]` | Per-detection natural language captions |
| `edge_image` | `np.ndarray` | The annotated frame |

**If `make_edges` is False or `vlm_client` is None:**
- `labels` = `detection_class_labels` from YOLO
- `edges` = `[]`
- `captions` = `[""] * N`

**Hard requirements:**
- The VLM server must be running and healthy before the loop starts. `wait_for_server()` polls for up to 120 seconds.
- The VLM sees a **2D annotated image** with bounding boxes and numeric IDs. The quality of captions and relations depends entirely on: (a) the annotation quality, (b) the VLM's visual understanding, and (c) the prompt templates in `prompts_standard.yaml`.
- Captions are **per-frame, per-detection**. The same physical object seen across 10 frames will accumulate 10 separate captions. These are consolidated after the loop in Stage 8.

**Bottleneck:** **This is the single largest time bottleneck.** Each VLM API call takes 1–5 seconds depending on the model and hardware. With `make_edges=True`, there are **two API calls per frame** (caption + relations). At stride 10 with 90 frames, that's 180 API calls = 3–15 minutes of VLM time alone. At stride 50 with 18 frames, ~36 API calls = 30s–3min.

---

## 7. Stage 4 — CLIP Feature Extraction

**Code:** `batch_vlm_mapping_api.py::compute_tinyclip_features_batched()` (lines 166–234)

**Model:** TinyCLIP (`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`), ~24M parameters, loaded via HuggingFace `transformers`.

**What it does:**

1. For each detection, crops the image with 20px padding around the bounding box.
2. Encodes all crops as a batch through TinyCLIP's image encoder → `image_feats (N, D)`.
3. Encodes the class label text for each detection through TinyCLIP's text encoder → `text_feats (N, D)`.
4. Both feature sets are L2-normalized.

**Optional VLM Encoder Features:**
If `extract_vlm_encoder_feats: True`, the VLM's own vision encoder is also used to encode the same crops, producing `vlm_vit_feats` and `vlm_proj_feats`. This is for research comparison and does not affect the mapping pipeline.

**Output per detection:**
| Field | Type | Shape | Used for |
|-------|------|-------|----------|
| `image_feats` | `np.ndarray` | `(N, D)` | Visual similarity in matching |
| `text_feats` | `np.ndarray` | `(N, D)` | Currently unused in matching |
| `image_crops` | `list[PIL.Image]` | N crops | Stored in raw_gobs, saved if `save_detections` |
| `vlm_vit_feats` | `np.ndarray` or None | `(N, D_vlm)` | Research comparison |
| `vlm_proj_feats` | `np.ndarray` or None | `(N, D_vlm)` | Research comparison |

**Hard requirements:**
- TinyCLIP is the **sole feature used for visual similarity during object matching** (Stage 6). The matching backbone cannot be changed by config alone — it requires code changes to swap the model.
- The crop padding (20px) is hardcoded. No 1.5x scaling or configurable crop factor.
- TinyCLIP's embedding dimension (512) determines the `clip_ft` dimension stored in every object and used for all downstream similarity computations.

**Bottleneck:** TinyCLIP is extremely fast (~5ms for a batch of 15 crops). This is not a bottleneck.

---

## 8. Stage 5 — Depth Unprojection & Point Cloud Construction

**Code:** `conceptgraph/slam/utils.py::detections_to_obj_pcd_and_bbox()` (lines 1241–1319), `init_process_pcd()` (line 237), `get_bounding_box()` (line 264)

**What it does:**

For each surviving detection mask:

1. **Batch unprojection:** Converts all masked pixels from 2D image coordinates to 3D camera-frame points using the depth map and camera intrinsics:
   ```
   x = (u - cx) * depth / fx
   y = (v - cy) * depth / fy
   z = depth
   ```
   This is done on GPU as a batched tensor operation (`batch_mask_depth_to_points_colors()`).

2. **Dynamic downsampling:** If the point count exceeds `obj_pcd_max_points` (default: 5000), randomly subsamples to approximately that count.

3. **Pose transformation:** Transforms points from camera frame to world frame using `pcd.transform(trans_pose)` where `trans_pose` is the camera-to-world pose.

4. **Voxel downsampling:** `pcd.voxel_down_sample(voxel_size=downsample_voxel_size)` reduces point density. Default voxel size: 0.01m (1cm) in `batch_vlm_mapping_api.yaml`, 0.025m in `base_mapping.yaml`.

5. **DBSCAN denoising:** `init_pcd_denoise_dbscan()` clusters points and keeps only the largest cluster. Parameters: `dbscan_eps=0.1`, `dbscan_min_points=10`.

6. **Bounding box computation:** `get_bounding_box()` produces either an Oriented Bounding Box (OBB) or Axis-Aligned Bounding Box (AABB) depending on `spatial_sim_type`:
   - If `spatial_sim_type` contains `"accurate"` or `"overlap"`: uses `pcd.get_oriented_bounding_box(robust=True)` (OBB)
   - Otherwise (including the default `"iou"`): uses `pcd.get_axis_aligned_bounding_box()` (AABB)

**Output per detection:**
| Field | Type | Description |
|-------|------|-------------|
| `pcd` | `o3d.geometry.PointCloud` | World-frame colored point cloud |
| `bbox` | `o3d.geometry.OrientedBoundingBox` or `AxisAlignedBoundingBox` | 3D bounding box |

Detections with fewer than `min_points_threshold` (16) valid depth pixels, or whose bounding box volume is < 1e-6, are discarded (set to `None`).

**Hard requirements:**
- Requires metric depth maps. Invalid depth (0 or negative) pixels are excluded.
- Requires camera-to-world poses. If poses are noisy or have drift, point clouds from different frames will not align correctly, and the matching/merging stage will fail to associate the same physical object across views.
- Open3D is a hard dependency for all point cloud and bounding box operations.
- The `min_points_threshold=16` is a hard floor — any detection that projects to fewer than 16 3D points is permanently dropped. For small or distant objects, or objects with missing depth, this silently removes them.

**Bottleneck:** The GPU-based batch unprojection is fast. The DBSCAN clustering (O(N log N)) runs per-object and is moderate. The overall stage is ~20–50ms per frame for typical detection counts.

---

## 9. Stage 6 — Object Matching & Merging (3D Map Building)

**Code:** `conceptgraph/slam/mapping.py` (all functions)

**What it does:**

This is the core incremental mapping stage. For each frame after the first, new detections are compared against all existing objects in the map:

### 6a. `compute_spatial_similarities()`
Computes a `(M, N)` matrix where M = new detections, N = existing objects.

The method depends on `spatial_sim_type` (default: `"iou"`):

| Type | Implementation | Notes |
|------|---------------|-------|
| `"iou"` | AABB IoU via `compute_iou_batch()` | **Default.** Fast, pure PyTorch, axis-aligned only. |
| `"giou"` | Generalized AABB IoU via `compute_giou_batch()` | Handles non-overlapping boxes better |
| `"iou_accurate"` | Delegates to `compute_iou_batch()` (AABB) | Legacy name preserved for config compatibility. Formerly used PyTorch3D OBB IoU; now equivalent to `"iou"`. |
| `"giou_accurate"` | Delegates to `compute_giou_batch()` (AABB) | Legacy name preserved. Formerly used PyTorch3D OBB GIoU; now equivalent to `"giou"`. |
| `"overlap"` | Point-based overlap via FAISS nearest-neighbor | Most expensive but most accurate. Uses AABB IoU as a prefilter. |

The `"overlap"` method (`compute_overlap_matrix_general()`) uses FAISS to compute what fraction of each new detection's points are within `downsample_voxel_size` distance of any existing object's points. It first computes a bounding box IoU matrix (AABB) to skip pairs with zero spatial overlap.

> **PyTorch3D removal (2026-03-23):** The `"iou_accurate"` and `"giou_accurate"`
> code paths previously used `pytorch3d.ops.box3d_overlap()` for exact oriented
> bounding box intersection. These now delegate to the AABB equivalents
> (`compute_iou_batch` / `compute_giou_batch`), which are pure PyTorch with
> zero external dependencies. The default `spatial_sim_type` was changed from
> `"overlap"` to `"iou"` simultaneously, making the per-frame matching path a
> simple tensor operation rather than a FAISS point-search. See
> [Appendix A](#appendix-a--pytorch3d-removal) for the full tradeoff analysis.

### 6b. `compute_visual_similarities()`
Computes cosine similarity between each new detection's `clip_ft` and each existing object's `clip_ft`. Returns a `(M, N)` tensor.

The `clip_ft` of an existing object is a **running weighted average** — it's the detection-count-weighted mean of all `clip_ft` vectors that have been merged into that object so far.

### 6c. `aggregate_similarities()`
Combines spatial and visual similarities:
```
agg_sim = (1 + phys_bias) * spatial_sim + (1 - phys_bias) * visual_sim
```
With `phys_bias=0.0` (default), this is simply `spatial + visual`.

### 6d. `match_detections_to_objects()`
For each detection, finds the object with the highest aggregated similarity. If the max similarity exceeds `sim_threshold` (default: 1.2), the detection is matched to that object. Otherwise, it becomes a new object.

### 6e. `merge_obj_matches()`
For each matched detection:
- Calls `merge_obj2_into_obj1()` which:
  - **Extends lists:** `image_idx`, `mask_idx`, `color_path`, `class_id`, `mask`, `xyxy`, `conf`, `contain_number`, `captions` all get the new detection's values appended. **This is the source of unbounded memory growth** — these lists grow with every merge and are never truncated during the loop.
  - **Merges point clouds:** `obj1['pcd'] += obj2['pcd']` followed by voxel downsampling and optional DBSCAN.
  - **Updates bounding box:** Recomputed from the merged point cloud.
  - **Averages CLIP features:** Weighted average by detection count, then re-normalized.
  - **Averages VLM encoder features:** Same weighted average approach.

For unmatched detections, a new object is appended to the `MapObjectList`.

### Edge processing
After matching, `process_edges()` maps VLM-inferred relation tuples to object indices in the map, creating or updating `MapEdge` objects. Edges that are detected only once and are older than 5 frames are pruned.

**Hard requirements:**
- The `"overlap"` spatial similarity method requires `faiss` (CPU) for nearest-neighbor search.
- `merge_obj2_into_obj1()` has an **explicit unhandled-key check**: if `obj2` contains any key not listed in `extend_attributes`, `add_attributes`, `skip_attributes`, or `custom_handled`, it raises a `ValueError`. Adding new fields to the detection dict requires updating this function.
- The `sim_threshold=1.2` means a detection needs a combined spatial+visual score above 1.2 to match an existing object. Since each component ranges from 0 to 1 (approximately), this requires meaningful overlap on both dimensions.

**Bottleneck:** With the default `spatial_sim_type: "iou"`, spatial similarity is a pure batched tensor operation and is negligible (~1ms per frame regardless of map size). If switched to `"overlap"`, it becomes the most expensive per-frame operation at O(M × N × P) where M = new detections, N = existing objects, P = points per object — potentially 100–500ms per frame at 100+ objects.

---

## 10. Stage 7 — Periodic Maintenance (Denoise / Filter / Merge-Overlap)

**Code:** `conceptgraph/slam/utils.py`: `denoise_objects()` (line 658), `filter_objects()` (line 700), `merge_objects()` (line 728)

These run at configurable intervals during the loop AND always on the final frame.

### 7a. `denoise_objects()` — Interval: every `denoise_interval` frames (default: 5)
For every object in the map, re-runs DBSCAN clustering on its accumulated point cloud. Keeps only the largest cluster. This removes outlier points that may have been merged from noisy depth readings.

**Cost:** O(N × K log K) where N = objects, K = points per object. Heavy.

### 7b. `filter_objects()` — Interval: every `filter_interval` frames (default: 5)
Removes objects with fewer than `obj_min_points` points or fewer than `obj_min_detections` detections. Currently `obj_min_points=0` and `obj_min_detections=1` in `batch_vlm_mapping_api.yaml`, so this is effectively a no-op until the final frame.

In `base_mapping.yaml`, `obj_min_detections=3`, which would filter out objects seen in fewer than 3 frames. This only takes effect if `batch_vlm_mapping_api.yaml` doesn't override it.

When objects are removed, edge indices are remapped.

### 7c. `merge_objects()` — Interval: every `merge_interval` frames (default: 5)
Computes the full pairwise overlap matrix between all existing objects. For any pair where overlap > `merge_overlap_thresh` (0.7) AND visual similarity > `merge_visual_sim_thresh` (0.7) AND text similarity > `merge_text_sim_thresh` (0.7), the objects are merged.

This is the **post-hoc merge** that catches objects that should have been matched during incremental mapping but weren't (e.g., because they first appeared from very different viewpoints).

**Cost:** O(N² × P) for the overlap matrix computation. The most expensive periodic operation.

**Hard requirement:** The merge thresholds are AND-conditions — all three (overlap, visual sim, text sim) must exceed their thresholds simultaneously. This is conservative and can leave duplicates unmerged if any one dimension disagrees.

---

## 11. Stage 8 — Caption Consolidation

**Code:** `batch_vlm_mapping_api.py`, lines 708–716

**What it does:**
After the frame loop completes, for each object that has accumulated captions:
1. Takes the first 20 captions (truncates if more).
2. Sends them to the VLM API via `consolidate_captions()` with the consolidation prompt.
3. Stores the result as `obj["consolidated_caption"]`.

If an object has no captions, its `consolidated_caption` is set to its YOLO `class_name`.

**Bottleneck:** One VLM API call per object. With 100 objects, this is another 100–500 seconds of VLM time. This runs **after** the main loop, so it's serial and blocking.

---

## 12. Stage 9 — Final Serialization & Output

**Code:** `conceptgraph/utils/general_utils.py`: `save_pointcloud()`, `save_obj_json()`, `save_edge_json()`

### Output artifacts:

#### `pcd_{exp_suffix}.pkl.gz`
A gzipped pickle containing:
```python
{
    'objects': [...],     # list of serialized object dicts
    'cfg': {...},         # Hydra config snapshot
    'class_names': [...], # class vocabulary
    'class_colors': {...} # per-class RGB colors
}
```
Each object dict contains:
- `clip_ft` (numpy array)
- `vlm_vit_ft`, `vlm_proj_ft` (numpy arrays or None)
- `pcd_np` (Nx3 float array of points)
- `bbox_np` (8x3 float array of OBB corners)
- `pcd_color_np` (Nx3 float array of colors)
- All metadata: `class_name`, `class_id`, `image_idx`, `captions`, `consolidated_caption`, `conf`, `xyxy`, `num_detections`, etc.

#### `semantic_{exp_suffix}.pkl.gz` (if `save_semantic_snapshot: True`)
Same structure but without geometry (`pcd_np`, `bbox_np`, `pcd_color_np` omitted). Lightweight snapshot for embedding-only analysis.

#### `obj_json_{exp_suffix}.json`
```json
{
  "object_1": {
    "id": 0,
    "object_tag": "chair",
    "object_caption": "A wooden office chair with armrests",
    "bbox_extent": [0.45, 0.52, 0.88],
    "bbox_center": [1.23, -0.15, 0.44],
    "bbox_volume": 0.21
  },
  ...
}
```

#### `edge_json_{exp_suffix}.json`
```json
{
  "edge_0": {
    "edge_id": 0,
    "edge_description": "chair next to table",
    "num_detections": 3,
    "object_1_id": 0,
    "object_1_tag": "chair",
    "object_2_id": 1,
    "object_2_tag": "table"
  },
  ...
}
```

**Hard requirements:**
- The `.pkl.gz` format uses Python pickle, which is not portable across different Python/NumPy versions.
- `MapObjectList.to_serializable()` calls `copy.deepcopy()` on every object, which temporarily doubles memory usage during saving.
- Open3D objects (`PointCloud`, `OrientedBoundingBox`) are converted to numpy arrays for serialization. Loading them back requires Open3D to reconstruct the objects.

---

## 13. Object Dictionary Schema

Every object in the `MapObjectList` is a Python dict. Here is the complete schema after merging:

| Key | Type | Cardinality | Source | Grows with merges? |
|-----|------|-------------|--------|--------------------|
| `id` | `uuid.UUID` | 1 | First detection | No |
| `image_idx` | `list[int]` | N_detections | Frame index | **Yes, unbounded** |
| `mask_idx` | `list[int]` | N_detections | Mask index within frame | **Yes, unbounded** |
| `color_path` | `list[Path]` | N_detections | RGB image path | **Yes, unbounded** |
| `class_name` | `str` | 1 | Most common class | Updated per frame |
| `class_id` | `list[int]` | N_detections | Class vocabulary index | **Yes, unbounded** |
| `captions` | `list[str]` | N_detections | VLM per-frame caption | **Yes, unbounded** |
| `num_detections` | `int` | 1 | Counter | Incremented |
| `mask` | `list[np.ndarray]` | **N_detections** | Binary masks `(H,W)` | **Yes, unbounded** — each mask is `H×W` booleans |
| `xyxy` | `list[np.ndarray]` | N_detections | Bounding boxes | **Yes, unbounded** |
| `conf` | `list[float]` | N_detections | Detection confidence | **Yes, unbounded** |
| `contain_number` | `list[None]` | N_detections | Placeholder | **Yes, unbounded** |
| `n_points` | `int` | 1 | Point count | Updated after merge |
| `inst_color` | `np.ndarray` | 1 | Random `(3,)` color | No |
| `is_background` | `bool` | 1 | Class membership | No |
| `pcd` | `o3d.geometry.PointCloud` | 1 | Merged + downsampled | Grows then stabilizes |
| `bbox` | `o3d.geometry.OrientedBoundingBox` | 1 | From merged PCD | Recomputed |
| `clip_ft` | `torch.Tensor` | 1 | Weighted average `(D,)` | Updated (averaged) |
| `vlm_vit_ft` | `torch.Tensor` or None | 1 | Weighted average `(D,)` | Updated (averaged) |
| `vlm_proj_ft` | `torch.Tensor` or None | 1 | Weighted average `(D,)` | Updated (averaged) |
| `num_obj_in_class` | `int` | 1 | Class instance counter | Added |
| `curr_obj_num` | `int` | 1 | Global object counter | No |
| `new_counter` | `int` | 1 | Brand-new object counter | No |
| `consolidated_caption` | `str` | 1 | Post-loop VLM call | Set once |

**Memory concern:** The fields marked "Yes, unbounded" grow linearly with the number of times an object is re-detected. For a chair seen in 20 frames, `mask` alone stores 20 full-resolution boolean masks. At 480×640 resolution, each mask is ~300KB, so 20 masks = ~6MB per object. With 100 objects, that's ~600MB of masks alone in system RAM.

---

## 14. Configuration Hierarchy

Hydra composes configs in this order (later overrides earlier):

```
base.yaml
  └── base_mapping.yaml           ← core thresholds, spatial_sim_type, intervals
        └── replica.yaml          ← dataset_root, dataset_config, scene_id
              └── sam.yaml        ← sam_variant (legacy, not actively used)
                    └── classes.yaml    ← classes_file, bg_classes, skip_bg
                          └── logging_level.yaml
                                └── prompts_standard.yaml  ← VLM prompt templates
                                      └── batch_vlm_mapping_api.yaml  ← top-level overrides
```

`batch_vlm_mapping_api.yaml` is the final layer and overrides values from all
parent configs. For example, `stride: 10` in this file overrides `stride: 50`
from `base_mapping.yaml`.

**Key config values and their defaults (as resolved):**

| Parameter | Value | Source | Effect |
|-----------|-------|--------|--------|
| `segmentation_backend` | `"sam_auto"` | `base_mapping.yaml` / env `SEG_BACKEND` | `"sam_auto"` (class-agnostic) or `"yolo_sam"` (legacy) |
| `stride` | 10 | `batch_vlm_mapping_api.yaml` | Frames sampled |
| `spatial_sim_type` | `"iou"` | `base_mapping.yaml` | AABB IoU (pure PyTorch) |
| `sim_threshold` | 1.2 | `base_mapping.yaml` | Match threshold |
| `mask_area_threshold` | 25 | `base_mapping.yaml` | Min mask pixels |
| `mask_conf_threshold` | 0.25 | `base_mapping.yaml` | Min detection confidence |
| `max_bbox_area_ratio` | 0.9 | `base_mapping.yaml` | Max box area fraction |
| `sam_auto_min_mask_area_pixels` | 100 | `base_mapping.yaml` | SAM auto: min mask area (pixels) |
| `sam_auto_max_mask_area_fraction` | 0.95 | `base_mapping.yaml` | SAM auto: max mask area (fraction of frame) |
| `sam_auto_nms_iou_threshold` | 0.7 | `base_mapping.yaml` | SAM auto: NMS overlap threshold |
| `skip_bg` | False | `classes.yaml` | Keep wall/floor/ceiling |
| `downsample_voxel_size` | 0.01 | `batch_vlm_mapping_api.yaml` | 1cm voxels |
| `obj_pcd_max_points` | 5000 | `batch_vlm_mapping_api.yaml` | Max points per object |
| `denoise_interval` | 5 | `batch_vlm_mapping_api.yaml` | DBSCAN every 5 frames |
| `filter_interval` | 5 | `batch_vlm_mapping_api.yaml` | Filter every 5 frames |
| `merge_interval` | 5 | `batch_vlm_mapping_api.yaml` | Merge-overlap every 5 frames |
| `obj_min_detections` | 1 | `batch_vlm_mapping_api.yaml` | Min detections to keep object |
| `merge_overlap_thresh` | 0.7 | `base_mapping.yaml` | 70% overlap to merge |
| `merge_visual_sim_thresh` | 0.7 | `base_mapping.yaml` | 70% visual sim to merge |
| `merge_text_sim_thresh` | 0.7 | `base_mapping.yaml` | 70% text sim to merge |
| `force_detection` | True | `batch_vlm_mapping_api.yaml` | Always re-run detection |
| `save_detections` | False | `batch_vlm_mapping_api.yaml` | Don't save detection artifacts |
| `make_edges` | True | env `MAKE_EDGES` | Enable VLM relations |

---

## 15. Known Bottlenecks

### 15.1 VLM API Calls (Dominant)
Two API calls per frame (caption + relation) at 1–5s each. At stride 10:
~180 calls = 3–15 min. Plus ~100 consolidation calls after the loop.

**Mitigation:** Set `make_edges: False` to cut calls in half. Increase stride to reduce frames. Use a faster VLM model.

### 15.2 Spatial Similarity Computation (Grows Over Time)
With the current default `spatial_sim_type: "iou"`, this is a fast batched tensor operation and no longer a bottleneck. If switched to `"overlap"`, every frame would compute FAISS nearest-neighbor between all new detections and all existing objects, with cost growing as O(N) with map size.

**Mitigation:** Keep the default `"iou"`. Only switch to `"overlap"` if rotated-object matching accuracy is critical and you accept the runtime cost.

### 15.3 Periodic Merge-Overlap (Quadratic)
The `merge_objects()` call computes a full N×N overlap matrix. At 200 objects, this involves 40,000 FAISS searches. This runs every `merge_interval` frames.

**Mitigation:** Increase `merge_interval` or set to `-1` to only run on the final frame. Reduce `merge_overlap_thresh` to merge more aggressively during incremental matching.

### 15.4 System RAM Accumulation
Unbounded growth of `mask`, `image_idx`, `color_path`, `xyxy`, `captions`, `conf` per object. A 100-object scene at stride 10 can consume 1–2 GB of RAM just for accumulated masks.

**Mitigation:** Currently none implemented. Would require truncating old masks or only keeping the most recent N masks per object.

### 15.5 VRAM Pressure
YOLO-World (~800MB), SAM 2.1 (~400MB), TinyCLIP (~100MB), and optionally a VLM encoder all share VRAM. Total resident: ~1.3–2.5 GB before any VLM encoder. Point cloud operations on GPU add temporary allocation.

**Mitigation:** Use `detection_only` mode to skip CLIP/VLM loading. Offload TinyCLIP to CPU. Unload detection models after the detection phase.

---

## 16. Hard Requirements

These are structural constraints that cannot be changed by config flags alone:

| Requirement | Reason | What would need to change |
|-------------|--------|---------------------------|
| **Closed-vocabulary detection** | Only with `yolo_sam` backend. Default `sam_auto` is class-agnostic | Set `segmentation_backend: sam_auto` (now the default) |
| **20px fixed crop padding** | Hardcoded in `compute_tinyclip_features_batched()` | Change to configurable `crop_scale_factor` |
| **TinyCLIP as matching backbone** | Hardcoded model name, loaded via HuggingFace `transformers` | Replace with OpenCLIP factory pattern |
| **FAISS for overlap computation** | `compute_overlap_matrix_general()` uses `faiss.IndexFlatL2` (only when `spatial_sim_type: "overlap"`) | Not on the default code path (`"iou"`); only needed if you switch to `"overlap"` |
| **Open3D for point clouds** | All PCD operations, bounding boxes, DBSCAN | No practical alternative; Open3D is deeply embedded |
| **`sv.Detections` as intermediate format** | All detection backends must produce `xyxy`, `confidence`, `class_id`, `mask` | Standard enough; most detectors can output this |
| **Pickle serialization** | `.pkl.gz` output format | Would need migration to HDF5, Parquet, or similar |
| **Single-process frame loop** | No parallelism between frames | Would require fundamental restructuring for multi-frame batching |
| **Camera-to-world poses required** | Point clouds are transformed to world frame for matching | Cannot work with pose-free datasets |
| **Metric depth required** | Depth values used directly for 3D coordinate computation | Disparity or relative depth would need conversion |

> **Note:** PyTorch3D is **no longer a dependency** as of 2026-03-23. All 3D
> bounding box IoU functions now use pure PyTorch AABB approximations. See
> [Appendix A](#appendix-a--pytorch3d-removal).

---

## 17. Information Loss Points

Every stage in the pipeline can permanently remove information. Here is where scene understanding is lost, ordered by impact:

### 17.1 Closed Vocabulary (Stage 1) — **ELIMINATED with `sam_auto`**
With the default `sam_auto` backend, this information loss point no longer exists.
Every maskable region is segmented regardless of category. Objects like
thermostats, outlets, artwork, food items, and novel objects are all captured.
The `yolo_sam` backend retains the original 200-class ScanNet vocabulary
constraint for backward compatibility.

### 17.2 Large Box Suppression (Stage 2) — **MITIGATED**
`max_bbox_area_ratio` has been relaxed from 0.5 to 0.9, allowing walls, large
furniture, and structural elements to pass through. Combined with `sam_auto`
(where all masks are treated as non-background), this filter now only removes
masks covering >90% of the frame — typically full-scene noise rather than
meaningful objects.

### 17.3 Confidence Threshold (Stage 2) — **MEDIUM IMPACT**
`mask_conf_threshold=0.25` removes uncertain detections. These are often partially occluded objects, objects at frame edges, or unusual viewpoints — exactly the cases that test a VLM's spatial reasoning.

### 17.4 Min Points Threshold (Stage 5) — **MEDIUM IMPACT**
`min_points_threshold=16` drops detections that project to fewer than 16 3D points. Small objects, objects with missing depth, and thin structures (poles, wires, picture frames) can fall below this threshold.

### 17.5 DBSCAN Denoising (Stage 5 & 7) — **LOW-MEDIUM IMPACT**
DBSCAN keeps only the largest cluster and discards all other points. For objects with disconnected geometry (e.g., a chair with thin legs), the legs might be a separate cluster and get removed.

### 17.6 Object Filtering (Stage 7) — **LOW IMPACT (currently)**
With `obj_min_detections=1`, no objects are filtered during the loop. But `base_mapping.yaml` has `obj_min_detections=3`, and switching to that (or not overriding it) would remove all objects seen in fewer than 3 frames — which at high stride means many objects.

### 17.7 Caption Truncation (Stage 8) — **LOW IMPACT**
Only the first 20 captions per object are sent for consolidation. For objects seen in 50+ frames, later captions (potentially from better viewpoints) are ignored.

---

---

## Appendix A — PyTorch3D Removal

**Date:** 2026-03-23

### What was removed

`pytorch3d` was eliminated as a dependency. It was used in exactly one file
(`conceptgraph/utils/ious.py`) in four functions, all of which called
`pytorch3d.ops.box3d_overlap()` — a CUDA kernel that computes exact
intersection volume between oriented 3D bounding boxes (OBBs) by clipping one
box against the planes of the other.

### Why it was removed

PyTorch3D is one of the most painful dependencies in the Python ML ecosystem:
- Requires building from source with a specific CUDA toolkit version matching
  your PyTorch installation.
- Build takes 10–30 minutes and frequently fails on non-standard environments.
- Pinned to a specific Git commit in `pyproject.toml` for reproducibility,
  which breaks whenever PyTorch or CUDA updates.
- The entire library is pulled in for a single function (`box3d_overlap`).

### What replaced it

All four functions now use **axis-aligned bounding box (AABB) approximations**
that were already present in the codebase as pure PyTorch operations:

| Function | Before | After |
|----------|--------|-------|
| `compute_3d_iou_accurate_batch()` | `pytorch3d.ops.box3d_overlap()` → exact OBB IoU | Delegates to `compute_iou_batch()` → AABB IoU |
| `compute_3d_giou_accurate_batch()` | `pytorch3d.ops.box3d_overlap()` + convex hull enclosing volume | Delegates to `compute_giou_batch()` → AABB GIoU |
| `compute_3d_giou_accurate()` | Single-pair `pytorch3d.ops.box3d_overlap()` | AABB min/max bounds via Open3D + NumPy |
| `compute_3d_contain_ratio_accurate_batch()` | `pytorch3d.ops.box3d_overlap()` → intersection volume / box1 volume | AABB intersection volume via pure PyTorch |

The function signatures and return types are unchanged. All existing callers
(`mapping.py`, `utils.py`) work without modification.

### Config change

`spatial_sim_type` in `base_mapping.yaml` was changed from `"overlap"` to
`"iou"`. This means per-frame matching now uses AABB IoU (a fast batched tensor
operation) instead of FAISS point-based overlap (which itself called
`compute_3d_iou_accurate_batch` as a prefilter). The `"overlap"` option is
still available and still works — it now uses AABB IoU for its prefilter step
instead of OBB IoU.

### Quality tradeoffs

**AABB vs OBB IoU accuracy:**

For an object whose oriented bounding box is rotated by angle θ around the
vertical axis, the AABB envelope overestimates the box area in the horizontal
plane by a factor of approximately `(|cos θ| + |sin θ|)²`. At θ=45° (worst
case), the AABB area is ~2x the OBB area in 2D, giving a volume overestimate
of roughly 40% in 3D (one rotation axis).

**Why this is acceptable for indoor scenes:**

1. **Most furniture is axis-aligned.** In Replica and ScanNet, chairs face
   tables, desks are against walls, beds are parallel to walls. The AABB is
   exact for all axis-aligned objects.

2. **Visual similarity dominates matching.** The aggregate matching score is
   `spatial_sim + visual_sim`. With TinyCLIP (or ViT-bigG-14), the visual
   component is the primary matching signal. A slight overestimate of spatial
   overlap for a rotated object increases the spatial score by a small amount,
   which is noise relative to the visual similarity signal.

3. **AABB overestimate is conservative, not destructive.** Overestimating
   overlap means slightly more objects pass the similarity threshold — i.e.,
   slightly more aggressive merging. For scene understanding evaluation, merging
   two fragments of the same object is preferable to leaving them separate.

4. **The `"overlap"` prefilter is unaffected in practice.** The AABB prefilter
   in `compute_overlap_matrix_general()` checks "is IoU > 1e-6?" (i.e., do
   the boxes overlap at all). AABB overestimation means a few extra FAISS
   nearest-neighbor checks for objects that are close but not truly overlapping.
   The final overlap decision still comes from point-to-point distance, which is
   unaffected by the bounding box representation.

**Expected metric impact:**

| Metric | Expected delta | Reasoning |
|--------|---------------|-----------|
| Object count | ±0–3 objects per scene | Slightly more aggressive merging for rotated objects |
| Merge accuracy | -0.5–1% | Rare false-positive merges for adjacent rotated objects |
| mIoU (semantic) | Negligible (<0.5%) | Downstream of merge quality; dominated by detection recall and VLM captioning |
| Runtime | -20–40% faster | `"iou"` eliminates FAISS computation from the per-frame matching path |

**Portability:**

The AABB approach works identically regardless of data source: Replica,
ScanNet, Record3D, custom depth cameras, etc. No BEV images, LiDAR data, or
special camera setups are required. The only input is the 8 corner points of
each bounding box, which Open3D computes from any point cloud.

### Files changed

| File | Change |
|------|--------|
| `conceptgraph/utils/ious.py` | Rewrote 4 functions to use AABB fallbacks |
| `conceptgraph/hydra_configs/base_mapping.yaml` | `spatial_sim_type: overlap` → `iou` |
| `pyproject.toml` | Removed `pytorch3d` from dependencies, build config, and sources |

---

*End of document. For implementation details of the staged pipeline v2 enhancements (encoder sweep, VLM caption sweep, edge construction, HPSG), see the implementation files in `conceptgraph/slam/vlm_run/`.*
