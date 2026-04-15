# Mapping Workflow

> **Last updated:** 2026-04-10
>
> This document describes how 2D detections become a 3D semantic scene graph,
> covering both the staged pipeline (recommended) and the legacy all-in-one mode.

---

## Overview

The mapping system transforms a sequence of posed RGBD frames into a 3D
Hierarchical Plane-enhanced Scene Graph (HPSG). The process has two distinct
phases:

1. **Phase A (Oracle Scene Construction):** Geometry-only. Produces an immutable
   3D map with point clouds, bounding boxes, structural planes, and candidate
   edges. No language models involved.

2. **Phase B (Semantic Evaluation):** Language-only. Attaches CLIP features,
   VLM captions, colors, materials, and labeled edges to the oracle map's objects.
   Runs N times with different encoder/VLM combinations.

---

## Phase A: Geometry Pipeline

### A1. Detection & 3D Lifting (`semgraph/stages/detect.py`)

Detection and segmentation are pluggable via the `Detector` and `Segmenter`
ABCs in `semgraph/detection/`. The `segmentation_backend` config selects the
combination. See [STAGE_DETECTION.md](STAGE_DETECTION.md) for the full
architecture.

For each sampled frame:

1. **Detection + Segmentation:** The configured detector/segmenter pair runs
   on the frame. Default (`sam_auto`): SAMSegmenter in auto mode finds all
   maskable regions. Alternative (`detect_sam`): detector (via `detector_type`)
   produces boxes, SAMSegmenter generates one mask per box.

2. **Filtering:** Removes masks below `mask_area_threshold` (25px), above
   `max_bbox_area_ratio` (90% of frame), and below `mask_conf_threshold` (0.25).
   Subtracts contained masks to prevent double-counting.

3. **3D lifting:** The geometry backend lifts each 2D mask to a 3D point cloud:
   - `trajectory`: depth unprojection using camera intrinsics + pose
   - `gt_mesh`: mesh vertex lookup
   - `sparse`: DUSt3R point maps

4. **1.5x projected crops:** For each detection, the 3D point cloud is projected
   back to the image plane. A bounding box is drawn around the 2D projection and
   scaled by 1.5x, giving a viewpoint-aware crop. Saved as JPEG.

5. **Camera metadata:** Pose (4x4 c2w), intrinsics (4x4), H, W are saved in
   every `FrameDataRecord`. No downstream stage needs the geometry backend.

**Output per frame:** `RawDetRecord` (pre-filter masks) + `FrameDataRecord`
(per-detection PCD, bbox corners, camera metadata) + crop JPEGs.

### A2. Oracle Feature Extraction (`semgraph/stages/embed.py`)

Loads a single CLIP-family encoder. For each frame's saved crops:

1. Batch-encodes crop images → `clip_ft (N, D)`
2. Encodes class name text → `text_ft (N, D)`
3. Writes features back into existing `FrameDataRecord` files.

These features are used during map building for visual similarity matching.

### A3. Map Building (`semgraph/stages/build_map.py`)

Incremental matching and merging loop, processing frames in sorted order
from frame 0:

1. **Reconstruct live objects:** Converts `FrameDataRecord` arrays back to
   Open3D point clouds and bounding boxes.

2. **Spatial similarity:** AABB IoU between new detections and existing objects.

3. **Visual similarity:** CLIP cosine similarity between detection `clip_ft`
   and object `clip_ft` (running weighted average).

4. **Aggregation:** `(1 + phys_bias) * spatial + (1 - phys_bias) * visual`.
   Default `phys_bias=0.0` gives equal weight.

5. **Matching:** If aggregated score > `sim_threshold` (1.2), detection merges
   into the matched object. Otherwise, it becomes a new object. Optional IoU
   merge fallback for sparse mode (`iou_merge_kappa=0.25`).

6. **Merging:** Point clouds are combined and voxel-downsampled. Bounding box
   is recomputed. CLIP features are updated via weighted average. `per_view_records`
   are stored on each object (frame_idx, clip_ft, n_points, crop_path, crop_bbox).

7. **Periodic maintenance** (every `denoise_interval` / `filter_interval` /
   `merge_interval` frames):
   - DBSCAN denoising: removes outlier points per object.
   - Object filtering: removes objects below `obj_min_points` / `obj_min_detections`.
   - Overlap merging: merges objects with >70% spatial overlap AND >70% visual
     AND >70% text similarity.

**Output:** `map/oracle_map.pkl.gz` — intermediate format with live Open3D
objects. This is the only pickle in the pipeline.

### A4. Oracle Finalization (`semgraph/stages/oracle_finalize.py`)

Converts the live map into the immutable oracle scene:

1. **MST edge construction:** Computes 3D bounding box IoU between all object
   pairs. Builds a maximum-weight minimum spanning tree (scipy). Produces
   unlabeled candidate edges — no LLM involved.

2. **HPSG plane detection:** RANSAC plane fitting on the concatenated scene
   point cloud (up to 5 major planes). Classifies by normal alignment:
   floor (Z-up), ceiling (Z-down), wall (horizontal normal). Each object is
   anchored to its nearest plane via `parent_plane_id`.

3. **Serialization:** Converts all live Open3D objects to NumPy arrays using
   the offset pattern. Saves as `OracleSceneRecord` (npz + JSON).

**Output:** `oracle/oracle_scene.npz + oracle_scene.json` — the immutable
geometric truth for all Phase B runs.

---

## Phase B: Semantic Pipeline

Phase B reads from the immutable oracle scene. It never modifies geometry.

### B1. Re-Embedding (`semgraph/stages/embed.py`, re-embed mode)

Loads a specified evaluation encoder (may differ from Phase A's encoder).
For each object's `per_view_records`:

1. Loads crop images for the object's best views.
2. Batch-encodes features.
3. Computes: weighted average (by n_points), per-view features, and
   entropy-selected best (lowest softmax entropy over a label set).

**Output:** `VariantRecord` — one per encoder.

### B2. VLM Captioning (`semgraph/stages/caption.py`)

For each object in the oracle scene:

1. Sorts views by `n_points` descending, takes top K.
2. Sends 1.5x crop images to VLM with three prompts: caption, color, material.
3. Runs LLM consolidation → `canonical_tag`, `candidate_tags`, `summary`.

**Output:** `CaptionsRecord` — one per VLM.

### B3. Semantic Assembly (`semgraph/stages/semantic_assemble.py`)

Combines oracle geometry, embed features, and VLM captions:

1. **Edge labeling:** For each unlabeled MST edge, sends object tags and
   positions to the VLM → spatial relationship (on, supports, in, contains,
   next to, none).
2. **Scene type inference:** Sends all canonical tags to VLM → room type.
3. **Plane captioning:** Templates each plane as "This is a {label} in the
   {scene_type}."
4. **HPSG assembly:** Produces the final scene graph JSON.

**Output:** `assembled/{enc}_{vlm}/scene_graph.json`

### B4. Evaluation (`semgraph/stages/eval.py`)

Reads the HPSG JSON and runs evaluation:

- **Classification:** Maps `object_tag` to 17-category groups, computes
  mIoU, F-mIoU, mAcc against ground truth.
- **QA:** Framework for ScanQA/Space3D-Bench (EM@1, BLEU, ROUGE-L, METEOR, CIDEr).
- **Retrieval:** Max-over-views similarity from embed variant (recall@K).

**Output:** `eval/{enc}_{vlm}/classification.json`

---

## Serialization

All stage boundaries use npz + JSON. The serialization stack:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Records | `semgraph/io/records.py` | Typed dataclasses with `to_arrays()` / `to_metadata()` / `from_arrays_and_metadata()` |
| Serializer | `semgraph/io/serializers/npz.py` | `np.savez_compressed` + JSON sidecar |
| Loaders | `semgraph/io/loaders.py` | `save_*/load_*` per record type |

Variable-length arrays (point clouds, features per view) use the offset
pattern: concatenate into one flat array, store cumulative offsets. On load,
slice by offsets to reconstruct per-element arrays.

The only pickle is `map/oracle_map.pkl.gz` between `build_map` and
`oracle_finalize`, because it contains live Open3D objects that cannot be
losslessly serialized to NumPy. `oracle_finalize` converts this to npz+JSON.

---

## Signal Flow: CLIP vs VLM

The system uses two complementary semantic sources:

- **CLIP embeddings** power the geometry pipeline. They provide dense, comparable
  vectors for spatial matching during map building (Phase A) and for retrieval
  evaluation (Phase B). CLIP is the matching backbone — it determines which
  detections merge into which objects.

- **VLM captions** provide human-readable metadata. The VLM describes each object
  in natural language, assigns canonical tags, infers colors/materials, and labels
  spatial relationships between objects. VLM outputs appear only in Phase B —
  they augment the scene graph for UI, filtering, or downstream LLM consumption.

When querying the graph:

1. **Vector search:** Encode a text query with CLIP and compare against stored
   embeddings. No LLM needed.
2. **Metadata filtering:** Filter by VLM-assigned tags, captions, colors, materials,
   or by the graph's spatial relationships.
3. **Hybrid:** Combine CLIP scores with VLM context for richer questions via
   a downstream LLM.

---

## Legacy All-in-One Mode

The monolith at `semgraph/slam/vlm_run/batch_vlm_mapping_api.py` runs the
full pipeline in a single frame-sequential loop. For each sampled frame it
executes segmentation, captioning, feature extraction, and 3D merging before
advancing to the next frame.

Key differences from the staged pipeline:

| Aspect | Staged | All-in-one |
|--------|--------|------------|
| Execution | 8 independent modules | Single Python process |
| VLM timing | Phase B only | Every frame in the loop |
| Serialization | npz + JSON | pkl.gz |
| Oracle map | Immutable, reusable | Not produced |
| Encoder sweep | Re-run Phase B only | Full re-run |
| VRAM | One model at a time | All models co-resident |

The staged pipeline is recommended for all new work. The monolith is retained
for backward compatibility.

---

## Key Configuration

| Parameter | Default | Effect |
|-----------|---------|--------|
| `stride` | `10` | Frames sampled (10 → ~90 frames on 900-frame Replica) |
| `segmentation_backend` | `sam_auto` | `sam_auto` (class-agnostic), `detect_sam` / `detect_sam3` (detector-first) |
| `spatial_sim_type` | `iou` | AABB IoU (fast, pure PyTorch) |
| `sim_threshold` | `1.2` | Aggregated score to match a detection to an object |
| `mask_area_threshold` | `25` | Min mask pixels |
| `max_bbox_area_ratio` | `0.9` | Max box area as fraction of frame |
| `denoise_interval` | `5` | DBSCAN every N frames |
| `merge_interval` | `5` | Overlap merge every N frames |
| `obj_min_detections` | `1` | Min detections to keep an object |

See `semgraph/hydra_configs/base_mapping.yaml` for the full parameter set.
