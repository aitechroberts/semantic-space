# Architecture

> **Last updated:** 2026-04-10

## Package Layout

```
semgraph/
    io/                         # data layer — records, serializers, save/load glue
        __init__.py             # flat re-exports for clean imports
        records.py              # Record dataclasses (RawDetRecord, FrameDataRecord, etc.)
        loaders.py              # High-level save_*/load_* functions
        serializers/
            __init__.py
            base.py             # BaseSerializer ABC
            npz.py              # NpzSerializer (np.savez_compressed + JSON sidecar)
    detection/                  # detection backends (Detector + Segmenter ABCs)
        __init__.py             # get_detector(), get_segmenter() factories
        base.py                 # Detector ABC, Segmenter ABC, DetectionResult, SegmentationResult
        yolo_world.py           # YOLOWorldDetector (YOLO-World v2 via ultralytics)
        yoloe.py                # YOLOEDetector (YOLOE via ultralytics)
        florence2.py            # Florence2Detector (Florence-2 via HuggingFace)
        sam.py                  # SAMSegmenter (SAM 2.1, auto + box-prompted)
        sam3.py                 # SAM3Segmenter (SAM 3, auto + box-prompted)
    stages/                     # execution layer — stage scripts + path resolution
        paths.py                # stage_paths(), RawGobs/SerializedDetection TypedDicts
        detect.py               # A1: segmentation + 3D lifting + 1.5x crops
        embed.py                # A2/B1: encoder feature extraction
        build_map.py            # A3: incremental map construction
        oracle_finalize.py      # A4: MST edges + HPSG planes
        caption.py              # B2: per-object VLM captioning
        semantic_assemble.py    # B3: HPSG JSON assembly
        eval.py                 # B4: classification / QA / retrieval
        postprocess.py          # legacy post-processing
    slam/
        geometry/               # geometry backends (trajectory, gt_mesh, sparse)
            base.py             # GeometryBackend ABC, FrameContext
            trajectory.py       # RGBD depth unprojection
            gt_mesh.py          # ground-truth mesh vertex lookup
            sparse.py           # DUSt3R point maps
            projection.py       # 3D-to-2D projection, 1.5x crop computation
            __init__.py         # get_geometry_backend() factory
        mapping.py              # spatial/visual similarity, matching, merging
        utils.py                # PCD processing, filtering, edge handling
        slam_classes.py         # MapObjectList, MapEdgeMapping
    utils/                      # shared utilities
    hydra_configs/              # Hydra YAML configuration
```

### Import Convention

Stage scripts import data infrastructure from `semgraph.io`:

```python
from semgraph.io import save_frame_data, load_oracle_scene, FrameDataRecord
```

Notebooks and external tools do the same — they never need to import from `semgraph.stages`.

---

## Two-Phase Pipeline Architecture

The system is organized around a **two-phase architecture** that cleanly separates geometric scene construction from semantic evaluation.

**Phase A (Oracle Scene Construction)** runs once per scene using the highest-quality models. It produces an immutable geometric map: 3D point clouds, bounding boxes, structural planes, and MST edges. No language models are involved.

**Phase B (Semantic Evaluation)** runs N times per scene, once per encoder/VLM combination being evaluated. It attaches features and captions to the oracle map's objects and produces a scored HPSG (Hierarchical Plane-enhanced Scene Graph) JSON.

```
Phase A (run once)                    Phase B (run N times)
─────────────────                     ────────────────────
detect.py ──► embed.py ──►           embed.py (re-embed) ──► caption.py ──►
build_map.py ──► oracle_finalize.py   semantic_assemble.py ──► eval.py
```

This separation means the geometry never changes between evaluation runs. Different encoders and VLMs are compared against the exact same 3D structure.

---

## Serialization: npz + JSON

All stage artifacts use a two-file format: `.npz` (compressed numpy arrays) + `.json` (metadata sidecar). There is no pickle anywhere in the staged pipeline.

The serialization stack has three layers:

1. **Records** (`semgraph/io/records.py`): Typed dataclasses (`RawDetRecord`, `FrameDataRecord`, `CaptionsRecord`, `OracleSceneRecord`, `VariantRecord`). Each has `to_arrays()`, `to_metadata()`, and `from_arrays_and_metadata()`. Variable-length arrays (e.g., point clouds that differ per detection) use an offset pattern.

2. **Serializer** (`semgraph/io/serializers/npz.py`): `NpzSerializer` writes `np.savez_compressed` + JSON sidecar. Implements the `BaseSerializer` ABC.

3. **Loaders** (`semgraph/io/loaders.py`): Thin glue — one `save_*/load_*` pair per record type.

Records never import serializers. Serializers never import records.

---

## Geometry Backends

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
- **sam_auto / detect_sam** (frame-first): Iterates over camera frames like trajectory. 2D masks are lifted to 3D by projecting mesh vertices into the frame and keeping those inside each mask.

### sparse

Runs DUSt3R on a set of RGB images (no depth, no poses required) to produce per-view 3D point maps and confidence masks. `complete` scene graph creates O(n^2) image pairs — impractical for more than ~20 images. Use `swin` or `logwin` for larger sets.

## Detection Backends

Detection and segmentation are two independent, composable jobs implemented as ABCs in `semgraph/detection/`:

| `segmentation_backend` | Detector | Segmenter | Description |
|---|---|---|---|
| `sam_auto` (default) | None | SAMSegmenter (auto) | Class-agnostic segment-everything (SAM 2.1) |
| `sam3_auto` | None | SAM3Segmenter (auto) | Class-agnostic segment-everything (SAM 3) |
| `detect_sam` | Any (via `detector_type`) | SAMSegmenter | Detector + SAM 2.1 box-prompted masks |
| `detect_sam3` | Any (via `detector_type`) | SAM3Segmenter | Detector + SAM 3 box-prompted masks |
| `gt_instances` | (bypassed) | (bypassed) | Ground-truth mesh instances |

The detector is selected independently via `detector_type` (`yoloe`, `yolo_world`, `florence2`, `gdino`) + `detector_name` (weights/model ID), mirroring `encoder_type` + `encoder_name`. Legacy strings (`yolo_sam`, `yoloe_sam`, etc.) are auto-shimmed with a deprecation warning.

`detect.py` composes a `Detector` (optional) and a `Segmenter` via factory functions. No stage script imports model libraries (ultralytics, etc.) directly. See [STAGE_DETECTION.md](STAGE_DETECTION.md) for the full detection architecture.

---

## The FrameContext.extra Dict

Each backend passes mode-specific data to `lift_to_3d()` via the `extra` dict on `FrameContext`:

| Backend | Keys in extra |
|---|---|
| `trajectory` | `depth_array` (H,W), `intrinsics_4x4` (4,4) |
| `gt_mesh` (frame-first) | `all_vertices` (V,3), `all_colors` (V,3) or None |
| `gt_mesh` (gt_instances) | `raw_gobs` (RawGobs), `instance_pcd` (o3d.PointCloud), `best_views` (list[dict]) |
| `sparse` | `pointmap` (H,W,3), `confidence` (H,W) |

**Why a dict:** `detect.py` treats `FrameContext` generically — it passes it to `backend.lift_to_3d()` without inspecting `extra`. The backend that created the context is the one that reads from it, so the type safety gap is contained.

---

## Artifact Paths

All artifacts live under `{dataset_root}/{scene_id}/stages/`:

| Stage | Artifacts |
|---|---|
| detect.py | `raw_detections/{frame:06d}.npz+.json`, `frame_data/{frame:06d}.npz+.json`, `crops/{frame:06d}_{det:03d}.jpg` |
| embed.py (Phase A) | Updates `frame_data/{frame:06d}.npz+.json` (adds clip_ft, text_ft) |
| build_map.py | `map/oracle_map.pkl.gz` (intermediate, contains live o3d objects) |
| oracle_finalize.py | `oracle/oracle_scene.npz+.json` (immutable) |
| embed.py (Phase B) | `variants/embed_{enc_slug}.npz+.json` |
| caption.py | `captions/{vlm_slug}/captions.npz+.json` |
| semantic_assemble.py | `assembled/{enc_slug}_{vlm_slug}/scene_graph.json` |
| eval.py | `eval/{enc_slug}_{vlm_slug}/classification.json` |

The only pickle in the pipeline is the intermediate `oracle_map.pkl.gz` between `build_map.py` and `oracle_finalize.py`, because it contains live Open3D objects. `oracle_finalize.py` converts this to the clean npz+JSON `OracleSceneRecord` format. Everything downstream of `oracle_finalize` is npz+JSON or pure JSON.

---

## Stage-Boundary Error Contracts

- **detect.py:** Loads detection/segmentation models via `Detector`/`Segmenter` ABC factories. Writes one `raw_det` + one `frame_data` per frame. If segmentation produces zero masks, frame is skipped (no files written). Camera pose, intrinsics, H, W are saved in `FrameDataRecord` so no downstream stage needs the geometry backend.
- **embed.py (Phase A):** Updates existing `frame_data` files to add `clip_ft` and `text_ft` arrays. Missing crop images are logged and skipped.
- **build_map.py:** Processes frames in sorted order from frame 0 (incremental loop). Missing `frame_data` files are skipped.
- **oracle_finalize.py:** Requires `map/oracle_map.pkl.gz`. Fails fast if missing. Outputs immutable `oracle_scene.npz+.json`.
- **embed.py (Phase B):** Requires `oracle/oracle_scene.npz+.json`. Outputs `VariantRecord`.
- **caption.py:** Requires `oracle/oracle_scene.npz+.json` and a running VLM server. Outputs `CaptionsRecord`.
- **semantic_assemble.py:** Requires oracle scene, embed variant, and captions. Outputs HPSG JSON.
- **eval.py:** Requires `assembled/*/scene_graph.json`. Outputs classification/QA/retrieval JSON.

---

## Config Hierarchy

All mode-specific keys live at the top level of `base_mapping.yaml`. Hydra composes configs in this order (later overrides earlier):

```
base.yaml
  -> base_mapping.yaml           # core thresholds, spatial_sim_type, intervals
    -> replica.yaml              # dataset_root, dataset_config, scene_id
      -> sam.yaml                # sam_variant
        -> classes.yaml          # classes_file, bg_classes, skip_bg
          -> logging_level.yaml
            -> prompts_standard.yaml   # VLM prompt templates
              -> batch_vlm_mapping_api.yaml  # top-level overrides + embed/caption/assemble/eval config groups
```

---

## Performance Notes

- **FrameContext memory:** Iterators are lazy — never materialize all frames into a list.
- **gt_mesh memory:** `load()` materializes the entire mesh as per-instance `o3d.PointCloud` objects. For large meshes, expect 1-2 GB.
- **DUSt3R:** `complete` scene graph creates O(n^2) pairs. Use `swin` or `logwin` for >20 images.
- **Staged pipeline disk I/O:** npz+JSON is compact (~1-2 MB/frame). Negligible vs model inference latency.
- **VRAM in detect.py:** SAM (~400 MB) only — no encoder loaded. Embed.py loads one encoder at a time.
