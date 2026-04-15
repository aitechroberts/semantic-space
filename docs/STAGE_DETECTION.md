# Stage A1: Detection Architecture

> **Last updated:** 2026-04-10
>
> Deep-dive reference for the detect stage. For the pipeline overview, see
> [STAGED_PIPELINE.md](STAGED_PIPELINE.md). For the full package layout, see
> [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Package Layout](#2-package-layout)
3. [ABC Contracts](#3-abc-contracts)
4. [Per-Frame Pipeline](#4-per-frame-pipeline)
5. [Filtering Chain](#5-filtering-chain)
6. [Output Artifacts](#6-output-artifacts)
7. [Data Contracts](#7-data-contracts)
8. [gt_instances Path](#8-gt_instances-path)
9. [Models and VRAM](#9-models-and-vram)
10. [Configuration Reference](#10-configuration-reference)
11. [Adding a New Detector](#11-adding-a-new-detector)

---

## 1. Architecture Overview

Detection has two independent jobs:

1. **Detection** — produce bounding boxes (and optionally class labels) from
   an image. This is what YOLO-World, Florence2, RT-DETR, GroundingDINO, and
   OWLv2 do.

2. **Segmentation** — produce pixel-precise masks, optionally prompted by
   boxes from step 1. This is what SAM does.

These two jobs are represented by two ABCs (`Detector` and `Segmenter`) that
compose into a detection pipeline. The combination is configured, not coded:

```
segmentation_backend    Detector              Segmenter
────────────────────    ────────              ─────────
"sam_auto"              None                  SAMSegmenter (auto mode)
"sam3_auto"             None                  SAM3Segmenter (auto mode)
"yolo_sam"              YOLOWorldDetector     SAMSegmenter (box-prompted)
"yoloe_sam"             YOLOEDetector         SAMSegmenter (box-prompted)
"florence2_sam"         Florence2Detector     SAMSegmenter (box-prompted)
"yolo_sam3"             YOLOWorldDetector     SAM3Segmenter (box-prompted)
"yoloe_sam3"            YOLOEDetector         SAM3Segmenter (box-prompted)
"florence2_sam3"        Florence2Detector     SAM3Segmenter (box-prompted)
"gt_instances"          (bypasses detection — geometry backend provides raw_gobs)

Future:
"rtdetr_sam"            RTDETRDetector        SAMSegmenter (box-prompted)
"owlv2_sam"             OWLv2Detector         SAMSegmenter (box-prompted)
```

When `detector` is `None` (auto mode), the segmenter runs in
segment-everything mode and the pipeline applies `filter_sam_auto_masks()`
afterward. When `detector` is present, it produces boxes and the segmenter
generates one mask per box.

The `gt_instances` path bypasses detection entirely — the geometry backend
builds `raw_gobs` from ground-truth mesh instances.

---

## 2. Package Layout

```
semgraph/detection/
    __init__.py         # get_detector(), get_segmenter() factories + re-exports
    base.py             # Detector ABC, Segmenter ABC, DetectionResult, SegmentationResult
    yolo_world.py       # YOLOWorldDetector (ultralytics YOLO-World v2)
    yoloe.py            # YOLOEDetector (ultralytics YOLOE, text-prompted)
    florence2.py        # Florence2Detector (HuggingFace Florence-2)
    sam.py              # SAMSegmenter (ultralytics SAM 2.1, auto + box-prompted)
    sam3.py             # SAM3Segmenter (ultralytics SAM 3, auto + box-prompted)
```

The consumer is `semgraph/stages/detect.py`, which calls the factories and
uses the ABC interfaces. No stage script imports `ultralytics` directly.

---

## 3. ABC Contracts

### Detector

```python
class Detector(ABC):
    def load(self, weights: str, device: str = "cuda", **kwargs) -> None: ...
    def detect(self, image_rgb: np.ndarray, *, color_path: Path | None = None) -> DetectionResult: ...
```

`image_rgb` is the primary input (`(H, W, 3)` uint8). `color_path` is
keyword-only optional context — ultralytics models prefer file paths,
HuggingFace models ignore it and work from the array.

`load()` takes only what the model needs. Weight path resolution and Hydra
config parsing stay in `detect.py`'s `load_models()`. Model classes never
import or depend on the pipeline config structure.

**`**kwargs` by implementation:**

| Implementation | kwargs | Effect |
|----------------|--------|--------|
| `YOLOWorldDetector` | `classes: list[str]` | Calls `model.set_classes()` to set the detection vocabulary |
| `YOLOEDetector` | `classes: list[str]` | Calls `model.set_classes()` to set the detection vocabulary |
| `Florence2Detector` | `task: str` | Override the default `<OD>` task prompt (e.g. `<DENSE_REGION_CAPTION>`) |

### Segmenter

```python
class Segmenter(ABC):
    def load(self, weights: str, device: str = "cuda", **kwargs) -> None: ...
    def segment(self, image_rgb: np.ndarray, boxes: np.ndarray | None = None,
                *, color_path: Path | None = None) -> SegmentationResult: ...
```

- `boxes=None` -- auto mode (segment everything)
- `boxes=(N, 4) float32` -- box-prompted mode (one mask per box)

### DetectionResult

```python
@dataclass
class DetectionResult:
    xyxy: np.ndarray        # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
    class_ids: np.ndarray   # (N,) int32
    class_labels: list[str] # per-detection, e.g. ["chair 0", "table 1"]
    classes: list[str]      # vocabulary, e.g. ["chair", "table", ...]
```

No `masks` field — detectors produce boxes only.

### SegmentationResult

```python
@dataclass
class SegmentationResult:
    masks: np.ndarray       # (N, H, W) bool
    xyxy: np.ndarray        # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
```

Always returns all three fields. In auto mode, SAM provides them natively.
In box-prompted mode, `xyxy` and `confidence` are pass-throughs from the
detector (or filled from the input boxes).

---

## 4. Per-Frame Pipeline

The full processing chain inside `detect.py`'s `process_frame()`:

```
FrameContext
    │
    ├─ skip_segmentation=True? ──► use raw_gobs from frame_ctx.extra (gt_instances)
    │
    └─ _run_detection()
         │
         ├─ detector present? ──► detector.detect() ──► segmenter.segment(boxes=...)
         │                        (DetectionResult)      (SegmentationResult)
         │
         └─ detector=None? ──► segmenter.segment(boxes=None)
                                (SegmentationResult)
                                    │
                                    └─ filter_sam_auto_masks()
    │
    ▼
  raw_gobs (RawGobs dict, 14 keys)
    │
    ▼
  resize_gobs() ──► resolution alignment
    │
    ▼
  filter_gobs() ──► four sequential filters
    │
    ▼
  _compute_surviving_indices() ──► raw-to-filtered index mapping
    │
    ▼
  mask_subtract_contained() ──► nested mask pixel subtraction
    │
    ▼
  backend.lift_to_3d() ──► geometry backend lifts masks to world-frame PCDs
    │
    ▼
  init_process_pcd() ──► voxel downsampling + DBSCAN denoising per detection
    │
    ▼
  get_bounding_box() ──► AABB or OBB computation
    │
    ▼
  make_detection_list_from_pcd_and_gobs() ──► detection_list
    │
    ▼
  (in main_standalone:)
  compute_projected_crop_bbox() ──► 1.5x crop via 3D-to-2D projection
    │
    ▼
  _save_crop() ──► JPEG write
    │
    ▼
  save_raw_det() + save_frame_data() ──► npz + JSON to disk
```

---

## 5. Filtering Chain

### `filter_sam_auto_masks()` (auto mode only, before filter_gobs)

Applied only when `detector` is `None` (SAM auto mode). Three steps:

1. **Area filter:** Drop masks with area < `sam_auto_min_mask_area_pixels`
   (default: 100) or > `sam_auto_max_mask_area_fraction` of frame (default: 0.95).
2. **Greedy NMS:** Sort by confidence descending, suppress masks whose
   axis-aligned bbox IoU exceeds `sam_auto_nms_iou_threshold` (default: 0.7).
3. Return surviving masks, xyxy, confidence.

### `filter_gobs()` (all modes, after _run_detection)

Four sequential filters applied in order:

1. **Mask area threshold** (`mask_area_threshold: 25`): Drop masks with
   fewer than 25 pixels.
2. **Background class skip** (`skip_bg: False`, `bg_classes: [wall, floor, ceiling]`):
   If enabled, drop detections whose class name is in `bg_classes`.
3. **Large bounding box suppression** (`max_bbox_area_ratio: 0.9`): Drop
   non-background detections whose bbox exceeds 90% of frame area.
4. **Confidence threshold** (`mask_conf_threshold: 0.25`): Drop detections
   below 25% confidence.

### `mask_subtract_contained()` (after filter_gobs)

Pairwise bbox containment check. If box A is largely contained within box B
(thresholds `th1=0.8`, `th2=0.7`), pixels of A's mask are subtracted from
B's mask. Prevents double-counting for nested objects (e.g., pillow on couch).

---

## 6. Output Artifacts

Per-frame data written to disk under `{dataset_root}/{scene_id}/stages/`:

| Artifact | Record Type | Key Fields |
|----------|-------------|------------|
| `raw_detections/{frame:06d}.npz+.json` | `RawDetRecord` | Pre-filter: masks (flat+offsets+shapes), xyxy, confidence, class_id, labels, classes. **Only written when `save_raw_detections` is true.** |
| `frame_data/{frame:06d}.npz+.json` | `FrameDataRecord` | Post-filter+lift: pcd_points/colors (offsets), bbox_corners, pose (4x4), intrinsics (4x4), H, W, `n_raw_detections`, surviving_indices, per-detection `_DetectionMeta` |
| `crops/{frame:06d}_{det:03d}.jpg` | JPEG | 1.5x projected crop per surviving detection |

**`_DetectionMeta` fields:** `bbox_type`, `class_name`, `class_id`, `inst_id`,
`n_points`, `crop_path`.

---

## 7. Data Contracts

### RawGobs dict (14 keys)

Defined in `semgraph/stages/paths.py::RawGobs` (TypedDict). This is what
`_run_detection()` returns:

| Key | Type | Shape | Set by |
|-----|------|-------|--------|
| `xyxy` | `np.ndarray` | `(N, 4)` float32 | detect |
| `confidence` | `np.ndarray` | `(N,)` float32 | detect |
| `class_id` | `np.ndarray` | `(N,)` int32 | detect |
| `mask` | `np.ndarray` | `(N, H, W)` bool | detect |
| `classes` | `list[str]` | vocabulary | detect |
| `detection_class_labels` | `list[str]` | N labels | detect |
| `labels` | `list[str]` | N labels | detect |
| `edges` | `list[tuple]` | relations | detect (empty) |
| `captions` | `list[str]` | N captions | detect (empty strings) |
| `image_crops` | `list` or `None` | N crops | embed (None from detect) |
| `image_feats` | `np.ndarray` or `None` | `(N, D)` | embed (None from detect) |
| `text_feats` | `np.ndarray` or `None` | `(N, D)` | embed (None from detect) |
| `vlm_vit_feats` | `np.ndarray` or `None` | `(N, D_vlm)` | embed (None from detect) |
| `vlm_proj_feats` | `np.ndarray` or `None` | `(N, D_vlm)` | embed (None from detect) |

Detect.py sets the first 9 keys. The feature fields (`image_crops` through
`vlm_proj_feats`) are set to `None` by detect.py and populated later by
embed.py.

### RawGobs to RawDetRecord mapping

`save_raw_det()` converts the RawGobs dict to a `RawDetRecord` dataclass,
which uses the offset pattern for variable-length mask storage (masks are
flattened, concatenated, and stored with offsets + shapes arrays).

### RawGobs to FrameDataRecord mapping

After filtering and 3D lifting, `main_standalone()` constructs a
`FrameDataRecord` from the detection list. This record stores per-detection
PCD points/colors (offset pattern), bbox corners, camera pose/intrinsics,
and `_DetectionMeta` for each detection.

The record also includes `n_raw_detections` — the number of masks in
`raw_gobs` before any filtering. Combined with
`len(surviving_indices)` (post-filter count), this gives the filtering
ratio for the frame, even when `save_raw_detections` is false and the
full raw detection files are not written to disk.

---

## 8. gt_instances Path

When `segmentation_backend=gt_instances` with the `gt_mesh` geometry backend:

1. **No models loaded:** `load_models()` returns `DetectionModels` with
   `detector=None`, `segmenter=None`.

2. **Iterator provides raw_gobs:** The `gt_mesh` backend's
   `_iter_gt_instances()` builds one `FrameContext` per GT mesh instance with:
   - `skip_segmentation=True`
   - `skip_matching=True`
   - `raw_gobs` pre-built in `extra` (projected bbox, binary mask from
     projected vertices, class name from `class_map`)

3. **`_run_detection` is never called:** `process_frame()` checks
   `frame_ctx.skip_segmentation` and reads `raw_gobs` from `frame_ctx.extra`.

4. **3D lifting uses mesh vertices:** `lift_to_3d()` reads `instance_pcd`
   from `frame_ctx.extra` instead of depth-unprojecting.

5. **No incremental matching:** `skip_matching=True` means `build_map.py`
   treats each instance as a new object rather than trying to match it
   against existing objects.

---

## 9. Models and VRAM

| Model | Weights | Size | Loaded When |
|-------|---------|------|-------------|
| SAM 2.1 Base | `sam2.1_b.pt` | ~162 MB | `*_sam` backends |
| SAM 3 | `sam3.pt` | ~3.5 GB | `*_sam3` backends |
| YOLO-World v2 Large | `yolov8l-worldv2.pt` | ~800 MB | `yolo_sam` / `yolo_sam3` |
| YOLOE v8-Large Seg | `yoloe-v8l-seg.pt` | ~800 MB | `yoloe_sam` / `yoloe_sam3` |
| Florence-2 Large | `microsoft/Florence-2-large` | ~1.6 GB | `florence2_sam` / `florence2_sam3` |

No CLIP encoder, no VLM — detect.py is geometry-only.

**Weight resolution:** `_resolve_weights()` in `detect.py` checks the
`CKPT_DIR` environment variable first. If the file exists under `CKPT_DIR`,
that path is used. Otherwise the bare filename is passed to ultralytics
(which auto-downloads) or used as a HuggingFace model ID (which
auto-downloads from the Hub).

**Note:** SAM 3 weights (`sam3.pt`) must be manually downloaded from
[HuggingFace](https://huggingface.co/facebook/sam3) — they are not
auto-downloaded by ultralytics.

---

## 10. Configuration Reference

All config parameters that affect the detect stage:

| Parameter | Default | Source | Effect |
|-----------|---------|--------|--------|
| `segmentation_backend` | `sam_auto` | `base_mapping.yaml` | Detection/segmentation combination (see [Architecture Overview](#1-architecture-overview) for full list) |
| `pipeline_mode` | `trajectory` | `base_mapping.yaml` | Geometry backend for 3D lifting |
| `device` | `cuda` | config | Target device for model loading |
| `skip_existing_detections` | `False` | config | Skip frames with existing `.npz` |
| `mask_area_threshold` | `25` | `base_mapping.yaml` | Min mask pixels in filter_gobs |
| `max_bbox_area_ratio` | `0.9` | `base_mapping.yaml` | Max bbox area fraction in filter_gobs |
| `mask_conf_threshold` | `0.25` | `base_mapping.yaml` | Min confidence in filter_gobs |
| `skip_bg` | `False` | `classes.yaml` | Drop background classes |
| `bg_classes` | `[wall, floor, ceiling]` | `classes.yaml` | Background class list |
| `sam_auto_min_mask_area_pixels` | `100` | `base_mapping.yaml` | SAM auto: min area filter |
| `sam_auto_max_mask_area_fraction` | `0.95` | `base_mapping.yaml` | SAM auto: max area fraction |
| `sam_auto_nms_iou_threshold` | `0.7` | `base_mapping.yaml` | SAM auto: NMS IoU threshold |
| `save_raw_detections` | `False` | `base_mapping.yaml` | Write `raw_detections/` npz+json per frame; when false the directory is not created |
| `downsample_voxel_size` | `0.01` | `batch_vlm_mapping_api.yaml` | Voxel size for PCD downsampling |
| `dbscan_remove_noise` | `True` | `base_mapping.yaml` | DBSCAN denoising per detection |
| `dbscan_eps` | `0.1` | `base_mapping.yaml` | DBSCAN neighborhood radius |
| `dbscan_min_points` | `10` | `base_mapping.yaml` | DBSCAN min cluster size |
| `spatial_sim_type` | `iou` | `base_mapping.yaml` | Affects bbox type (AABB vs OBB) |
| `min_points_threshold` | `16` | `base_mapping.yaml` | Min 3D points to keep detection |

---

## 11. Adding a New Detector

To add a new detection model (e.g., Florence2):

1. **Create** `semgraph/detection/florence2.py`:

```python
from semgraph.detection.base import Detector, DetectionResult

class Florence2Detector(Detector):
    def load(self, weights, device="cuda", **kwargs):
        # lazy-import the model library
        from transformers import AutoModelForCausalLM, AutoProcessor
        self._model = AutoModelForCausalLM.from_pretrained(weights, ...)
        self._processor = AutoProcessor.from_pretrained(weights, ...)

    def detect(self, image_rgb, *, color_path=None):
        # convert image_rgb to PIL, run model, extract boxes
        ...
        return DetectionResult(xyxy=..., confidence=..., class_ids=...,
                               class_labels=..., classes=...)
```

2. **Add to factory** in `semgraph/detection/__init__.py`:

```python
def get_detector(name):
    ...
    if name == "florence2":
        from semgraph.detection.florence2 import Florence2Detector
        return Florence2Detector()
    ...
```

3. **Use it** via config: `segmentation_backend=florence2_sam`

   In `detect.py`'s `load_models()`, add the mapping from `"florence2_sam"`
   to `get_detector("florence2")` + `get_segmenter("sam")`.

No changes needed to `_run_detection()`, `process_frame()`, or any
downstream stage.

### Adding a New Segmenter

Same pattern — implement `Segmenter`, add to `get_segmenter()`. The
segmenter must support both `boxes=None` (auto) and `boxes=(N,4)`
(box-prompted) modes via the `segment()` method.
