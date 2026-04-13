# Staged Pipeline Reference

> **Last updated:** 2026-04-10
>
> This document describes the two-phase staged pipeline: what each stage
> does, what data it produces and consumes, the serialization format, and
> the orchestration model.

---

## Table of Contents

1. [Design Principles](#1-design-principles)
2. [Phase A — Oracle Scene Construction](#2-phase-a--oracle-scene-construction)
   - [A1: detect.py](#a1-detectpy)
   - [A2: embed.py (oracle mode)](#a2-embedpy-oracle-mode)
   - [A3: build_map.py](#a3-build_mappy)
   - [A4: oracle_finalize.py](#a4-oracle_finalizepy)
3. [Phase B — Semantic Evaluation](#3-phase-b--semantic-evaluation)
   - [B1: embed.py (re-embed mode)](#b1-embedpy-re-embed-mode)
   - [B2: caption.py](#b2-captionpy)
   - [B3: semantic_assemble.py](#b3-semantic_assemblepy)
   - [B4: eval.py](#b4-evalpy)
4. [Serialization Architecture](#4-serialization-architecture)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Shell Orchestration](#6-shell-orchestration)
7. [Stage Output Artifacts](#7-stage-output-artifacts)
8. [HPSG Output Schema](#8-hpsg-output-schema)
9. [Legacy All-in-One Mode](#9-legacy-all-in-one-mode)

---

## 1. Design Principles

**Two-phase separation.** Phase A builds geometry. Phase B attaches semantics. The oracle map produced by Phase A is immutable — Phase B never modifies it.

**One encoder at a time.** Detect.py loads no encoders. Embed.py loads exactly one CLIP-family encoder. Caption.py loads no encoders (only a VLM client). This keeps VRAM usage predictable.

**Every stage independently runnable.** Each stage is a standalone Python module with a Hydra entry point:

```bash
python -m semgraph.stages.detect <hydra overrides>
python -m semgraph.stages.embed embed.mode=re_embed embed.encoder_name=...
```

**npz + JSON everywhere.** All stage artifacts use compressed numpy arrays (`.npz`) with a JSON metadata sidecar (`.json`). No pickle in the pipeline except the intermediate map between `build_map` and `oracle_finalize`.

**Camera metadata saved early.** `detect.py` saves camera `pose`, `intrinsics`, `H`, `W` in every `FrameDataRecord`, so no downstream stage ever needs the geometry backend.

---

## 2. Phase A — Oracle Scene Construction

Phase A runs once per scene. It uses the highest-quality models available. No language models are involved — Phase A is purely geometric.

### A1: detect.py

**Code:** `semgraph/stages/detect.py`

**What it does:**
1. Loads detection and segmentation models via the `Detector`/`Segmenter` ABC factories in `semgraph/detection/`. The combination is configured by `segmentation_backend` (e.g., `sam_auto` = SAMSegmenter in auto mode, `yolo_sam` = YOLOWorldDetector + SAMSegmenter box-prompted).
2. For each frame: runs detection + segmentation, filters masks, lifts 2D masks to 3D via the geometry backend.
3. For each detection: projects the 3D point cloud back to 2D, computes a 1.5x scaled bounding box around the projection, saves the crop as a JPEG.
4. Saves camera metadata (pose, intrinsics, H, W) so downstream stages never need the geometry backend.

See [STAGE_DETECTION.md](STAGE_DETECTION.md) for the full detection architecture, ABC contracts, filtering chain, and guide for adding new detectors.

**Models loaded:**

| Model | Weights | VRAM | Purpose |
|-------|---------|------|---------|
| SAM 2.1 Base | `sam2.1_b.pt` | ~400 MB | Automatic mask generation or box-prompted |
| YOLO-World v2 Large | `yolov8l-worldv2.pt` | ~800 MB | Object detection (`yolo_sam` mode only) |

No CLIP encoder, no VLM — geometry only.

**1.5x projected crops:** Instead of the legacy 20px pixel padding, detect.py uses `compute_projected_crop_bbox()` from `semgraph/slam/geometry/projection.py`. This projects the detection's 3D point cloud onto the image plane, computes the tight 2D bounding box of the projection, scales it by 1.5x, and clamps to image bounds. The result is a viewpoint-aware crop that captures context proportional to the object's projected size.

**Outputs:**

| Artifact | Format | Contents |
|----------|--------|----------|
| `raw_detections/{frame:06d}.npz+.json` | `RawDetRecord` | Pre-filter masks, xyxy, confidence, class_id, labels |
| `frame_data/{frame:06d}.npz+.json` | `FrameDataRecord` | Per-detection PCD points/colors (offset pattern), bbox corners, camera pose/intrinsics, detection metadata |
| `crops/{frame:06d}_{det:03d}.jpg` | JPEG | 1.5x projected crop per detection |

**Key config:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `segmentation_backend` | `sam_auto` | `sam_auto` (class-agnostic) or `yolo_sam` (closed vocabulary) |
| `skip_existing_detections` | `False` | Skip frames that already have `.npz` output |
| `mask_area_threshold` | `25` | Min mask pixels to survive filtering |
| `max_bbox_area_ratio` | `0.9` | Max box area as fraction of frame |

---

### A2: embed.py (oracle mode)

**Code:** `semgraph/stages/embed.py` (default mode, `embed.mode=phase_a`)

**What it does:**
1. Loads a single CLIP-family encoder (default: `openai/clip-vit-large-patch14`).
2. For each frame's `FrameDataRecord`: loads saved 1.5x crop images, batch-encodes through the image encoder.
3. Optionally runs SAM fusion: re-segments the crop, blacks out background, encodes again, averages with full-crop feature.
4. Encodes class name text through the text encoder.
5. Writes `clip_ft` and `text_ft` arrays back into the `FrameDataRecord`.

**Models loaded:** One CLIP-family model via HuggingFace `transformers`. Configurable via `embed.encoder_name`.

**Outputs:** Updates existing `frame_data/{frame:06d}.npz+.json` files to include `clip_ft (N, D)` and `text_ft (N, D)` arrays.

**Key config:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `embed.encoder_name` | `openai/clip-vit-large-patch14` | Which encoder to use |
| `embed.use_sam_fusion` | `False` | SAM-based background removal for crops |

---

### A3: build_map.py

**Code:** `semgraph/stages/build_map.py`

**What it does:**
1. Iterates over all `frame_data` files in sorted order from frame 0.
2. For each frame: reconstructs live Open3D objects from the `FrameDataRecord` arrays.
3. Runs the incremental matching/merging loop: spatial similarity (IoU), visual similarity (CLIP cosine), aggregated similarity, threshold-based matching, and point cloud merging.
4. Stores `per_view_records` on each merged object (frame_idx, clip_ft, n_points, crop_path, crop_bbox) for downstream use.
5. Runs periodic maintenance: DBSCAN denoising, object filtering, overlap-based merging.

**Must always run from frame 0** — the matching/merging loop is incremental. Frame N depends on accumulated state from frames 0 through N-1.

**Matching pipeline:**

| Step | Function | Output |
|------|----------|--------|
| Spatial similarity | `compute_spatial_similarities()` | (M, N) IoU matrix |
| Visual similarity | `compute_visual_similarities()` | (M, N) cosine sim matrix |
| Aggregation | `aggregate_similarities()` | `(1+phys_bias)*spatial + (1-phys_bias)*visual` |
| Matching | `match_detections_to_objects()` | Per-detection: matched object index or None |
| Merging | `merge_obj_matches()` | Updated MapObjectList with per_view_records |

**3D bbox IoU OR-condition:** For sparse mode, `match_detections_to_objects` supports an optional IoU merge fallback (`iou_merge_kappa`, default 0.25). If the aggregated similarity is below threshold but the 3D bounding box IoU exceeds kappa, the detection is still merged.

**Outputs:** `map/oracle_map.pkl.gz` — intermediate format containing live Open3D objects. This is the only pickle in the pipeline; `oracle_finalize.py` converts it to npz+JSON.

---

### A4: oracle_finalize.py

**Code:** `semgraph/stages/oracle_finalize.py`

**What it does:**
1. Loads the intermediate map from `build_map.py`.
2. **MST edge construction:** Computes 3D bounding box IoU between all object pairs, builds a maximum-weight minimum spanning tree via scipy. Outputs unlabeled candidate edges (no LLM involved).
3. **HPSG plane detection:** Runs RANSAC plane fitting on the concatenated scene point cloud (up to 5 major planes). Classifies planes as floor/wall/ceiling by normal alignment. Anchors each object to its nearest plane.
4. Converts live objects to an `OracleSceneRecord` with all arrays in offset pattern and saves as npz+JSON.

**Outputs:** `oracle/oracle_scene.npz+.json` — the immutable oracle scene. Contains:
- Per-object: PCD points/colors (offsets), bbox corners, per_view_records clip features (offsets)
- Metadata: class names, parent_plane_ids, plane records, MST edges, per_view_records (frame_idx, n_points, crop_path)

---

## 3. Phase B — Semantic Evaluation

Phase B runs N times per scene — once per encoder/VLM combination being evaluated. It reads from the immutable oracle scene and produces scored outputs.

### B1: embed.py (re-embed mode)

**Code:** `semgraph/stages/embed.py` with `embed.mode=re_embed`

**What it does:**
1. Loads the oracle scene and a specified evaluation encoder.
2. For each object's `per_view_records`: loads crop images, batch-encodes features.
3. Computes three feature representations:
   - **Weighted average** (by `n_points` per view)
   - **Per-view features** (full set)
   - **Entropy-selected best** (the single view whose feature has lowest softmax entropy over a label set)
4. Saves as a `VariantRecord`.

**Outputs:** `variants/embed_{enc_slug}.npz+.json`

---

### B2: caption.py

**Code:** `semgraph/stages/caption.py`

**What it does:**
1. Loads the oracle scene's `per_view_meta` (crop paths and n_points). Does NOT load point clouds or any geometry.
2. For each object: sorts views by `n_points` descending, takes top K, sends 1.5x crop images to VLM with three prompts (caption, color, material).
3. Runs LLM consolidation to produce `canonical_tag`, `candidate_tags`, `summary`.
4. Saves as a `CaptionsRecord`.

**VLM dependency:** Requires a running vLLM server at `vlm_api_url`. Configurable via `caption.vlm_name`.

**Outputs:** `captions/{vlm_slug}/captions.npz+.json`

---

### B3: semantic_assemble.py

**Code:** `semgraph/stages/semantic_assemble.py`

**What it does:**
1. Loads the oracle scene (geometry), embed variant (features), and captions.
2. **LLM edge labeling:** For each unlabeled MST edge, sends object tags and positions to the VLM and gets a spatial relationship (on, supports, in, contains, next to, none).
3. **Scene type inference:** Sends the list of all canonical tags to the VLM to infer room type.
4. **Plane captions:** Templates each plane as "This is a {label} in the {scene_type}."
5. Assembles the final HPSG JSON.

**Outputs:** `assembled/{enc_slug}_{vlm_slug}/scene_graph.json`

---

### B4: eval.py

**Code:** `semgraph/stages/eval.py`

**What it does:** Reads the HPSG JSON and runs three evaluation modes:

| Mode | Metrics | Notes |
|------|---------|-------|
| Classification | mIoU, F-mIoU, mAcc | Against 1687-label eval list with 17-category grouping |
| QA | Framework ready (EM@1, BLEU, ROUGE-L, METEOR, CIDEr) | Requires external question sets (ScanQA, Space3D-Bench) |
| Retrieval | Framework ready (recall@K) | Max-over-views similarity from embed variant |

**Outputs:** `eval/{enc_slug}_{vlm_slug}/classification.json`

---

## 4. Serialization Architecture

### Three-Layer Design

```
Layer 1: Records         semgraph/io/records.py
          Typed dataclasses with to_arrays() / to_metadata() / from_arrays_and_metadata()

Layer 2: Serializer      semgraph/io/serializers/npz.py
          NpzSerializer: np.savez_compressed + JSON sidecar

Layer 3: Loaders         semgraph/io/loaders.py
          One save_*/load_* pair per record type
```

Records never import serializers. Serializers never import records.

### Record Types

| Record | Producer | Consumer | Arrays | Metadata |
|--------|----------|----------|--------|----------|
| `RawDetRecord` | detect.py | embed.py | xyxy, confidence, class_id, masks (flat + offsets + shapes) | classes, labels, captions |
| `FrameDataRecord` | detect.py, embed.py | build_map.py | pcd_points + offsets, pcd_colors + offsets, bbox_corners, pose, intrinsics, clip_ft, text_ft | frame_idx, color_path, H, W, per-detection metadata |
| `CaptionsRecord` | caption.py | semantic_assemble.py | (placeholder) | per-object canonical_tag, candidate_tags, summary, color, material |
| `OracleSceneRecord` | oracle_finalize.py | embed.py (B), caption.py, semantic_assemble.py | per-object PCD points/colors + offsets, bbox corners, per-view clip_ft + offsets | class_names, planes, mst_edges, parent_plane_ids, per_view_meta |
| `VariantRecord` | embed.py (B) | semantic_assemble.py, eval.py | clip_ft_weighted_avg, clip_ft_best, per-view features + offsets, best_entropy | encoder_name |

### Offset Pattern

Variable-length arrays (e.g., point clouds differ per detection) are stored flat with an offsets array:

```python
# Save
all_points = np.concatenate([det.pcd_points for det in detections])
offsets = np.cumsum([0] + [len(det.pcd_points) for det in detections])
# Arrays: {"pcd_points": all_points, "pcd_offsets": offsets}

# Load
per_det_points = [all_points[offsets[i]:offsets[i+1]] for i in range(n_dets)]
```

---

## 5. Data Flow Diagram

```
                        PHASE A (geometry only, run once)
                        ─────────────────────────────────

  Dataset ──► detect.py ──► embed.py (oracle) ──► build_map.py ──► oracle_finalize.py
                │                │                      │                   │
                ▼                ▼                      ▼                   ▼
          raw_det/*.npz    frame_data/*.npz       map/oracle_map      oracle/oracle_scene
          crops/*.jpg      (+ clip_ft, text_ft)   (.pkl.gz, temp)     (.npz + .json)
                                                                           │
              ┌────────────────────────────────────────────────────────────┘
              │
              │         PHASE B (semantic, run N times)
              │         ───────────────────────────────
              ▼
         embed.py ────────► caption.py ────────► semantic_assemble.py ──► eval.py
         (re-embed)            │                        │                    │
              │                ▼                        ▼                    ▼
              ▼          captions/{vlm}/         assembled/{e}_{v}/    eval/{e}_{v}/
        variants/        captions.npz+.json     scene_graph.json      classification.json
        embed_{enc}
        .npz+.json
```

---

## 6. Shell Orchestration

The full pipeline is orchestrated by `shells/run_staged_pipeline.sh`:

```bash
# Phase A (all Hydra overrides forwarded)
python -m semgraph.stages.detect "$@"
python -m semgraph.stages.embed "$@"
python -m semgraph.stages.build_map "$@"
python -m semgraph.stages.oracle_finalize "$@"

# Phase B (encoder and VLM configurable via env vars)
ENCODER="${ENCODER:-openai/clip-vit-large-patch14}"
VLM="${VLM:-Qwen/Qwen3-VL-2B-Instruct}"

python -m semgraph.stages.embed "$@" embed.mode=re_embed "embed.encoder_name=${ENCODER}"
python -m semgraph.stages.caption "$@" "caption.vlm_name=${VLM}"
python -m semgraph.stages.semantic_assemble "$@" "assemble.encoder=${SAFE_ENC}" "assemble.vlm=${SAFE_VLM}"
python -m semgraph.stages.eval "$@" "eval.encoder=${SAFE_ENC}" "eval.vlm=${SAFE_VLM}"
```

To evaluate with multiple encoders/VLMs, run Phase B repeatedly with different `ENCODER` and `VLM` values. Phase A does not need to be re-run.

---

## 7. Stage Output Artifacts

All artifacts live under `{dataset_root}/{scene_id}/stages/`:

```
stages/
    raw_detections/
        000000.npz + 000000.json          # RawDetRecord per frame
    frame_data/
        000000.npz + 000000.json          # FrameDataRecord per frame
    crops/
        000000_000.jpg                     # 1.5x crop per detection
    map/
        oracle_map.pkl.gz                  # intermediate (live o3d objects)
    oracle/
        oracle_scene.npz + oracle_scene.json   # immutable oracle
    captions/
        {vlm_slug}/
            captions.npz + captions.json   # CaptionsRecord
    variants/
        embed_{enc_slug}.npz + .json       # VariantRecord
    assembled/
        {enc_slug}_{vlm_slug}/
            scene_graph.json               # HPSG JSON
    eval/
        {enc_slug}_{vlm_slug}/
            classification.json            # eval results
```

---

## 8. HPSG Output Schema

The `scene_graph.json` produced by `semantic_assemble.py`:

```json
{
  "scene_type": "office",
  "objects": [
    {
      "id": 0,
      "bbox_extent": [0.45, 0.52, 0.88],
      "bbox_center": [1.23, -0.15, 0.44],
      "object_tag": "office chair",
      "caption": "A black rolling office chair with armrests",
      "color": "black",
      "material": "plastic",
      "candidate_tags": ["desk chair", "swivel chair", "task chair"],
      "best_entropy": 1.23,
      "n_views": 12,
      "parent_plane_id": 0
    }
  ],
  "planes": [
    {
      "plane_id": 0,
      "label": "floor",
      "caption": "This is a floor in the office.",
      "normal": [0.0, 0.0, 1.0],
      "offset": -0.02
    }
  ],
  "edges": [
    {
      "source": 0,
      "target": 1,
      "relationship": "next to",
      "iou_score": 0.15
    }
  ]
}
```

---

## 9. Legacy All-in-One Mode

The monolith at `semgraph/slam/vlm_run/batch_vlm_mapping_api.py` still exists and runs the entire pipeline in a single process/loop. It delegates to the same stage modules (`detect`, `caption`, `build_map`, `postprocess`) but executes everything sequentially within one frame loop.

The staged pipeline is the recommended execution model. The monolith is retained for backward compatibility and single-process debugging.
