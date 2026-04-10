"""
Stage A1 — Segmentation + 3D lifting + 1.5x crop saving.

Geometry-only stage: loads SAM/YOLO for segmentation, lifts masks to 3D
via the geometry backend, computes 1.5x projected crops and saves them to
disk.  No CLIP, no VLM, no language models.

Saves camera pose, intrinsics, H, W in each FrameDataRecord so downstream
stages (embed.py, build_map.py, etc.) never need the geometry backend.

Standalone usage::

    python -m conceptgraph.stages.detect <hydra overrides>
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import supervision as sv
import torch

from PIL import Image

from conceptgraph.slam.geometry.base import FrameContext
from conceptgraph.slam.geometry.projection import compute_projected_crop_bbox
from conceptgraph.slam.utils import (
    filter_gobs,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    resize_gobs,
)
from conceptgraph.utils.ious import mask_subtract_contained


# ---------------------------------------------------------------------------
# SAM Automatic Mask Filtering (moved from monolith)
# ---------------------------------------------------------------------------

def filter_sam_auto_masks(
    masks_np: np.ndarray,
    xyxy_np: np.ndarray,
    confidences: np.ndarray,
    image_rgb: np.ndarray,
    min_area_pixels: int = 100,
    max_area_fraction: float = 0.95,
    nms_iou_threshold: float = 0.7,
) -> tuple:
    """Filter SAM automatic masks by area bounds and apply NMS."""
    if masks_np.shape[0] == 0:
        return masks_np, xyxy_np, confidences

    H, W = image_rgb.shape[:2]
    image_area = H * W
    max_area_pixels = max_area_fraction * image_area

    keep = []
    for i in range(masks_np.shape[0]):
        area = masks_np[i].sum()
        if min_area_pixels <= area <= max_area_pixels:
            keep.append(i)

    if not keep:
        return (
            np.empty((0, H, W), dtype=np.bool_),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    masks_np = masks_np[keep]
    xyxy_np = xyxy_np[keep]
    confidences = confidences[keep]

    if nms_iou_threshold < 1.0 and len(xyxy_np) > 1:
        order = np.argsort(-confidences)
        nms_keep = []
        suppressed = set()
        for idx in order:
            if idx in suppressed:
                continue
            nms_keep.append(idx)
            x1_a, y1_a, x2_a, y2_a = xyxy_np[idx]
            area_a = (x2_a - x1_a) * (y2_a - y1_a)
            for jdx in order:
                if jdx in suppressed or jdx == idx:
                    continue
                x1_b, y1_b, x2_b, y2_b = xyxy_np[jdx]
                inter_x1, inter_y1 = max(x1_a, x1_b), max(y1_a, y1_b)
                inter_x2, inter_y2 = min(x2_a, x2_b), min(y2_a, y2_b)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_b = (x2_b - x1_b) * (y2_b - y1_b)
                union = area_a + area_b - inter_area
                if union > 0 and inter_area / union > nms_iou_threshold:
                    suppressed.add(jdx)
        nms_keep = sorted(nms_keep)
        masks_np = masks_np[nms_keep]
        xyxy_np = xyxy_np[nms_keep]
        confidences = confidences[nms_keep]

    return masks_np, xyxy_np, confidences


# ---------------------------------------------------------------------------
# Models container
# ---------------------------------------------------------------------------

@dataclass
class DetectionModels:
    """Holds segmentation models only — no encoders or VLMs."""
    sam_predictor: Any = None
    detection_model: Any = None      # YOLO-World (yolo_sam mode)
    seg_backend: str = "sam_auto"


def load_models(cfg: Any) -> DetectionModels:
    """Load SAM and YOLO (if yolo_sam). No CLIP or VLM encoder."""
    from conceptgraph.utils.general_utils import measure_time

    seg_backend = cfg.get("segmentation_backend", "sam_auto")
    models = DetectionModels(seg_backend=seg_backend)

    if seg_backend != "gt_instances":
        from ultralytics import SAM, YOLO

        ckpt_dir = os.environ.get("CKPT_DIR", "")
        sam_weights = "sam2.1_b.pt"
        if ckpt_dir and (Path(ckpt_dir) / sam_weights).exists():
            sam_weights = str(Path(ckpt_dir) / sam_weights)
        models.sam_predictor = SAM(sam_weights)

        if seg_backend == "yolo_sam":
            yolo_weights = "yolov8l-worldv2.pt"
            if ckpt_dir and (Path(ckpt_dir) / yolo_weights).exists():
                yolo_weights = str(Path(ckpt_dir) / yolo_weights)
            models.detection_model = measure_time(YOLO)(yolo_weights)

    return models


# ---------------------------------------------------------------------------
# Per-frame processing
# ---------------------------------------------------------------------------

def process_frame(
    frame_ctx: FrameContext,
    models: DetectionModels,
    cfg: Any,
    backend: Any,
    obj_classes: Any,
) -> tuple[dict | None, list | None, np.ndarray | None]:
    """Run detection + features + filtering + 3D lifting for one frame.

    Returns
    -------
    raw_gobs : dict (RawGobs) or None if frame should be skipped
    detection_list : list or None
    surviving_indices : np.ndarray or None — maps filtered idx -> raw idx
    """
    image_rgb = frame_ctx.image_rgb
    color_path = frame_ctx.color_path

    # ----- gt_instances: raw_gobs already built by the iterator -----
    if frame_ctx.skip_segmentation:
        raw_gobs = frame_ctx.extra.get("raw_gobs")
        if raw_gobs is None:
            return None, None, None
    else:
        raw_gobs = _run_detection(frame_ctx, models, cfg, obj_classes)
        if raw_gobs is None:
            return None, None, None

    # ----- Filter + lift -----
    resized_gobs = resize_gobs(raw_gobs, image_rgb)
    filtered_gobs = filter_gobs(
        resized_gobs,
        image_rgb,
        skip_bg=cfg.skip_bg,
        BG_CLASSES=obj_classes.get_bg_classes_arr(),
        mask_area_threshold=cfg.mask_area_threshold,
        max_bbox_area_ratio=cfg.max_bbox_area_ratio,
        mask_conf_threshold=cfg.mask_conf_threshold,
    )

    if len(filtered_gobs["mask"]) == 0:
        return raw_gobs, None, None

    # Track which raw indices survived filtering
    n_raw = len(raw_gobs["mask"])
    n_filtered = len(filtered_gobs["mask"])
    surviving_indices = _compute_surviving_indices(raw_gobs, filtered_gobs, n_raw, n_filtered)

    filtered_gobs["mask"] = mask_subtract_contained(filtered_gobs["xyxy"], filtered_gobs["mask"])

    obj_pcds_and_bboxes = backend.lift_to_3d(filtered_gobs["mask"], frame_ctx, cfg)

    for obj in obj_pcds_and_bboxes:
        if obj:
            obj["pcd"] = init_process_pcd(
                pcd=obj["pcd"],
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
            )
            obj["bbox"] = get_bounding_box(
                spatial_sim_type=cfg["spatial_sim_type"],
                pcd=obj["pcd"],
            )

    detection_list = make_detection_list_from_pcd_and_gobs(
        obj_pcds_and_bboxes, filtered_gobs, color_path, obj_classes, frame_ctx.frame_idx
    )

    return raw_gobs, detection_list if len(detection_list) > 0 else None, surviving_indices


def _compute_surviving_indices(raw_gobs, filtered_gobs, n_raw, n_filtered):
    """Best-effort mapping from filtered indices back to raw indices.

    Uses xyxy bounding-box matching as a proxy since filter_gobs may
    reorder or drop entries.
    """
    if n_filtered == 0:
        return np.array([], dtype=np.int32)

    raw_xyxy = raw_gobs["xyxy"]
    filt_xyxy = filtered_gobs["xyxy"]
    surviving = np.arange(n_filtered, dtype=np.int32)

    if n_filtered <= n_raw:
        indices = []
        used = set()
        for fi in range(n_filtered):
            best_ri = fi  # default: same position
            best_dist = float("inf")
            for ri in range(n_raw):
                if ri in used:
                    continue
                dist = np.sum(np.abs(filt_xyxy[fi] - raw_xyxy[ri]))
                if dist < best_dist:
                    best_dist = dist
                    best_ri = ri
            indices.append(best_ri)
            used.add(best_ri)
        surviving = np.array(indices, dtype=np.int32)

    return surviving


def _run_detection(
    frame_ctx: FrameContext,
    models: DetectionModels,
    cfg: Any,
    obj_classes: Any,
) -> dict | None:
    """Run segmentation only (no CLIP, no VLM). Returns geometry-only RawGobs."""
    color_path = frame_ctx.color_path
    image_rgb = frame_ctx.image_rgb

    seg_backend = models.seg_backend

    if seg_backend == "yolo_sam":
        masks_np, xyxy_np, confidences, detection_class_ids, detection_class_labels, classes_arr = (
            _segment_yolo_sam(models, color_path, image_rgb, obj_classes)
        )
    else:
        masks_np, xyxy_np, confidences, detection_class_ids, detection_class_labels, classes_arr = (
            _segment_sam_auto(models, color_path, image_rgb, cfg)
        )

    if masks_np.shape[0] == 0:
        return None

    return {
        "xyxy": xyxy_np,
        "confidence": confidences,
        "class_id": detection_class_ids,
        "mask": masks_np,
        "classes": classes_arr,
        "image_crops": None,
        "image_feats": None,
        "text_feats": None,
        "detection_class_labels": detection_class_labels,
        "labels": detection_class_labels,
        "edges": [],
        "captions": [""] * len(xyxy_np),
        "vlm_vit_feats": None,
        "vlm_proj_feats": None,
    }


def _segment_yolo_sam(models, color_path, image_rgb, obj_classes):
    """YOLO-World + SAM box-prompted segmentation."""
    results = models.detection_model.predict(color_path, conf=0.1, verbose=False)
    confidences = results[0].boxes.conf.cpu().numpy()
    detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    detection_class_labels = [
        f"{obj_classes.get_classes_arr()[cid]} {ci}"
        for ci, cid in enumerate(detection_class_ids)
    ]
    xyxy_tensor = results[0].boxes.xyxy
    xyxy_np = xyxy_tensor.cpu().numpy()

    if xyxy_tensor.numel() != 0:
        sam_out = models.sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.detach().cpu().numpy()
        if masks_np.dtype != np.bool_:
            masks_np = masks_np > 0.5

        n = min(xyxy_np.shape[0], masks_np.shape[0])
        if n == 0:
            H, W = image_rgb.shape[:2]
            return np.empty((0, H, W), dtype=np.bool_), np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), [], obj_classes.get_classes_arr()
        xyxy_np, confidences, detection_class_ids, masks_np = xyxy_np[:n], confidences[:n], detection_class_ids[:n], masks_np[:n]
    else:
        H, W = image_rgb.shape[:2]
        return np.empty((0, H, W), dtype=np.bool_), np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), [], obj_classes.get_classes_arr()

    return masks_np, xyxy_np, confidences, detection_class_ids, detection_class_labels, obj_classes.get_classes_arr()


def _segment_sam_auto(models, color_path, image_rgb, cfg):
    """SAM automatic mask generation (class-agnostic)."""
    sam_results = models.sam_predictor.predict(color_path, verbose=False)
    r = sam_results[0]

    H, W = image_rgb.shape[:2]
    if r.masks is not None and r.masks.data.numel() > 0:
        masks_np = r.masks.data.detach().cpu().numpy()
        if masks_np.dtype != np.bool_:
            masks_np = masks_np > 0.5
        xyxy_np = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(xyxy_np), dtype=np.float32)

        masks_np, xyxy_np, confidences = filter_sam_auto_masks(
            masks_np, xyxy_np, confidences, image_rgb,
            min_area_pixels=cfg.get("sam_auto_min_mask_area_pixels", 100),
            max_area_fraction=cfg.get("sam_auto_max_mask_area_fraction", 0.95),
            nms_iou_threshold=cfg.get("sam_auto_nms_iou_threshold", 0.7),
        )
    else:
        masks_np = np.empty((0, H, W), dtype=np.bool_)
        xyxy_np = np.empty((0, 4), dtype=np.float32)
        confidences = np.empty((0,), dtype=np.float32)

    detection_class_ids = np.zeros(len(xyxy_np), dtype=np.int32)
    detection_class_labels = [f"object {i}" for i in range(len(xyxy_np))]
    classes_arr = ["object"]

    return masks_np, xyxy_np, confidences, detection_class_ids, detection_class_labels, classes_arr


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _save_crop(image_rgb: np.ndarray, crop_bbox: tuple, crop_path: Path) -> None:
    """Save a 1.5x projected crop to disk as JPEG."""
    x_min, y_min, x_max, y_max = crop_bbox
    crop = image_rgb[y_min:y_max, x_min:x_max]
    pil = Image.fromarray(crop)
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(crop_path, quality=95)


def main_standalone(cfg):
    """Standalone detection stage — writes raw_det + frame_data + crops."""
    from tqdm import tqdm
    from conceptgraph.stages.paths import stage_paths, FrameDataRecord
    from conceptgraph.stages import io as stage_io
    from conceptgraph.slam.geometry import get_geometry_backend
    from conceptgraph.slam.utils import process_cfg
    from conceptgraph.utils.general_utils import ObjectClasses, cfg_to_dict

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)

    backend = get_geometry_backend(cfg.get("pipeline_mode", "trajectory"))
    geo_ctx = backend.load(cfg)
    models = load_models(cfg)

    det_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=det_cfg["classes_file"],
        bg_classes=det_cfg["bg_classes"],
        skip_bg=det_cfg["skip_bg"],
    )
    if models.seg_backend == "yolo_sam" and models.detection_model is not None:
        models.detection_model.set_classes(obj_classes.get_classes_arr())

    skip_existing = cfg.get("skip_existing_detections", False)

    for frame_ctx in tqdm(
        backend.get_iterator(geo_ctx),
        total=backend.num_iterations(geo_ctx),
        desc="detect",
    ):
        if skip_existing:
            existing = paths["raw_det"] / f"{frame_ctx.frame_idx:06d}.pkl.gz"
            if existing.is_file():
                continue

        raw_gobs, det_list, surviving = process_frame(
            frame_ctx, models, cfg, backend, obj_classes
        )
        if raw_gobs is not None:
            stage_io.save_raw_det(paths["raw_det"], frame_ctx.frame_idx, raw_gobs)
        if det_list is not None and len(det_list) > 0 and surviving is not None:
            pose = frame_ctx.pose if frame_ctx.pose is not None else np.eye(4)
            intrinsics = frame_ctx.intrinsics if frame_ctx.intrinsics is not None else np.eye(3)
            H, W = frame_ctx.image_rgb.shape[:2]

            serialized_dets = []
            for det_idx, det in enumerate(det_list):
                pcd_pts = np.asarray(det["pcd"].points)
                crop_bbox = compute_projected_crop_bbox(
                    pcd_pts, pose, intrinsics, H, W, scale=1.5,
                )
                crop_rel = ""
                if crop_bbox is not None:
                    crop_fname = f"{frame_ctx.frame_idx:06d}_{det_idx:03d}.jpg"
                    crop_abs = paths["crops"] / crop_fname
                    _save_crop(frame_ctx.image_rgb, crop_bbox, crop_abs)
                    crop_rel = str(crop_abs)

                det["crop_path"] = crop_rel
                serialized_dets.append(
                    stage_io.serialize_detection(det, cfg.get("spatial_sim_type", "iou"))
                )

            record = FrameDataRecord(
                frame_idx=frame_ctx.frame_idx,
                color_path=str(frame_ctx.color_path),
                skip_matching=frame_ctx.skip_matching,
                surviving_indices=surviving,
                detections=serialized_dets,
                pose=pose,
                intrinsics=intrinsics,
                H=H,
                W=W,
            )
            stage_io.save_frame_data(paths["frame_data"], frame_ctx.frame_idx, record)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
