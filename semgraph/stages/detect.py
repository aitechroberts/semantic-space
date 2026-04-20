"""
Stage A1 — Segmentation + 3D lifting + 1.5x crop saving.

Geometry-only stage: loads SAM/YOLO for segmentation, lifts masks to 3D
via the geometry backend, computes 1.5x projected crops and saves them to
disk.  No CLIP, no VLM, no language models.

Saves camera pose, intrinsics, H, W in each FrameDataRecord so downstream
stages (embed.py, build_map.py, etc.) never need the geometry backend.

Standalone usage::

    python -m semgraph.stages.detect <hydra overrides>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
import torch

from PIL import Image

from semgraph.detection.base import Detector, Segmenter
from semgraph.slam.geometry.base import FrameContext
from semgraph.slam.geometry.projection import compute_projected_crop_bbox
from semgraph.slam.utils import (
    filter_gobs,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    resize_gobs,
)
from semgraph.utils.ious import mask_subtract_contained


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

import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionModels:
    """Holds detection/segmentation backends — no encoders or VLMs."""
    detector: Detector | None = None
    segmenter: Segmenter | None = None
    seg_backend: str = "sam_auto"

    @property
    def has_class_labels(self) -> bool:
        """True when a detector is present and producing class labels."""
        return self.detector is not None

    @property
    def is_vocab_driven(self) -> bool:
        """True when the detector maps detections to a fixed vocabulary."""
        return self.detector is not None and self.detector.vocab_driven

    @property
    def vocabulary(self) -> list[str] | None:
        """The detector's class vocabulary, if any."""
        return self.detector.classes if self.detector is not None else None


# ---------------------------------------------------------------------------
# Backend / legacy mapping
# ---------------------------------------------------------------------------

_SEGMENTER_MAP: dict[str, tuple[str, str]] = {
    "sam_auto":    ("sam",  "sam2.1_b.pt"),
    "sam3_auto":   ("sam3", "sam3.pt"),
    "detect_sam":  ("sam",  "sam2.1_b.pt"),
    "detect_sam3": ("sam3", "sam3.pt"),
}

_LEGACY_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "yolo_sam":       ("detect_sam",  "yolo_world"),
    "yoloe_sam":      ("detect_sam",  "yoloe"),
    "florence2_sam":  ("detect_sam",  "florence2"),
    "yolo_sam3":      ("detect_sam3", "yolo_world"),
    "yoloe_sam3":     ("detect_sam3", "yoloe"),
    "florence2_sam3": ("detect_sam3", "florence2"),
}


def _apply_legacy_shim(
    seg_backend: str, cfg: Any,
) -> tuple[str, Any]:
    """Map deprecated backend strings to the new (detect_* + detector_type) form."""
    if seg_backend not in _LEGACY_BACKEND_MAP:
        return seg_backend, cfg

    new_backend, det_type = _LEGACY_BACKEND_MAP[seg_backend]
    logger.warning(
        "segmentation_backend='%s' is deprecated. "
        "Use segmentation_backend='%s' detector_type='%s' instead.",
        seg_backend, new_backend, det_type,
    )

    if not cfg.get("detector_type"):
        try:
            from omegaconf import OmegaConf
            OmegaConf.update(cfg, "detector_type", det_type)
        except (ImportError, Exception):
            cfg["detector_type"] = det_type

    return new_backend, cfg


def load_models(cfg: Any, obj_classes: Any = None) -> DetectionModels:
    """Load detection and segmentation models via the ABC factories.

    Weight path resolution and config parsing happen here — model classes
    never see the Hydra config.
    """
    from semgraph.detection import get_detector, get_segmenter
    from semgraph.detection.base import resolve_weights

    seg_backend = cfg.get("segmentation_backend", "sam_auto")
    seg_backend, cfg = _apply_legacy_shim(seg_backend, cfg)

    if seg_backend == "gt_instances":
        return DetectionModels(seg_backend=seg_backend)

    if seg_backend not in _SEGMENTER_MAP:
        raise ValueError(
            f"Unknown segmentation_backend '{seg_backend}'. "
            f"Valid options: {', '.join(list(_SEGMENTER_MAP) + ['gt_instances'])}"
        )

    device = cfg.get("device", "cuda")

    seg_type, seg_default_weights = _SEGMENTER_MAP[seg_backend]
    segmenter = get_segmenter(seg_type)
    seg_weights = cfg.get("seg_weights_override", seg_default_weights)
    segmenter.load(weights=resolve_weights(seg_weights), device=device)

    detector = None
    if seg_backend.startswith("detect_"):
        detector_type = cfg.get("detector_type", "yoloe")
        detector_name = cfg.get("detector_name", None) or None

        load_kwargs: dict[str, Any] = {}
        if obj_classes is not None:
            load_kwargs["classes"] = obj_classes.get_classes_arr()

        detector = get_detector(
            detector_type=detector_type,
            detector_name=detector_name,
            device=device,
            **load_kwargs,
        )

    return DetectionModels(detector=detector, segmenter=segmenter, seg_backend=seg_backend)


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
    """Run detection + segmentation via ABC backends. Returns geometry-only RawGobs."""
    image_rgb = frame_ctx.image_rgb
    color_path = frame_ctx.color_path

    if models.detector is not None:
        # Detector produces boxes; segmenter produces masks prompted by those boxes
        det_result = models.detector.detect(image_rgb, color_path=color_path)
        seg_result = models.segmenter.segment(
            image_rgb, boxes=det_result.xyxy, color_path=color_path,
        )
        n = min(len(det_result.xyxy), len(seg_result.masks))
        if n == 0:
            return None
        masks_np = seg_result.masks[:n]
        xyxy_np = det_result.xyxy[:n]
        confidences = det_result.confidence[:n]
        detection_class_ids = det_result.class_ids[:n]
        detection_class_labels = det_result.class_labels[:n]
        classes_arr = det_result.classes
    else:
        # Auto mode: segmenter finds everything, then pipeline filters
        seg_result = models.segmenter.segment(image_rgb, boxes=None, color_path=color_path)
        masks_np, xyxy_np, confidences = filter_sam_auto_masks(
            seg_result.masks, seg_result.xyxy, seg_result.confidence, image_rgb,
            min_area_pixels=cfg.get("sam_auto_min_mask_area_pixels", 100),
            max_area_fraction=cfg.get("sam_auto_max_mask_area_fraction", 0.95),
            nms_iou_threshold=cfg.get("sam_auto_nms_iou_threshold", 0.7),
        )
        detection_class_ids = np.zeros(len(xyxy_np), dtype=np.int32)
        detection_class_labels = [f"object {i}" for i in range(len(xyxy_np))]
        classes_arr = ["object"]

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
    from semgraph.stages.paths import stage_paths
    from semgraph.io import (
        FrameDataRecord,
        save_raw_det,
        save_frame_data,
        serialize_detection,
    )
    from semgraph.io.records import _DetectionMeta
    from semgraph.slam.geometry import get_geometry_backend
    from semgraph.slam.utils import process_cfg
    from semgraph.utils.general_utils import ObjectClasses, cfg_to_dict

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    save_raw = cfg.get("save_raw_detections", False)
    for key, d in paths.items():
        if key == "raw_det" and not save_raw:
            continue
        d.mkdir(parents=True, exist_ok=True)

    backend = get_geometry_backend(cfg.get("pipeline_mode", "trajectory"))
    geo_ctx = backend.load(cfg)

    # --- frame selection ---
    from semgraph.sampling import get_frame_selector

    fs_cfg = cfg.get("frame_selection", {"method": "stride", "stride": cfg.get("stride", 10)})
    selector = get_frame_selector(fs_cfg["method"])
    all_poses = backend.get_poses(geo_ctx)
    selection = selector.select(
        all_poses, **{k: v for k, v in fs_cfg.items() if k != "method"}
    )
    selected_set = set(selection.frame_indices.tolist())

    import json as _json

    meta_path = paths["frame_data"] / "_selection_metadata.json"
    with open(meta_path, "w") as _f:
        _json.dump(
            {"method": selection.method, **selection.metadata,
             "n_selected": len(selection.frame_indices)},
            _f, indent=2,
        )

    det_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=det_cfg["classes_file"],
        bg_classes=det_cfg["bg_classes"],
        skip_bg=det_cfg["skip_bg"],
    )
    models = load_models(cfg, obj_classes=obj_classes)

    skip_existing = cfg.get("skip_existing_detections", False)

    for frame_ctx in tqdm(
        backend.get_iterator(geo_ctx),
        total=backend.num_iterations(geo_ctx),
        desc="detect",
    ):
        if frame_ctx.frame_idx not in selected_set:
            continue

        if skip_existing:
            existing = paths["frame_data"] / f"{frame_ctx.frame_idx:06d}.npz"
            if existing.is_file():
                continue

        raw_gobs, det_list, surviving = process_frame(
            frame_ctx, models, cfg, backend, obj_classes
        )
        if raw_gobs is not None and save_raw:
            save_raw_det(paths["raw_det"], frame_ctx.frame_idx, raw_gobs)
        if det_list is not None and len(det_list) > 0 and surviving is not None:
            pose = frame_ctx.pose if frame_ctx.pose is not None else np.eye(4)
            intrinsics = frame_ctx.intrinsics if frame_ctx.intrinsics is not None else np.eye(3)
            H, W = frame_ctx.image_rgb.shape[:2]

            pcd_points_list = []
            pcd_colors_list = []
            bbox_corners_list = []
            det_meta_list = []

            for det_idx, det in enumerate(det_list):
                pcd_pts = np.asarray(det["pcd"].points)
                pcd_cols = np.asarray(det["pcd"].colors) if det["pcd"].has_colors() else np.zeros_like(pcd_pts)
                bbox_corners = np.asarray(det["bbox"].get_box_points())

                crop_bbox = compute_projected_crop_bbox(
                    pcd_pts, pose, intrinsics, H, W, scale=1.5,
                )
                crop_rel = ""
                if crop_bbox is not None:
                    crop_fname = f"{frame_ctx.frame_idx:06d}_{det_idx:03d}.jpg"
                    crop_abs = paths["crops"] / crop_fname
                    _save_crop(frame_ctx.image_rgb, crop_bbox, crop_abs)
                    crop_rel = str(crop_abs)

                pcd_points_list.append(pcd_pts)
                pcd_colors_list.append(pcd_cols)
                bbox_corners_list.append(bbox_corners)
                # Prefer the per-det fields (set by the GT backend in
                # _lift_gt_instance); fall back to the FrameContext's
                # instance_id for older producers that set it only there.
                gt_iid = det.get("gt_instance_id")
                if gt_iid is None and frame_ctx.instance_id is not None:
                    gt_iid = int(frame_ctx.instance_id)
                n_vis = det.get("n_visible")
                if n_vis is None:
                    n_vis = frame_ctx.extra.get("n_visible")
                det_meta_list.append(_DetectionMeta(
                    bbox_type="axis_aligned",
                    class_name=det.get("class_name", "object"),
                    class_id=int(det.get("class_id", [0])[0]) if isinstance(det.get("class_id"), list) else int(det.get("class_id", 0)),
                    inst_id=int(det.get("curr_obj_num", det_idx)),
                    n_points=len(pcd_pts),
                    crop_path=crop_rel,
                    gt_instance_id=int(gt_iid) if gt_iid is not None else None,
                    n_visible=int(n_vis) if n_vis is not None else None,
                ))

            record = FrameDataRecord(
                frame_idx=frame_ctx.frame_idx,
                color_path=str(frame_ctx.color_path),
                skip_matching=frame_ctx.skip_matching,
                H=H,
                W=W,
                n_raw_detections=len(raw_gobs["mask"]),
                pose=pose,
                intrinsics=intrinsics,
                surviving_indices=surviving,
                pcd_points_list=pcd_points_list,
                pcd_colors_list=pcd_colors_list,
                bbox_corners=np.stack(bbox_corners_list) if bbox_corners_list else np.empty((0, 8, 3)),
                det_meta=det_meta_list,
            )
            save_frame_data(paths["frame_data"], frame_ctx.frame_idx, record)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
