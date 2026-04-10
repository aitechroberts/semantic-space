"""
Stage path resolution and formalized detection schemas.

Provides:
  - ``stage_paths(cfg)`` — canonical output directories for each stage
  - ``RawGobs`` TypedDict — the 14-key detection contract
  - ``SerializedDetection`` TypedDict — disk-serializable per-detection record
  - ``FrameDataRecord`` TypedDict — per-frame output from detect.py
  - ``make_empty_gobs()`` — factory for GT-instance and test paths
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_output_base(cfg: Any) -> Path:
    """Resolve output directory with fallbacks: cfg.output_root > $OUTPUT_ROOT > cfg.dataset_root."""
    output_override = None
    if hasattr(cfg, "__contains__"):
        if "output_root" in cfg:
            output_override = cfg.get("output_root") if hasattr(cfg, "get") else getattr(cfg, "output_root", None)
    if not output_override:
        output_override = os.getenv("OUTPUT_ROOT")
    if output_override:
        return Path(output_override)
    return Path(cfg.dataset_root)


def _build_exp_path(base_root: Path, scene_id: str, exp_suffix: str, create: bool = True) -> Path:
    """Build experiment output path."""
    path = base_root / scene_id / "exps" / exp_suffix
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def stage_paths(cfg: Any) -> dict[str, Path]:
    """Return canonical output directories for each pipeline stage.

    Keys: ``raw_det``, ``frame_data``, ``crops``, ``captions``, ``map``,
    ``oracle_scene``, ``variants``, ``exp_out``.
    """
    base = _resolve_output_base(cfg)
    scene = cfg.scene_id
    stages = base / scene / "stages"
    return {
        "raw_det":       stages / "raw_detections",
        "frame_data":    stages / "frame_data",
        "crops":         stages / "crops",
        "captions":      stages / "captions",
        "map":           stages / "map",
        "oracle_scene":  stages / "oracle_scene",
        "variants":      stages / "variants",
        "exp_out":       _build_exp_path(base, scene, cfg.exp_suffix),
    }


# ---------------------------------------------------------------------------
# RawGobs — canonical detection schema (pre-filter)
# ---------------------------------------------------------------------------

class RawGobs(TypedDict, total=False):
    """Detection data per frame. Geometry fields are required; feature fields
    are optional (populated by embed.py, not detect.py)."""

    # --- required (set by detect.py) ---
    xyxy: np.ndarray                       # (N, 4) float32
    confidence: np.ndarray                 # (N,) float32
    class_id: np.ndarray                   # (N,) int32
    mask: np.ndarray                       # (N, H, W) bool
    classes: list[str]                     # vocabulary
    detection_class_labels: list[str]      # N labels
    labels: list[str]                      # N labels (from VLM or same as above)
    edges: list[tuple[str, str, str]]      # VLM-inferred relations
    captions: list[str]                    # N captions (empty strings if no VLM)
    # --- optional (set by embed.py) ---
    image_crops: list[Image.Image] | None  # N crops, or None
    image_feats: np.ndarray | None         # (N, D) float32, or None
    text_feats: np.ndarray | None          # (N, D) float32, or None
    vlm_vit_feats: np.ndarray | None       # (N, D_vlm) or None
    vlm_proj_feats: np.ndarray | None      # (N, D_vlm) or None


def make_empty_gobs(
    n_detections: int,
    feat_dim: int = 512,
    *,
    xyxy: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    classes: list[str] | None = None,
    class_names: list[str] | None = None,
) -> RawGobs:
    """Create a RawGobs dict with zero-filled features and empty captions.

    This is the ONLY place that constructs a RawGobs from scratch outside
    of the detection pipeline.  Both the ``gt_instances`` path and test
    fixtures must use this factory to stay in sync with the schema.

    Parameters
    ----------
    n_detections : int
        Number of detections.
    feat_dim : int
        Dimensionality for CLIP feature vectors.
    xyxy : (N, 4) float32, optional
        Bounding boxes. Defaults to zeros.
    mask : (N, H, W) bool, optional
        Segmentation masks. Defaults to empty (N, 0, 0).
    classes : list[str], optional
        Vocabulary list. Defaults to ``["object"]``.
    class_names : list[str], optional
        Per-detection label names. Defaults to ``["object 0", ...]``.
    """
    N = n_detections
    if xyxy is None:
        xyxy = np.zeros((N, 4), dtype=np.float32)
    if mask is None:
        mask = np.zeros((N, 0, 0), dtype=bool)
    if classes is None:
        classes = ["object"]
    if class_names is None:
        class_names = [f"object {i}" for i in range(N)]

    return RawGobs(
        xyxy=xyxy,
        confidence=np.ones(N, dtype=np.float32),
        class_id=np.zeros(N, dtype=np.int32),
        mask=mask,
        classes=classes,
        image_crops=None,
        image_feats=np.zeros((N, feat_dim), dtype=np.float32),
        text_feats=np.zeros((N, feat_dim), dtype=np.float32),
        detection_class_labels=list(class_names),
        labels=list(class_names),
        edges=[],
        captions=[""] * N,
        vlm_vit_feats=None,
        vlm_proj_feats=None,
    )


# ---------------------------------------------------------------------------
# SerializedDetection — disk-safe per-detection record
# ---------------------------------------------------------------------------

class SerializedDetection(TypedDict, total=False):
    """Numpy-only representation of a single detection, safe for pickle.

    Geometry fields (pcd_points, bbox_corners, etc.) are always present.
    Feature fields (clip_ft, text_ft) are ``None`` after detect.py and
    populated by embed.py.
    """

    # --- always present (from detect.py) ---
    pcd_points: np.ndarray       # (P, 3) float64
    pcd_colors: np.ndarray       # (P, 3) float64
    bbox_corners: np.ndarray     # (8, 3) float64
    bbox_type: str               # "axis_aligned" or "oriented"
    class_name: str
    class_id: int
    inst_id: int                 # detection index within this frame
    n_points: int                # len(pcd_points), for quick filtering
    crop_path: str               # path to saved 1.5x crop image
    # --- optional (populated by embed.py) ---
    clip_ft: np.ndarray | None   # (D,) float32
    text_ft: np.ndarray | None   # (D,) float32
    vlm_vit_ft: np.ndarray | None
    vlm_proj_ft: np.ndarray | None


# ---------------------------------------------------------------------------
# FrameDataRecord — per-frame output from detect.py
# ---------------------------------------------------------------------------

class FrameDataRecord(TypedDict):
    """What detect.py saves for downstream stages: filtered detections + camera."""

    frame_idx: int
    color_path: str
    skip_matching: bool
    surviving_indices: np.ndarray          # (K,) int — maps filtered idx -> raw idx
    detections: list[SerializedDetection]  # K typed detection dicts
    # Camera metadata — so embed.py / build_map.py never need the geometry backend
    pose: np.ndarray                       # (4, 4) camera-to-world
    intrinsics: np.ndarray                 # (3, 3) camera intrinsic matrix
    H: int
    W: int
