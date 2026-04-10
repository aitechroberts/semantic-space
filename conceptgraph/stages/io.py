"""
Stage I/O — serialization, deserialization, and atomic file helpers.

This module is the most critical part of the staged pipeline refactor.
It provides the ``serialize_detection`` / ``deserialize_detection`` pair
that converts between live detection dicts (with o3d objects and torch
tensors) and numpy-only dicts safe for pickle.

Serialization format: **npz + JSON** via record dataclasses with
``to_arrays()`` / ``from_arrays_and_metadata()`` for future-proofing.
Legacy ``pkl.gz`` files are still readable for backward compat.

All ``save_*`` functions write to a ``.tmp`` file first, then atomically
rename.  All ``load_*`` functions return ``None`` on missing files (the
consumer logs a warning and skips that frame).
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from conceptgraph.stages.paths import (
    FrameDataRecord,
    RawGobs,
    SerializedDetection,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# npz + JSON record dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """Dataclass wrapper around SerializedDetection for npz+JSON I/O."""
    pcd_points: np.ndarray
    pcd_colors: np.ndarray
    bbox_corners: np.ndarray
    bbox_type: str = "axis_aligned"
    class_name: str = "object"
    class_id: int = 0
    inst_id: int = 0
    n_points: int = 0
    crop_path: str = ""
    clip_ft: np.ndarray | None = None
    text_ft: np.ndarray | None = None

    def to_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Split into numpy arrays dict and JSON-safe metadata dict."""
        arrays = {
            "pcd_points": self.pcd_points,
            "pcd_colors": self.pcd_colors,
            "bbox_corners": self.bbox_corners,
        }
        if self.clip_ft is not None:
            arrays["clip_ft"] = self.clip_ft
        if self.text_ft is not None:
            arrays["text_ft"] = self.text_ft
        meta = {
            "bbox_type": self.bbox_type,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "inst_id": self.inst_id,
            "n_points": self.n_points,
            "crop_path": self.crop_path,
        }
        return arrays, meta

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> "DetectionRecord":
        return cls(
            pcd_points=arrays["pcd_points"],
            pcd_colors=arrays["pcd_colors"],
            bbox_corners=arrays["bbox_corners"],
            bbox_type=meta.get("bbox_type", "axis_aligned"),
            class_name=meta.get("class_name", "object"),
            class_id=meta.get("class_id", 0),
            inst_id=meta.get("inst_id", 0),
            n_points=meta.get("n_points", 0),
            crop_path=meta.get("crop_path", ""),
            clip_ft=arrays.get("clip_ft"),
            text_ft=arrays.get("text_ft"),
        )

    def to_serialized_detection(self) -> SerializedDetection:
        """Convert to TypedDict for backward compat with existing code."""
        return SerializedDetection(
            pcd_points=self.pcd_points,
            pcd_colors=self.pcd_colors,
            bbox_corners=self.bbox_corners,
            bbox_type=self.bbox_type,
            class_name=self.class_name,
            class_id=self.class_id,
            inst_id=self.inst_id,
            n_points=self.n_points,
            crop_path=self.crop_path,
            clip_ft=self.clip_ft,
            text_ft=self.text_ft,
            vlm_vit_ft=None,
            vlm_proj_ft=None,
        )

    @classmethod
    def from_serialized_detection(cls, sd: SerializedDetection) -> "DetectionRecord":
        return cls(
            pcd_points=sd["pcd_points"],
            pcd_colors=sd["pcd_colors"],
            bbox_corners=sd["bbox_corners"],
            bbox_type=sd.get("bbox_type", "axis_aligned"),
            class_name=sd.get("class_name", "object"),
            class_id=sd.get("class_id", 0),
            inst_id=sd.get("inst_id", 0),
            n_points=sd.get("n_points", 0),
            crop_path=sd.get("crop_path", ""),
            clip_ft=sd.get("clip_ft"),
            text_ft=sd.get("text_ft"),
        )


@dataclass
class FrameRecord:
    """Dataclass wrapper around FrameDataRecord for npz+JSON I/O."""
    frame_idx: int = 0
    color_path: str = ""
    skip_matching: bool = False
    surviving_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    detections: list[DetectionRecord] = field(default_factory=list)
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))
    H: int = 0
    W: int = 0

    def to_arrays(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arrays = {
            "surviving_indices": self.surviving_indices,
            "pose": self.pose,
            "intrinsics": self.intrinsics,
        }
        det_arrays_list = []
        det_meta_list = []
        for det in self.detections:
            a, m = det.to_arrays()
            det_arrays_list.append(a)
            det_meta_list.append(m)

        meta = {
            "frame_idx": self.frame_idx,
            "color_path": self.color_path,
            "skip_matching": self.skip_matching,
            "H": self.H,
            "W": self.W,
            "detections_meta": det_meta_list,
        }
        return arrays, meta, det_arrays_list

    @classmethod
    def from_arrays_and_metadata(
        cls,
        arrays: dict[str, np.ndarray],
        meta: dict[str, Any],
        det_arrays_list: list[dict[str, np.ndarray]] | None = None,
    ) -> "FrameRecord":
        det_meta_list = meta.get("detections_meta", [])
        detections = []
        if det_arrays_list:
            for da, dm in zip(det_arrays_list, det_meta_list):
                detections.append(DetectionRecord.from_arrays_and_metadata(da, dm))
        return cls(
            frame_idx=meta.get("frame_idx", 0),
            color_path=meta.get("color_path", ""),
            skip_matching=meta.get("skip_matching", False),
            surviving_indices=arrays.get("surviving_indices", np.array([], dtype=np.int32)),
            detections=detections,
            pose=arrays.get("pose", np.eye(4)),
            intrinsics=arrays.get("intrinsics", np.eye(3)),
            H=meta.get("H", 0),
            W=meta.get("W", 0),
        )


# ---------------------------------------------------------------------------
# Detection serialize / deserialize
# ---------------------------------------------------------------------------

def serialize_detection(det: dict, spatial_sim_type: str) -> SerializedDetection:
    """Convert a live detection dict to a numpy-only ``SerializedDetection``.

    The live dict comes from ``make_detection_list_from_pcd_and_gobs`` and
    contains ``o3d.geometry.PointCloud``, ``o3d`` bounding-box objects, and
    ``torch.Tensor`` feature vectors.
    """
    import open3d as o3d

    pcd = det["pcd"]
    bbox = det["bbox"]

    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors) if pcd.has_colors() else np.zeros_like(pcd_points)

    bbox_corners = np.asarray(bbox.get_box_points())

    if isinstance(bbox, o3d.geometry.AxisAlignedBoundingBox):
        bbox_type = "axis_aligned"
    else:
        bbox_type = "oriented"

    def _to_numpy(t: Any) -> np.ndarray | None:
        if t is None:
            return None
        if hasattr(t, "cpu"):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    return SerializedDetection(
        pcd_points=pcd_points,
        pcd_colors=pcd_colors,
        bbox_corners=bbox_corners,
        bbox_type=bbox_type,
        class_name=det.get("class_name", "object"),
        class_id=int(det.get("class_id", [0])[0]) if isinstance(det.get("class_id"), list) else int(det.get("class_id", 0)),
        inst_id=int(det.get("curr_obj_num", 0)),
        n_points=len(pcd_points),
        crop_path=det.get("crop_path", ""),
        clip_ft=_to_numpy(det.get("clip_ft")),
        text_ft=_to_numpy(det.get("text_ft")),
        vlm_vit_ft=_to_numpy(det.get("vlm_vit_ft")),
        vlm_proj_ft=_to_numpy(det.get("vlm_proj_ft")),
    )


def deserialize_detection(data: SerializedDetection, device: str = "cpu") -> dict:
    """Reconstruct a live detection dict from a ``SerializedDetection``.

    The result is compatible with ``merge_obj_matches``,
    ``compute_visual_similarities``, and ``MapObjectList.append``.
    """
    import open3d as o3d
    import torch

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data["pcd_points"])
    if data["pcd_colors"] is not None and len(data["pcd_colors"]) > 0:
        pcd.colors = o3d.utility.Vector3dVector(data["pcd_colors"])

    corners = data["bbox_corners"]
    if data["bbox_type"] == "axis_aligned":
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=corners.min(axis=0),
            max_bound=corners.max(axis=0),
        )
    else:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners)
        )

    def _to_tensor(arr: np.ndarray | None) -> Any:
        if arr is None:
            return None
        return torch.from_numpy(arr).to(device)

    return {
        "pcd": pcd,
        "bbox": bbox,
        "clip_ft": _to_tensor(data.get("clip_ft")),
        "text_ft": _to_tensor(data.get("text_ft")),
        "vlm_vit_ft": _to_tensor(data.get("vlm_vit_ft")),
        "vlm_proj_ft": _to_tensor(data.get("vlm_proj_ft")),
        "class_name": data["class_name"],
        "class_id": [data["class_id"]],
        "n_points": data["n_points"],
        "inst_id": data["inst_id"],
        "crop_path": data.get("crop_path", ""),
    }


# ---------------------------------------------------------------------------
# Atomic file I/O primitives
# ---------------------------------------------------------------------------

def _atomic_save(path: Path, obj: Any) -> None:
    """Pickle-gz *obj* to *path* via a temp file + atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with gzip.open(fd, "wb") as f:
            pickle.dump(obj, f)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _load_pkl_gz(path: Path) -> Any | None:
    """Load a pickle-gz file, returning ``None`` if it does not exist."""
    if not path.is_file():
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _frame_path(directory: Path, frame_idx: int) -> Path:
    return directory / f"{frame_idx:06d}.pkl.gz"


# ---------------------------------------------------------------------------
# Stage-specific save / load
# ---------------------------------------------------------------------------

def save_raw_det(path: Path, frame_idx: int, raw_gobs: RawGobs) -> None:
    """Atomic write of pre-filter detection metadata for caption.py."""
    _atomic_save(_frame_path(path, frame_idx), dict(raw_gobs))


def load_raw_det(path: Path, frame_idx: int) -> RawGobs | None:
    """Load raw detection metadata.  Returns ``None`` if file missing."""
    result = _load_pkl_gz(_frame_path(path, frame_idx))
    if result is None:
        logger.warning("Missing raw_det for frame %d at %s", frame_idx, path)
    return result


def save_frame_data(path: Path, frame_idx: int, record: FrameDataRecord) -> None:
    """Atomic write of processed (post-filter) detection list for build_map.py."""
    _atomic_save(_frame_path(path, frame_idx), dict(record))


def load_frame_data(path: Path, frame_idx: int) -> FrameDataRecord | None:
    """Load processed detection list.  Returns ``None`` if file missing."""
    result = _load_pkl_gz(_frame_path(path, frame_idx))
    if result is None:
        logger.warning("Missing frame_data for frame %d at %s", frame_idx, path)
    return result


def save_captions(
    path: Path,
    frame_idx: int,
    captions: list[str],
    edges: list[tuple],
    labels: list[str],
) -> None:
    """Atomic write of VLM-produced captions for a single frame."""
    _atomic_save(
        _frame_path(path, frame_idx),
        {"captions": captions, "edges": edges, "labels": labels},
    )


def load_captions(path: Path, frame_idx: int) -> dict | None:
    """Load captions.  Returns ``None`` if file missing (empty captions used)."""
    result = _load_pkl_gz(_frame_path(path, frame_idx))
    if result is None:
        logger.debug("Missing captions for frame %d (empty captions used)", frame_idx)
    return result


def save_map(
    path: Path,
    objects: Any,
    edges: Any,
    cfg: Any,
) -> None:
    """Save the accumulated map (MapObjectList + MapEdgeMapping)."""
    _atomic_save(
        path / "map.pkl.gz",
        {"objects": objects, "edges": edges, "cfg": cfg},
    )


def load_map(path: Path) -> tuple[Any, Any, Any] | None:
    """Load the map.  Returns ``(objects, edges, cfg)`` or ``None``."""
    result = _load_pkl_gz(path / "map.pkl.gz")
    if result is None:
        return None
    return result["objects"], result["edges"], result["cfg"]


def list_frame_indices(path: Path) -> list[int]:
    """Discover which frames have been processed (sorted ascending)."""
    if not path.is_dir():
        return []
    indices: list[int] = []
    for f in path.iterdir():
        if f.suffix == ".gz" and f.stem.endswith(".pkl"):
            name = f.stem.replace(".pkl", "")
            try:
                indices.append(int(name))
            except ValueError:
                continue
    indices.sort()
    return indices


# ---------------------------------------------------------------------------
# Oracle scene save / load
# ---------------------------------------------------------------------------

def save_oracle_scene(
    path: Path,
    objects: Any,
    edges: Any,
    planes: list,
    mst_edges: list,
    cfg: Any,
) -> None:
    """Save the immutable oracle scene (Phase A output)."""
    _atomic_save(
        path / "oracle_scene.pkl.gz",
        {
            "objects": objects,
            "edges": edges,
            "planes": planes,
            "mst_edges": mst_edges,
            "cfg": cfg,
        },
    )


def load_oracle_scene(path: Path) -> dict | None:
    """Load the oracle scene.  Returns the full dict or ``None``."""
    result = _load_pkl_gz(path / "oracle_scene.pkl.gz")
    if result is None:
        logger.warning("oracle_scene.pkl.gz not found at %s", path)
    return result


# ---------------------------------------------------------------------------
# Variant save / load (Phase B outputs)
# ---------------------------------------------------------------------------

def _variant_filename(encoder: str, vlm: str) -> str:
    """Build a filesystem-safe variant filename."""
    safe_enc = encoder.replace("/", "_")
    safe_vlm = vlm.replace("/", "_")
    return f"variant_{safe_enc}_{safe_vlm}.pkl.gz"


def save_variant(
    path: Path,
    encoder: str,
    vlm: str,
    data: dict,
) -> None:
    """Save a Phase B variant keyed by encoder and VLM name."""
    path.mkdir(parents=True, exist_ok=True)
    _atomic_save(path / _variant_filename(encoder, vlm), data)


def load_variant(path: Path, encoder: str, vlm: str) -> dict | None:
    """Load a Phase B variant.  Returns dict or ``None``."""
    return _load_pkl_gz(path / _variant_filename(encoder, vlm))


# ---------------------------------------------------------------------------
# Scene graph JSON (final HPSG output for eval.py)
# ---------------------------------------------------------------------------

def write_scene_graph_json(
    objects: Any,
    planes: list,
    labeled_edges: list,
    scene_type: str,
    output_path: Path,
) -> None:
    """Write the final HPSG JSON with the explicit node schema.

    ``objects`` is a MapObjectList or list of dicts. Each object must have
    at minimum a ``bbox`` (Open3D bounding box) for geometry and semantic
    fields populated by Phase B stages.
    """
    import json

    nodes = []
    for idx, obj in enumerate(objects):
        bbox = obj.get("bbox")
        if bbox is not None:
            bbox_min = np.asarray(bbox.min_bound)
            bbox_max = np.asarray(bbox.max_bound)
            extent = (bbox_max - bbox_min).tolist()
            center = ((bbox_min + bbox_max) / 2.0).tolist()
        else:
            extent = [0.0, 0.0, 0.0]
            center = [0.0, 0.0, 0.0]

        nodes.append({
            "id": obj.get("id", idx),
            "bbox_extent": extent,
            "bbox_center": center,
            "object_tag": obj.get("canonical_tag", obj.get("class_name", "unknown")),
            "caption": obj.get("summary", obj.get("consolidated_caption", "")),
            "color": obj.get("color", ""),
            "material": obj.get("material", ""),
            "candidate_tags": obj.get("candidate_tags", []),
            "best_entropy": float(obj.get("best_entropy", 0.0)),
            "n_views": int(obj.get("n_views", len(obj.get("per_view_records", [])))),
            "parent_plane_id": obj.get("parent_plane_id"),
        })

    plane_records = []
    for p in planes:
        plane_records.append({
            "plane_id": p.get("plane_id"),
            "label": p.get("label", ""),
            "caption": p.get("caption", ""),
            "normal": [float(x) for x in p.get("normal", [0, 0, 0])],
            "offset": float(p.get("offset", 0.0)),
        })

    scene_graph = {
        "scene_type": scene_type,
        "objects": nodes,
        "planes": plane_records,
        "edges": labeled_edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_graph, f, indent=2)
