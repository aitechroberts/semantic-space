"""
High-level save/load functions — thin glue combining records + serializer.

One save/load pair per record type.  Also keeps ``serialize_detection`` /
``deserialize_detection`` (live o3d ↔ numpy conversion at the build_map
boundary) and ``write_scene_graph_json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from semgraph.io.records import (
    CaptionsRecord,
    FrameDataRecord,
    OracleSceneRecord,
    RawDetRecord,
    VariantRecord,
)
from semgraph.io.serializers import NpzSerializer
from semgraph.stages.paths import RawGobs, SerializedDetection

logger = logging.getLogger(__name__)

_serializer = NpzSerializer()


# =====================================================================
# raw_det
# =====================================================================

def save_raw_det(directory: Path, frame_idx: int, raw_gobs: RawGobs) -> None:
    """Save pre-filter detection metadata for one frame."""
    masks = raw_gobs.get("mask")
    if masks is None:
        mask_list: list[np.ndarray] = []
    elif isinstance(masks, np.ndarray) and masks.ndim == 3:
        mask_list = [masks[i] for i in range(masks.shape[0])]
    else:
        mask_list = list(masks)

    record = RawDetRecord(
        xyxy=np.asarray(raw_gobs["xyxy"]),
        confidence=np.asarray(raw_gobs["confidence"]),
        class_id=np.asarray(raw_gobs["class_id"]),
        masks=mask_list,
        classes=raw_gobs.get("classes", []),
        detection_class_labels=raw_gobs.get("detection_class_labels", []),
        labels=raw_gobs.get("labels", []),
        captions=raw_gobs.get("captions", []),
    )
    path = Path(directory) / f"{frame_idx:06d}"
    _serializer.save(record.to_arrays(), record.to_metadata(), path)


def load_raw_det(directory: Path, frame_idx: int) -> RawGobs | None:
    """Load raw detection metadata.  Returns ``None`` if missing."""
    path = Path(directory) / f"{frame_idx:06d}"
    if not path.with_suffix(".npz").is_file():
        logger.warning("Missing raw_det for frame %d at %s", frame_idx, directory)
        return None
    arrays, meta = _serializer.load(path)
    record = RawDetRecord.from_arrays_and_metadata(arrays, meta)
    masks_3d = np.stack(record.masks) if record.masks else np.empty((0, 0, 0), dtype=bool)
    return RawGobs(
        xyxy=record.xyxy,
        confidence=record.confidence,
        class_id=record.class_id,
        mask=masks_3d,
        classes=record.classes,
        detection_class_labels=record.detection_class_labels,
        labels=record.labels,
        captions=record.captions,
        edges=[],
        image_crops=None,
        image_feats=None,
        text_feats=None,
        vlm_vit_feats=None,
        vlm_proj_feats=None,
    )


# =====================================================================
# frame_data
# =====================================================================

def save_frame_data(directory: Path, frame_idx: int, record: FrameDataRecord) -> None:
    """Save post-filter detection list + camera for one frame."""
    path = Path(directory) / f"{frame_idx:06d}"
    _serializer.save(record.to_arrays(), record.to_metadata(), path)


def load_frame_data(directory: Path, frame_idx: int) -> FrameDataRecord | None:
    """Load frame data.  Returns ``None`` if missing."""
    path = Path(directory) / f"{frame_idx:06d}"
    if not path.with_suffix(".npz").is_file():
        logger.warning("Missing frame_data for frame %d at %s", frame_idx, directory)
        return None
    arrays, meta = _serializer.load(path)
    return FrameDataRecord.from_arrays_and_metadata(arrays, meta)


# =====================================================================
# captions
# =====================================================================

def save_captions(directory: Path, record: CaptionsRecord) -> None:
    """Save caption data (pure JSON — the npz is a placeholder)."""
    path = Path(directory) / "captions"
    _serializer.save(record.to_arrays(), record.to_metadata(), path)


def load_captions(directory: Path) -> CaptionsRecord | None:
    """Load captions.  Returns ``None`` if missing."""
    path = Path(directory) / "captions"
    if not path.with_suffix(".npz").is_file():
        logger.debug("No captions found at %s", directory)
        return None
    arrays, meta = _serializer.load(path)
    return CaptionsRecord.from_arrays_and_metadata(arrays, meta)


# =====================================================================
# map (build_map.py output — intermediate before oracle_finalize)
# =====================================================================

def _serialize_obj(obj: dict) -> dict:
    """Convert a live detection/object dict to a pickle-safe form.

    Open3D CUDA geometry objects cannot be pickled, so we convert
    ``pcd`` and ``bbox`` to numpy arrays and reconstruct on load.
    """
    import open3d as o3d

    out = {}
    for k, v in obj.items():
        if isinstance(v, (o3d.geometry.PointCloud,)):
            pts = np.asarray(v.points)
            cols = np.asarray(v.colors) if v.has_colors() else np.zeros_like(pts)
            out[k] = {"__o3d_pcd__": True, "points": pts, "colors": cols}
        elif isinstance(v, (o3d.geometry.AxisAlignedBoundingBox,)):
            out[k] = {
                "__o3d_bbox__": "axis_aligned",
                "min_bound": np.asarray(v.min_bound),
                "max_bound": np.asarray(v.max_bound),
            }
        elif isinstance(v, (o3d.geometry.OrientedBoundingBox,)):
            out[k] = {
                "__o3d_bbox__": "oriented",
                "corners": np.asarray(v.get_box_points()),
            }
        else:
            out[k] = v
    return out


def _deserialize_obj(obj: dict) -> dict:
    """Reconstruct Open3D objects from the numpy-serialized form."""
    import open3d as o3d

    out = {}
    for k, v in obj.items():
        if isinstance(v, dict) and v.get("__o3d_pcd__"):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(v["points"])
            if v["colors"] is not None and len(v["colors"]) > 0:
                pcd.colors = o3d.utility.Vector3dVector(v["colors"])
            out[k] = pcd
        elif isinstance(v, dict) and v.get("__o3d_bbox__") == "axis_aligned":
            out[k] = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=v["min_bound"], max_bound=v["max_bound"],
            )
        elif isinstance(v, dict) and v.get("__o3d_bbox__") == "oriented":
            out[k] = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(v["corners"])
            )
        else:
            out[k] = v
    return out


def save_map(directory: Path, objects: Any, edges: Any, cfg: Any) -> None:
    """Save the accumulated map (MapObjectList + MapEdgeMapping).

    Open3D geometry objects are converted to numpy arrays before pickling
    to avoid issues with unpicklable CUDA-backed Open3D types.
    """
    import pickle, gzip, tempfile, os  # noqa: E401
    path = Path(directory) / "oracle_map"
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized_objects = [_serialize_obj(obj) for obj in objects]

    # MapEdgeMapping.objects holds a ref to the live MapObjectList which
    # contains unpicklable CUDA Open3D objects.  Temporarily detach it.
    saved_ref = getattr(edges, "objects", None)
    if hasattr(edges, "objects"):
        edges.objects = None

    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with gzip.open(tmp, "wb") as f:
            pickle.dump({
                "objects": serialized_objects,
                "edges": edges,
                "cfg": cfg,
            }, f)
        target = path.with_suffix(".pkl.gz")
        os.rename(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    finally:
        if saved_ref is not None and hasattr(edges, "objects"):
            edges.objects = saved_ref


def load_map(directory: Path) -> tuple[Any, Any, Any] | None:
    """Load the intermediate map.  Returns ``(objects, edges, cfg)``."""
    import pickle, gzip  # noqa: E401
    path = Path(directory) / "oracle_map.pkl.gz"
    if not path.is_file():
        return None
    with gzip.open(path, "rb") as f:
        result = pickle.load(f)  # noqa: S301

    objects = [_deserialize_obj(obj) for obj in result["objects"]]
    edges = result["edges"]
    if hasattr(edges, "objects"):
        edges.objects = objects
    return objects, edges, result["cfg"]


# =====================================================================
# oracle_scene
# =====================================================================

def save_oracle_scene(directory: Path, record: OracleSceneRecord) -> None:
    """Save the immutable oracle scene."""
    path = Path(directory) / "oracle_scene"
    _serializer.save(record.to_arrays(), record.to_metadata(), path)


def load_oracle_scene(directory: Path) -> OracleSceneRecord | None:
    """Load oracle scene.  Returns ``None`` if missing."""
    path = Path(directory) / "oracle_scene"
    if not path.with_suffix(".npz").is_file():
        logger.warning("oracle_scene.npz not found at %s", directory)
        return None
    arrays, meta = _serializer.load(path)
    return OracleSceneRecord.from_arrays_and_metadata(arrays, meta)


# =====================================================================
# variant (Phase B outputs)
# =====================================================================

def save_variant(directory: Path, record: VariantRecord, slug: str) -> None:
    """Save a Phase B variant keyed by a slug (e.g. encoder name)."""
    path = Path(directory) / slug
    path.parent.mkdir(parents=True, exist_ok=True)
    _serializer.save(record.to_arrays(), record.to_metadata(), path)


def load_variant(directory: Path, slug: str) -> VariantRecord | None:
    """Load a Phase B variant."""
    path = Path(directory) / slug
    if not path.with_suffix(".npz").is_file():
        return None
    arrays, meta = _serializer.load(path)
    return VariantRecord.from_arrays_and_metadata(arrays, meta)


# =====================================================================
# list_frame_indices
# =====================================================================

def list_frame_indices(directory: Path) -> list[int]:
    """Discover which frames have been processed (sorted ascending)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    indices: list[int] = []
    for f in directory.iterdir():
        if f.suffix == ".npz":
            try:
                indices.append(int(f.stem))
            except ValueError:
                continue
    indices.sort()
    return indices


# =====================================================================
# Detection serialize / deserialize (live o3d ↔ numpy dict)
# =====================================================================

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
        gt_instance_id=det.get("gt_instance_id"),
        n_visible=det.get("n_visible"),
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
        return torch.from_numpy(arr)

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
        "gt_instance_id": data.get("gt_instance_id"),
        "n_visible": data.get("n_visible"),
    }


# =====================================================================
# Scene graph JSON (final HPSG output for eval.py)
# =====================================================================

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene_graph, f, indent=2)
