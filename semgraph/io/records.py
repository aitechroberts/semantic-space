"""
Record dataclasses — one per stage boundary.

Each record has:
  - ``to_arrays()``  -> dict[str, np.ndarray]
  - ``to_metadata()`` -> dict[str, Any]   (JSON-safe)
  - ``from_arrays_and_metadata(arrays, meta)`` -> Self

Variable-length arrays use the **offset pattern**::

    all_points = np.concatenate([d.pcd_points for d in detections])
    offsets    = np.cumsum([0] + [len(d.pcd_points) for d in detections])

Records never import any serializer.  Serializers never import any record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────

def _concat_with_offsets(
    arrays: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate variable-length arrays and return (flat, offsets)."""
    if not arrays or all(len(a) == 0 for a in arrays):
        flat = np.empty((0, arrays[0].shape[1] if arrays and arrays[0].ndim > 1 else 0), dtype=np.float64)
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        return flat, offsets
    lengths = [len(a) for a in arrays]
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.concatenate(arrays, axis=0)
    return flat, offsets


def _split_by_offsets(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Inverse of ``_concat_with_offsets``."""
    offsets = offsets.astype(np.int64)
    return [flat[int(offsets[i]):int(offsets[i + 1])] for i in range(len(offsets) - 1)]


# =========================================================================
# RawDetRecord — detect.py → embed.py
# =========================================================================

@dataclass
class RawDetRecord:
    """Pre-filter detection metadata for a single frame."""

    xyxy: np.ndarray                       # (N, 4) float32
    confidence: np.ndarray                 # (N,)   float32
    class_id: np.ndarray                   # (N,)   int32
    masks: list[np.ndarray]                # N variable-size bool masks
    classes: list[str]
    detection_class_labels: list[str]
    labels: list[str]
    captions: list[str]

    # ── serialization ────────────────────────────────────────────────

    def to_arrays(self) -> dict[str, np.ndarray]:
        flat_masks, mask_offsets = _concat_with_offsets(
            [m.reshape(-1).astype(np.uint8) for m in self.masks]
        )
        mask_shapes = np.array(
            [m.shape for m in self.masks], dtype=np.int64,
        ) if self.masks else np.empty((0, 2), dtype=np.int64)
        return {
            "xyxy": np.asarray(self.xyxy, dtype=np.float32),
            "confidence": np.asarray(self.confidence, dtype=np.float32),
            "class_id": np.asarray(self.class_id, dtype=np.int32),
            "masks_flat": flat_masks,
            "mask_offsets": mask_offsets,
            "mask_shapes": mask_shapes,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "classes": self.classes,
            "detection_class_labels": self.detection_class_labels,
            "labels": self.labels,
            "captions": self.captions,
        }

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any],
    ) -> RawDetRecord:
        flat_masks = arrays["masks_flat"]
        mask_offsets = arrays["mask_offsets"]
        mask_shapes = arrays["mask_shapes"]
        flat_chunks = _split_by_offsets(flat_masks, mask_offsets)
        masks = []
        for i, chunk in enumerate(flat_chunks):
            if i < len(mask_shapes):
                shape = tuple(int(s) for s in mask_shapes[i])
                masks.append(chunk.reshape(shape).astype(bool))
            else:
                masks.append(chunk.astype(bool))
        return cls(
            xyxy=arrays["xyxy"],
            confidence=arrays["confidence"],
            class_id=arrays["class_id"],
            masks=masks,
            classes=meta.get("classes", []),
            detection_class_labels=meta.get("detection_class_labels", []),
            labels=meta.get("labels", []),
            captions=meta.get("captions", []),
        )


# =========================================================================
# FrameDataRecord — detect.py → build_map.py
# =========================================================================

@dataclass
class _DetectionMeta:
    """JSON-safe metadata for one detection inside a frame."""
    bbox_type: str = "axis_aligned"
    class_name: str = "object"
    class_id: int = 0
    inst_id: int = 0
    n_points: int = 0
    crop_path: str = ""


@dataclass
class FrameDataRecord:
    """Post-filter detection list + camera for a single frame.

    Variable-length per-detection arrays (pcd_points, pcd_colors) are stored
    via offsets.  Fixed-shape per-detection arrays (bbox_corners 8×3,
    clip_ft D, text_ft D) are stacked into (N, ...) blocks.
    """

    frame_idx: int = 0
    color_path: str = ""
    skip_matching: bool = False
    H: int = 0
    W: int = 0

    # per-frame
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    intrinsics: np.ndarray = field(default_factory=lambda: np.eye(3))
    surviving_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32),
    )

    # per-detection (variable-length)
    pcd_points_list: list[np.ndarray] = field(default_factory=list)
    pcd_colors_list: list[np.ndarray] = field(default_factory=list)

    # per-detection (fixed-shape stacked)
    bbox_corners: np.ndarray = field(
        default_factory=lambda: np.empty((0, 8, 3)),
    )  # (N, 8, 3)
    clip_ft: np.ndarray | None = None   # (N, D)  — None until embed.py
    text_ft: np.ndarray | None = None   # (N, D)  — None until embed.py

    # per-detection metadata
    det_meta: list[_DetectionMeta] = field(default_factory=list)

    # ── serialization ────────────────────────────────────────────────

    def to_arrays(self) -> dict[str, np.ndarray]:
        pts_flat, pts_off = _concat_with_offsets(
            self.pcd_points_list if self.pcd_points_list
            else [np.empty((0, 3))]
        )
        col_flat, col_off = _concat_with_offsets(
            self.pcd_colors_list if self.pcd_colors_list
            else [np.empty((0, 3))]
        )
        arrays: dict[str, np.ndarray] = {
            "pose": np.asarray(self.pose, dtype=np.float64),
            "intrinsics": np.asarray(self.intrinsics, dtype=np.float64),
            "surviving_indices": np.asarray(self.surviving_indices, dtype=np.int32),
            "pcd_points": pts_flat,
            "pcd_offsets": pts_off,
            "pcd_colors": col_flat,
            "pcd_color_offsets": col_off,
            "bbox_corners": self.bbox_corners.reshape(-1, 3),  # (N*8, 3)
        }
        if self.clip_ft is not None:
            arrays["clip_ft"] = np.asarray(self.clip_ft, dtype=np.float32)
        if self.text_ft is not None:
            arrays["text_ft"] = np.asarray(self.text_ft, dtype=np.float32)
        return arrays

    def to_metadata(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "color_path": self.color_path,
            "skip_matching": self.skip_matching,
            "H": self.H,
            "W": self.W,
            "detections": [
                {
                    "bbox_type": dm.bbox_type,
                    "class_name": dm.class_name,
                    "class_id": dm.class_id,
                    "inst_id": dm.inst_id,
                    "n_points": dm.n_points,
                    "crop_path": dm.crop_path,
                }
                for dm in self.det_meta
            ],
        }

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any],
    ) -> FrameDataRecord:
        pcd_points_list = _split_by_offsets(arrays["pcd_points"], arrays["pcd_offsets"])
        pcd_colors_list = _split_by_offsets(arrays["pcd_colors"], arrays["pcd_color_offsets"])

        bbox_flat = arrays["bbox_corners"]  # (N*8, 3)
        n_det = len(meta.get("detections", []))
        if n_det > 0 and len(bbox_flat) >= n_det * 8:
            bbox_corners = bbox_flat.reshape(n_det, 8, 3)
        else:
            bbox_corners = np.empty((0, 8, 3))

        det_dicts = meta.get("detections", [])
        det_meta = [
            _DetectionMeta(
                bbox_type=d.get("bbox_type", "axis_aligned"),
                class_name=d.get("class_name", "object"),
                class_id=int(d.get("class_id", 0)),
                inst_id=int(d.get("inst_id", 0)),
                n_points=int(d.get("n_points", 0)),
                crop_path=d.get("crop_path", ""),
            )
            for d in det_dicts
        ]

        return cls(
            frame_idx=meta.get("frame_idx", 0),
            color_path=meta.get("color_path", ""),
            skip_matching=meta.get("skip_matching", False),
            H=meta.get("H", 0),
            W=meta.get("W", 0),
            pose=arrays.get("pose", np.eye(4)),
            intrinsics=arrays.get("intrinsics", np.eye(3)),
            surviving_indices=arrays.get("surviving_indices", np.array([], dtype=np.int32)),
            pcd_points_list=pcd_points_list,
            pcd_colors_list=pcd_colors_list,
            bbox_corners=bbox_corners,
            clip_ft=arrays.get("clip_ft"),
            text_ft=arrays.get("text_ft"),
            det_meta=det_meta,
        )

    # ── convenience ──────────────────────────────────────────────────

    @property
    def n_detections(self) -> int:
        return len(self.det_meta)


# =========================================================================
# CaptionsRecord — caption.py output (pure metadata, no arrays)
# =========================================================================

@dataclass
class CaptionsRecord:
    """Per-object captions produced by the VLM captioning stage."""

    entries: dict[int, dict[str, Any]] = field(default_factory=dict)
    """obj_idx -> {canonical_tag, candidate_tags, summary, color, material, raw_captions}"""

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"_placeholder": np.array([0], dtype=np.int8)}

    def to_metadata(self) -> dict[str, Any]:
        return {"entries": {str(k): v for k, v in self.entries.items()}}

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any],
    ) -> CaptionsRecord:
        raw = meta.get("entries", {})
        entries = {int(k): v for k, v in raw.items()}
        return cls(entries=entries)


# =========================================================================
# OracleSceneRecord — oracle_finalize.py output
# =========================================================================

@dataclass
class _PerViewMeta:
    """Non-array fields of a per_view_record (JSON-safe)."""
    frame_idx: int = 0
    n_points: int = 0
    crop_path: str = ""
    crop_bbox: list[int] = field(default_factory=list)


@dataclass
class OracleSceneRecord:
    """Immutable oracle scene: geometry + planes + MST edges.

    Heavy arrays (per-object PCD, per-view CLIP features) use offsets.
    Metadata carries class names, plane records, MST edges, per-view
    non-array fields, and parent_plane_ids.
    """

    # per-object variable-length
    obj_pcd_points_list: list[np.ndarray] = field(default_factory=list)
    obj_pcd_colors_list: list[np.ndarray] = field(default_factory=list)
    obj_bbox_corners: np.ndarray = field(
        default_factory=lambda: np.empty((0, 8, 3)),
    )

    # per-view CLIP features (offset pattern across all objects)
    pv_clip_ft_list: list[np.ndarray] = field(default_factory=list)

    # metadata
    class_names: list[str] = field(default_factory=list)
    parent_plane_ids: list[int | None] = field(default_factory=list)
    planes: list[dict[str, Any]] = field(default_factory=list)
    mst_edges: list[tuple[int, int, float]] = field(default_factory=list)
    per_view_meta: list[list[_PerViewMeta]] = field(default_factory=list)

    # ── serialization ────────────────────────────────────────────────

    def to_arrays(self) -> dict[str, np.ndarray]:
        pts_flat, pts_off = _concat_with_offsets(
            self.obj_pcd_points_list or [np.empty((0, 3))]
        )
        col_flat, col_off = _concat_with_offsets(
            self.obj_pcd_colors_list or [np.empty((0, 3))]
        )
        ft_flat, ft_off = _concat_with_offsets(
            self.pv_clip_ft_list or [np.empty((0, 0))]
        )
        arrays: dict[str, np.ndarray] = {
            "obj_pcd_points": pts_flat,
            "obj_pcd_offsets": pts_off,
            "obj_pcd_colors": col_flat,
            "obj_pcd_color_offsets": col_off,
            "obj_bbox_corners": self.obj_bbox_corners.reshape(-1, 3),
            "pv_clip_ft": ft_flat,
            "pv_clip_ft_offsets": ft_off,
        }
        return arrays

    def to_metadata(self) -> dict[str, Any]:
        pv_meta_serializable: list[list[dict]] = []
        for obj_pvr in self.per_view_meta:
            pv_meta_serializable.append([
                {
                    "frame_idx": p.frame_idx,
                    "n_points": p.n_points,
                    "crop_path": p.crop_path,
                    "crop_bbox": p.crop_bbox,
                }
                for p in obj_pvr
            ])
        return {
            "class_names": self.class_names,
            "parent_plane_ids": self.parent_plane_ids,
            "planes": self.planes,
            "mst_edges": [list(e) for e in self.mst_edges],
            "per_view_meta": pv_meta_serializable,
        }

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any],
    ) -> OracleSceneRecord:
        pts_list = _split_by_offsets(arrays["obj_pcd_points"], arrays["obj_pcd_offsets"])
        col_list = _split_by_offsets(arrays["obj_pcd_colors"], arrays["obj_pcd_color_offsets"])
        ft_list = _split_by_offsets(arrays["pv_clip_ft"], arrays["pv_clip_ft_offsets"])

        n_obj = len(meta.get("class_names", []))
        bbox_flat = arrays["obj_bbox_corners"]
        if n_obj > 0 and len(bbox_flat) >= n_obj * 8:
            bbox_corners = bbox_flat.reshape(n_obj, 8, 3)
        else:
            bbox_corners = np.empty((0, 8, 3))

        raw_pv = meta.get("per_view_meta", [])
        per_view_meta = []
        for obj_pvr in raw_pv:
            per_view_meta.append([
                _PerViewMeta(
                    frame_idx=int(r.get("frame_idx", 0)),
                    n_points=int(r.get("n_points", 0)),
                    crop_path=r.get("crop_path", ""),
                    crop_bbox=r.get("crop_bbox", []),
                )
                for r in obj_pvr
            ])

        raw_edges = meta.get("mst_edges", [])
        mst_edges = [(int(e[0]), int(e[1]), float(e[2])) for e in raw_edges]

        return cls(
            obj_pcd_points_list=pts_list,
            obj_pcd_colors_list=col_list,
            obj_bbox_corners=bbox_corners,
            pv_clip_ft_list=ft_list,
            class_names=meta.get("class_names", []),
            parent_plane_ids=meta.get("parent_plane_ids", []),
            planes=meta.get("planes", []),
            mst_edges=mst_edges,
            per_view_meta=per_view_meta,
        )


# =========================================================================
# VariantRecord — embed.py (re-embed) output
# =========================================================================

@dataclass
class VariantRecord:
    """Phase B encoder variant: per-object weighted-avg, best, per-view features."""

    encoder_name: str = ""
    # per-object (N, D)
    clip_ft_weighted_avg: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32),
    )
    clip_ft_best: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32),
    )
    best_entropy: list[float] = field(default_factory=list)
    # per-view features (offset pattern)
    pv_feats_list: list[np.ndarray] = field(default_factory=list)

    # ── serialization ────────────────────────────────────────────────

    def to_arrays(self) -> dict[str, np.ndarray]:
        ft_flat, ft_off = _concat_with_offsets(
            self.pv_feats_list or [np.empty((0, 0), dtype=np.float32)]
        )
        return {
            "clip_ft_weighted_avg": np.asarray(self.clip_ft_weighted_avg, dtype=np.float32),
            "clip_ft_best": np.asarray(self.clip_ft_best, dtype=np.float32),
            "best_entropy": np.asarray(self.best_entropy, dtype=np.float32),
            "pv_feats": ft_flat.astype(np.float32),
            "pv_feats_offsets": ft_off,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {"encoder_name": self.encoder_name}

    @classmethod
    def from_arrays_and_metadata(
        cls, arrays: dict[str, np.ndarray], meta: dict[str, Any],
    ) -> VariantRecord:
        pv_list = _split_by_offsets(arrays["pv_feats"], arrays["pv_feats_offsets"])
        return cls(
            encoder_name=meta.get("encoder_name", ""),
            clip_ft_weighted_avg=arrays.get(
                "clip_ft_weighted_avg", np.empty((0, 0), dtype=np.float32),
            ),
            clip_ft_best=arrays.get(
                "clip_ft_best", np.empty((0, 0), dtype=np.float32),
            ),
            best_entropy=arrays.get(
                "best_entropy", np.empty(0, dtype=np.float32),
            ).tolist(),
            pv_feats_list=pv_list,
        )
