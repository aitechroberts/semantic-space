"""
GeometryBackend ABC and FrameContext dataclass.

Every geometry backend implements three operations:
  1. load()          -- read dataset/mesh/images at startup
  2. get_iterator()  -- yield per-iteration FrameContext objects
  3. lift_to_3d()    -- convert 2D masks to world-frame point clouds

The lift_to_3d() contract mirrors detections_to_obj_pcd_and_bbox() in
conceptgraph/slam/utils.py: it returns list[dict | None] where each dict
contains 'pcd' (o3d.geometry.PointCloud) and 'bbox' (o3d bounding box).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


@dataclass
class FrameContext:
    """Per-iteration context yielded by a GeometryBackend."""

    frame_idx: int
    color_path: Path
    image_rgb: np.ndarray  # (H, W, 3) uint8

    intrinsics: Optional[np.ndarray] = None  # (3, 3) camera matrix
    pose: Optional[np.ndarray] = None  # (4, 4) camera-to-world

    skip_segmentation: bool = False
    skip_matching: bool = False

    instance_id: Optional[int] = None

    extra: dict = field(default_factory=dict)


class GeometryBackend(ABC):
    """Strategy interface for pipeline geometry modes."""

    @abstractmethod
    def load(self, cfg: Any) -> Any:
        """Load all required data.  Returns an opaque context object."""
        ...

    @abstractmethod
    def get_iterator(self, ctx: Any) -> Iterator[FrameContext]:
        """Yield one FrameContext per main-loop iteration."""
        ...

    @abstractmethod
    def lift_to_3d(
        self,
        masks: np.ndarray,
        frame_ctx: FrameContext,
        cfg: Any,
    ) -> list[dict | None]:
        """Lift 2D masks to 3D point clouds in world frame.

        Returns a list parallel to *masks* where each element is either
        ``{"pcd": <PointCloud>, "bbox": <BoundingBox>}`` or ``None``.
        """
        ...

    @abstractmethod
    def num_iterations(self, ctx: Any) -> int | None:
        """Total iteration count for progress bars, or None if unknown."""
        ...

    @staticmethod
    def load_camera_frames(cfg: Any) -> tuple[list[dict], Any]:
        """Load camera trajectory via GradSLAMDataset.

        Returns
        -------
        frames : list[dict]
            Each dict has ``color_path``, ``pose`` (4x4), ``intrinsics`` (3x3),
            ``H``, ``W``.
        dataset : GradSLAMDataset
            The raw dataset object, needed by TrajectoryBackend for per-frame
            color/depth tensor access.
        """
        import torch
        from conceptgraph.dataset.datasets_common import get_dataset

        dataset = get_dataset(
            dataconfig=cfg.dataset_config,
            start=cfg.start,
            end=cfg.end,
            stride=cfg.stride,
            basedir=cfg.dataset_root,
            sequence=cfg.scene_id,
            desired_height=cfg.image_height,
            desired_width=cfg.image_width,
            device="cpu",
            dtype=torch.float,
        )

        frames: list[dict] = []
        for i in range(len(dataset)):
            _, _, intrinsics, *_ = dataset[i]
            frames.append({
                "color_path": dataset.color_paths[i],
                "pose": dataset.poses[i].cpu().numpy(),
                "intrinsics": intrinsics.cpu().numpy()[:3, :3],
                "H": cfg.image_height or dataset.orig_height,
                "W": cfg.image_width or dataset.orig_width,
            })

        return frames, dataset
