"""
TrajectoryBackend — the default RGBD geometry path.

Wraps the existing dataset loader (GradSLAMDataset) and depth-unprojection
logic (detections_to_obj_pcd_and_bbox) behind the GeometryBackend interface.
Behaviour is identical to the pre-refactor pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from conceptgraph.slam.geometry.base import FrameContext, GeometryBackend
from conceptgraph.slam.utils import detections_to_obj_pcd_and_bbox


@dataclass
class TrajectoryContext:
    """Opaque context returned by TrajectoryBackend.load()."""
    dataset: Any  # GradSLAMDataset instance


class TrajectoryBackend(GeometryBackend):
    """RGBD depth + poses -> depth unprojection (the original pipeline path)."""

    def load(self, cfg: Any) -> TrajectoryContext:
        _frames, dataset = self.load_camera_frames(cfg)
        return TrajectoryContext(dataset=dataset)

    def get_iterator(self, ctx: TrajectoryContext) -> Iterator[FrameContext]:
        dataset = ctx.dataset
        for frame_idx in range(len(dataset)):
            color_path = Path(dataset.color_paths[frame_idx])
            color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

            depth_tensor = depth_tensor[..., 0]
            depth_array = depth_tensor.cpu().numpy()
            image_rgb = color_tensor.cpu().numpy().astype(np.uint8)

            pose = dataset.poses[frame_idx].cpu().numpy()

            yield FrameContext(
                frame_idx=frame_idx,
                color_path=color_path,
                image_rgb=image_rgb,
                intrinsics=intrinsics.cpu().numpy()[:3, :3],
                pose=pose,
                extra={
                    "depth_array": depth_array,
                    "intrinsics_4x4": intrinsics.cpu().numpy(),
                },
            )

    def lift_to_3d(
        self,
        masks: np.ndarray,
        frame_ctx: FrameContext,
        cfg: Any,
    ) -> list[dict | None]:
        return detections_to_obj_pcd_and_bbox(
            depth_array=frame_ctx.extra["depth_array"],
            masks=masks,
            cam_K=frame_ctx.intrinsics,
            image_rgb=frame_ctx.image_rgb,
            trans_pose=frame_ctx.pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

    def num_iterations(self, ctx: TrajectoryContext) -> int:
        return len(ctx.dataset)
