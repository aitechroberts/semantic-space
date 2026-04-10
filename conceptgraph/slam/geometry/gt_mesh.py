"""
GTMeshBackend — ground-truth mesh geometry path.

Two sub-modes controlled by ``segmentation_backend``:

* **gt_instances** (object-first): iterates over GT instances, selects
  best views, sets ``skip_segmentation=True`` and ``skip_matching=True``.
  The 3D point cloud comes directly from the mesh vertices.

* **sam_auto / yolo_sam** (frame-first): iterates over camera frames like
  the trajectory backend.  2D masks are lifted to 3D by projecting mesh
  vertices onto the frame and keeping those that fall inside each mask.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import open3d as o3d

from conceptgraph.slam.geometry.base import FrameContext, GeometryBackend
from conceptgraph.slam.geometry.mesh_io import load_instance_mesh
from conceptgraph.slam.geometry.projection import (
    project_points_to_frame,
    select_best_views,
)
from conceptgraph.slam.utils import get_bounding_box


# ---------------------------------------------------------------------------
# Context dataclass
# ---------------------------------------------------------------------------

@dataclass
class GTMeshContext:
    instance_pcds: dict[int, o3d.geometry.PointCloud]
    all_vertices: np.ndarray       # (V, 3)
    all_colors: Optional[np.ndarray]  # (V, 3) or None
    vertex_instance_ids: np.ndarray  # (V,)
    frames: list[dict]
    class_map: dict[int, str]
    seg_backend: str


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class GTMeshBackend(GeometryBackend):
    """Ground-truth mesh geometry backend."""

    def load(self, cfg: Any) -> GTMeshContext:
        mesh_path = cfg.mesh_path
        if not mesh_path:
            raise ValueError("gt_mesh mode requires cfg.mesh_path to be set.")

        mesh_format = cfg.get("mesh_format", "replica")
        label_key = cfg.get("instance_label_key", "objectId")

        vertices, colors, instance_ids = load_instance_mesh(
            mesh_path, mesh_format, label_key
        )

        instance_pcds: dict[int, o3d.geometry.PointCloud] = {}
        for iid in np.unique(instance_ids):
            mask = instance_ids == iid
            pts = vertices[mask]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(colors[mask])
            instance_pcds[int(iid)] = pcd

        frames = self._load_camera_trajectory(cfg)

        class_map: dict[int, str] = {}
        class_map_path = cfg.get("instance_class_map", None)
        if class_map_path:
            with open(class_map_path) as f:
                raw = json.load(f)
            class_map = {int(k): v for k, v in raw.items()}

        seg_backend = cfg.get("segmentation_backend", "sam_auto")

        return GTMeshContext(
            instance_pcds=instance_pcds,
            all_vertices=vertices,
            all_colors=colors,
            vertex_instance_ids=instance_ids,
            frames=frames,
            class_map=class_map,
            seg_backend=seg_backend,
        )

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def get_iterator(self, ctx: GTMeshContext) -> Iterator[FrameContext]:
        if ctx.seg_backend == "gt_instances":
            yield from self._iter_gt_instances(ctx)
        else:
            yield from self._iter_frames(ctx)

    def _iter_gt_instances(self, ctx: GTMeshContext) -> Iterator[FrameContext]:
        """Object-first loop: one FrameContext per GT instance."""
        top_k = 5
        min_vis = 50

        for obj_idx, (iid, pcd) in enumerate(ctx.instance_pcds.items()):
            pts = np.asarray(pcd.points)
            if len(pts) < min_vis:
                continue

            best_views = select_best_views(
                pts, ctx.frames, top_k=top_k, min_visible=min_vis
            )
            if not best_views:
                continue

            best = best_views[0]
            image_rgb = self._load_image(best["color_path"])
            H, W = image_rgb.shape[:2]

            pixel_coords, valid = project_points_to_frame(
                pts, best["pose"], best["intrinsics"], H, W,
            )
            u = pixel_coords[valid, 0].astype(int)
            v = pixel_coords[valid, 1].astype(int)
            x_min, x_max = max(0, u.min()), min(W, u.max() + 1)
            y_min, y_max = max(0, v.min()), min(H, v.max() + 1)
            xyxy = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)

            mask = np.zeros((1, H, W), dtype=bool)
            mask[0, v, u] = True

            class_name = ctx.class_map.get(iid, "object")

            raw_gobs = {
                "xyxy": xyxy,
                "confidence": np.array([1.0], dtype=np.float32),
                "class_id": np.array([0], dtype=np.int32),
                "mask": mask,
                "classes": [class_name],
                "image_crops": [],
                "image_feats": np.zeros((1, 512), dtype=np.float32),
                "text_feats": np.zeros((1, 512), dtype=np.float32),
                "detection_class_labels": [f"{class_name} 0"],
                "labels": [f"{class_name} 0"],
                "edges": [],
                "captions": [class_name],
                "vlm_vit_feats": None,
                "vlm_proj_feats": None,
            }

            yield FrameContext(
                frame_idx=obj_idx,
                color_path=Path(best["color_path"]),
                image_rgb=image_rgb,
                intrinsics=best["intrinsics"],
                pose=best["pose"],
                skip_segmentation=True,
                skip_matching=True,
                instance_id=iid,
                extra={
                    "raw_gobs": raw_gobs,
                    "instance_pcd": pcd,
                    "best_views": best_views,
                },
            )

    def _iter_frames(self, ctx: GTMeshContext) -> Iterator[FrameContext]:
        """Frame-first loop: one FrameContext per camera frame."""
        for fi, fr in enumerate(ctx.frames):
            image_rgb = self._load_image(fr["color_path"])
            yield FrameContext(
                frame_idx=fi,
                color_path=Path(fr["color_path"]),
                image_rgb=image_rgb,
                intrinsics=fr["intrinsics"],
                pose=fr["pose"],
                extra={
                    "all_vertices": ctx.all_vertices,
                    "all_colors": ctx.all_colors,
                },
            )

    # ------------------------------------------------------------------
    # 3D lifting
    # ------------------------------------------------------------------

    def lift_to_3d(
        self,
        masks: np.ndarray,
        frame_ctx: FrameContext,
        cfg: Any,
    ) -> list[dict | None]:
        if frame_ctx.skip_segmentation and frame_ctx.extra.get("instance_pcd"):
            return self._lift_gt_instance(frame_ctx, cfg)
        return self._lift_mask_via_mesh(masks, frame_ctx, cfg)

    def _lift_gt_instance(
        self, frame_ctx: FrameContext, cfg: Any
    ) -> list[dict | None]:
        """Direct PCD from mesh — used in gt_instances mode."""
        pcd = frame_ctx.extra["instance_pcd"]
        bbox = get_bounding_box(cfg.get("spatial_sim_type", "iou"), pcd)
        return [{"pcd": pcd, "bbox": bbox}]

    def _lift_mask_via_mesh(
        self,
        masks: np.ndarray,
        frame_ctx: FrameContext,
        cfg: Any,
    ) -> list[dict | None]:
        """Project mesh vertices into the frame, keep those inside each mask."""
        all_vertices = frame_ctx.extra["all_vertices"]
        all_colors = frame_ctx.extra.get("all_colors")
        H, W = frame_ctx.image_rgb.shape[:2]

        pose_w2c = np.linalg.inv(frame_ctx.pose)
        R, t = pose_w2c[:3, :3], pose_w2c[:3, 3:]
        pts_cam = (R @ all_vertices.T + t).T

        in_front = pts_cam[:, 2] > 0
        z_safe = np.where(in_front, pts_cam[:, 2], 1.0)

        K = frame_ctx.intrinsics
        u = (pts_cam[:, 0] * K[0, 0] / z_safe + K[0, 2]).astype(np.int32)
        v = (pts_cam[:, 1] * K[1, 1] / z_safe + K[1, 2]).astype(np.int32)
        in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        min_pts = cfg.get("min_points_threshold", 16)
        processed: list[dict | None] = [None] * len(masks)

        for i in range(len(masks)):
            mask_2d = masks[i]
            inside = in_bounds.copy()
            inside[in_bounds] &= mask_2d[v[in_bounds], u[in_bounds]]

            if inside.sum() < min_pts:
                continue

            pts = all_vertices[inside]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            if all_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(all_colors[inside])
            else:
                colors_from_image = frame_ctx.image_rgb[v[inside], u[inside]] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_from_image)

            bbox = get_bounding_box(cfg.get("spatial_sim_type", "iou"), pcd)
            if bbox.volume() < 1e-6:
                continue
            processed[i] = {"pcd": pcd, "bbox": bbox}

        return processed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def num_iterations(self, ctx: GTMeshContext) -> int:
        if ctx.seg_backend == "gt_instances":
            return self._precount_gt_instances(ctx)
        return len(ctx.frames)

    def _precount_gt_instances(self, ctx: GTMeshContext) -> int:
        """Count instances passing both point-threshold and view-availability filters.

        Mirrors the exact filtering logic in ``_iter_gt_instances`` so
        ``num_iterations()`` and the actual iterator yield count agree.
        """
        top_k = 5
        min_vis = 50
        count = 0
        for pcd in ctx.instance_pcds.values():
            pts = np.asarray(pcd.points)
            if len(pts) < min_vis:
                continue
            views = select_best_views(
                pts, ctx.frames, top_k=top_k, min_visible=min_vis
            )
            if not views:
                continue
            count += 1
        return count

    @staticmethod
    def _load_camera_trajectory(cfg: Any) -> list[dict]:
        """Load camera frames, discarding the dataset object."""
        frames, _dataset = GeometryBackend.load_camera_frames(cfg)
        return frames

    @staticmethod
    def _load_image(color_path: str | Path) -> np.ndarray:
        import cv2
        img = cv2.imread(str(color_path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
