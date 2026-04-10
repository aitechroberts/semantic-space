"""
SparseBackend — DUSt3R RGB-only geometry path.

Runs DUSt3R on a set of RGB images (no depth, no poses, no intrinsics
required) to produce per-view 3D point maps and confidence masks.
2D detection masks are lifted to 3D by element-wise multiplication with the
pre-computed point map, following Sparse3DPR Equation 2:

    P_j_i = X_i ⊙ (M_j_i ⊙ C_i)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import cv2
import numpy as np
import open3d as o3d

from conceptgraph.slam.geometry.base import FrameContext, GeometryBackend
from conceptgraph.slam.utils import get_bounding_box


@dataclass
class SparseContext:
    """Opaque context returned by SparseBackend.load()."""
    image_paths: list[Path]
    pointmaps: list[Any]   # list of (H, W, 3) tensors/arrays per view
    confidence: list[Any]  # list of (H, W) bool/float tensors per view
    poses: list[Any]       # list of (4, 4) recovered camera-to-world
    focals: list[Any]      # list of recovered focal lengths


class SparseBackend(GeometryBackend):
    """DUSt3R RGB-only → per-pixel point map lifting."""

    def load(self, cfg: Any) -> SparseContext:
        try:
            from dust3r.inference import inference
            from dust3r.model import AsymmetricCroCo3DStereo
            from dust3r.utils.image import load_images
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        except ImportError:
            raise ImportError(
                "DUSt3R is required for sparse mode.  Install from: "
                "https://github.com/naver/dust3r"
            )

        import torch

        image_dir = cfg.get("image_dir", None)
        if not image_dir:
            raise ValueError("sparse mode requires cfg.image_dir to be set.")

        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No images found in {image_dir}")

        dust3r_model_name = cfg.get(
            "dust3r_model", "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        )
        device = cfg.get("device", "cuda")

        print(f"[sparse] Loading DUSt3R model: {dust3r_model_name}")
        model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_model_name).to(device)

        dust3r_size = cfg.get("dust3r_image_size", 512)
        images = load_images([str(p) for p in image_paths], size=dust3r_size)

        scene_graph = cfg.get("dust3r_scene_graph", "complete")
        pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)

        batch_size = cfg.get("dust3r_batch_size", 1)
        print(f"[sparse] Running DUSt3R inference on {len(image_paths)} images "
              f"({len(pairs)} pairs, scene_graph={scene_graph})")
        output = inference(pairs, model, device, batch_size=batch_size)

        niter = cfg.get("dust3r_niter", 300)
        schedule = cfg.get("dust3r_schedule", "cosine")
        lr = cfg.get("dust3r_lr", 0.01)

        if len(image_paths) <= 2:
            mode = GlobalAlignerMode.PairViewer
        else:
            mode = GlobalAlignerMode.PointCloudOptimizer

        print(f"[sparse] Running global alignment (mode={mode.name}, niter={niter})")
        scene = global_aligner(output, device=device, mode=mode)
        if mode != GlobalAlignerMode.PairViewer:
            scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        pointmaps = scene.get_pts3d()
        confidence = scene.get_masks()
        poses = scene.get_im_poses()
        focals = scene.get_focals()

        del model
        torch.cuda.empty_cache()
        print(f"[sparse] DUSt3R complete. {len(pointmaps)} point maps generated.")

        return SparseContext(
            image_paths=image_paths,
            pointmaps=pointmaps,
            confidence=confidence,
            poses=poses,
            focals=focals,
        )

    def get_iterator(self, ctx: SparseContext) -> Iterator[FrameContext]:
        import torch

        for i, img_path in enumerate(ctx.image_paths):
            image_rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

            ptmap = ctx.pointmaps[i]
            if torch.is_tensor(ptmap):
                ptmap = ptmap.detach().cpu().numpy()

            conf = ctx.confidence[i]
            if torch.is_tensor(conf):
                conf = conf.detach().cpu().numpy()

            pose = ctx.poses[i]
            if torch.is_tensor(pose):
                pose = pose.detach().cpu().numpy()

            yield FrameContext(
                frame_idx=i,
                color_path=img_path,
                image_rgb=image_rgb,
                pose=pose,
                extra={
                    "pointmap": ptmap,
                    "confidence": conf,
                },
            )

    def lift_to_3d(
        self,
        masks: np.ndarray,
        frame_ctx: FrameContext,
        cfg: Any,
    ) -> list[dict | None]:
        pointmap = frame_ctx.extra["pointmap"]  # (H, W, 3)
        conf = frame_ctx.extra["confidence"]     # (H, W)

        min_conf = cfg.get("dust3r_min_confidence", 0.5)
        min_pts = cfg.get("min_points_threshold", 16)

        if conf.dtype == bool:
            conf_mask = conf
        else:
            conf_mask = conf > min_conf

        processed: list[dict | None] = [None] * len(masks)

        for i in range(len(masks)):
            valid = masks[i] & conf_mask

            pts = pointmap[valid]
            if len(pts) < min_pts:
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)

            colors = frame_ctx.image_rgb[valid] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

            bbox = get_bounding_box(cfg.get("spatial_sim_type", "iou"), pcd)
            if bbox.volume() < 1e-6:
                continue

            processed[i] = {"pcd": pcd, "bbox": bbox}

        return processed

    def num_iterations(self, ctx: SparseContext) -> int:
        return len(ctx.image_paths)
