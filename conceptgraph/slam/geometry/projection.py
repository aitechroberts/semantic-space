"""
3D-to-2D projection utilities.

Pure geometric math — no pipeline or backend dependencies.  Used by
``gt_mesh.py`` for mesh-based lifting and view selection, by
``detect.py`` for 1.5x projected crop computation, and by
``postprocess.py`` for Rerun visualization.
"""

from __future__ import annotations

import numpy as np


def project_points_to_frame(
    points_world: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics_3x3: np.ndarray,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame 3D points onto a camera image plane.

    Parameters
    ----------
    points_world : (N, 3)
    pose_c2w : (4, 4) camera-to-world
    intrinsics_3x3 : (3, 3) camera intrinsic matrix
    H, W : image dimensions

    Returns
    -------
    pixel_coords : (N, 2)  (u, v) — may be out of bounds
    valid_mask   : (N,) bool — in front of camera AND inside image bounds
    """
    pose_w2c = np.linalg.inv(pose_c2w)
    R, t = pose_w2c[:3, :3], pose_w2c[:3, 3:]
    pts_cam = (R @ points_world.T + t).T  # (N, 3)

    in_front = pts_cam[:, 2] > 0
    z_safe = np.where(in_front, pts_cam[:, 2], 1.0)

    fx, fy = intrinsics_3x3[0, 0], intrinsics_3x3[1, 1]
    cx, cy = intrinsics_3x3[0, 2], intrinsics_3x3[1, 2]
    u = pts_cam[:, 0] * fx / z_safe + cx
    v = pts_cam[:, 1] * fy / z_safe + cy

    in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pixel_coords = np.stack([u, v], axis=1)
    return pixel_coords, in_bounds


def select_best_views(
    instance_pcd_points: np.ndarray,
    frames: list[dict],
    top_k: int = 5,
    min_visible: int = 50,
) -> list[dict]:
    """Rank camera frames by how many instance points project into their bounds.

    Parameters
    ----------
    instance_pcd_points : (N, 3) world-frame points for one object instance.
    frames : list of dicts with keys ``pose``, ``intrinsics``, ``H``, ``W``.
    top_k : maximum number of views to return.
    min_visible : discard views with fewer visible points.

    Returns
    -------
    list[dict]
        Up to *top_k* frame dicts, sorted by descending visibility count.
    """
    scores: list[tuple[int, int]] = []
    for fi, fr in enumerate(frames):
        _, valid = project_points_to_frame(
            instance_pcd_points,
            fr["pose"],
            fr["intrinsics"],
            fr["H"],
            fr["W"],
        )
        scores.append((int(valid.sum()), fi))

    scores.sort(key=lambda x: -x[0])
    return [frames[fi] for count, fi in scores[:top_k] if count >= min_visible]


def compute_projected_crop_bbox(
    pcd_points: np.ndarray,
    pose_c2w: np.ndarray,
    intrinsics_3x3: np.ndarray,
    H: int,
    W: int,
    scale: float = 1.5,
) -> tuple[int, int, int, int] | None:
    """Project 3D points onto image plane and return a scaled 2D crop bbox.

    Projects the object's 3D point cloud into the camera frame, computes
    the tight 2D bounding box around valid projections, scales it by
    *scale* from center, and clamps to image bounds.

    Parameters
    ----------
    pcd_points : (N, 3)
        World-frame 3D points for one detection/object.
    pose_c2w : (4, 4) camera-to-world transform.
    intrinsics_3x3 : (3, 3) camera intrinsic matrix.
    H, W : image height and width.
    scale : expansion factor applied to the tight bbox (default 1.5).

    Returns
    -------
    (x_min, y_min, x_max, y_max) as ints clamped to [0, W) x [0, H),
    or ``None`` if no points project inside the image.
    """
    if len(pcd_points) == 0:
        return None

    pixel_coords, valid_mask = project_points_to_frame(
        pcd_points, pose_c2w, intrinsics_3x3, H, W,
    )
    if not valid_mask.any():
        return None

    valid_uv = pixel_coords[valid_mask]
    u_min, v_min = valid_uv.min(axis=0)
    u_max, v_max = valid_uv.max(axis=0)

    cx = (u_min + u_max) / 2.0
    cy = (v_min + v_max) / 2.0
    half_w = (u_max - u_min) / 2.0 * scale
    half_h = (v_max - v_min) / 2.0 * scale

    x_min = int(max(0, cx - half_w))
    y_min = int(max(0, cy - half_h))
    x_max = int(min(W, cx + half_w))
    y_max = int(min(H, cy + half_h))

    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)
