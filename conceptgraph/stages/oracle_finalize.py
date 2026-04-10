"""
Stage A4 — Oracle finalization: geometric post-processing.

Runs after build_map.py. Purely geometric — no model calls, no LLM,
no captions, no scene type inference.

1. MST edge construction: 3D bbox IoU between all object pairs, max-weight
   MST via scipy, output unlabeled candidate edges.
2. HPSG plane detection: RANSAC plane fitting on concatenated scene PCD,
   DBSCAN in parameter space, normal-based classification (floor/wall/ceiling),
   object-to-plane anchoring.

Saves the immutable oracle_scene.

Standalone usage::

    python -m conceptgraph.stages.oracle_finalize <hydra overrides>
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MST edge construction (unlabeled)
# ---------------------------------------------------------------------------

def _build_mst_edges(objects: Any) -> list[tuple[int, int, float]]:
    """Compute 3D bbox IoU between all pairs, return max-weight MST edges."""
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    from conceptgraph.slam.mapping import compute_3d_bbox_iou

    n = len(objects)
    if n < 2:
        return []

    iou_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        bbox_i = objects[i].get("bbox")
        if bbox_i is None:
            continue
        for j in range(i + 1, n):
            bbox_j = objects[j].get("bbox")
            if bbox_j is None:
                continue
            iou = compute_3d_bbox_iou(bbox_i, bbox_j)
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    # Negate for max-weight MST (scipy computes min-weight)
    neg_matrix = -iou_matrix
    neg_matrix[neg_matrix == 0] = 0  # keep zeros as zeros (no edge)
    sparse = csr_matrix(np.triu(neg_matrix, k=1))
    mst = minimum_spanning_tree(sparse)
    mst_dense = mst.toarray()

    edges = []
    rows, cols = np.nonzero(mst_dense)
    for r, c in zip(rows, cols):
        score = iou_matrix[r, c]
        if score > 0:
            edges.append((int(r), int(c), float(score)))

    return edges


# ---------------------------------------------------------------------------
# HPSG plane detection + object anchoring
# ---------------------------------------------------------------------------

def _detect_planes(objects: Any) -> tuple[list[dict], dict[int, int | None]]:
    """Detect structural planes from the scene PCD and anchor objects.

    Returns
    -------
    planes : list of plane records with plane_id, label, normal, offset
    anchoring : dict mapping object index -> parent_plane_id or None
    """
    all_points = []
    for obj in objects:
        pcd = obj.get("pcd")
        if pcd is not None:
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                all_points.append(pts)

    if not all_points:
        return [], {i: None for i in range(len(objects))}

    scene_pts = np.concatenate(all_points, axis=0)

    # RANSAC plane fitting — detect up to 5 major planes
    planes = []
    remaining_pts = scene_pts.copy()
    for plane_iter in range(5):
        if len(remaining_pts) < 100:
            break

        plane, inlier_mask = _ransac_plane(remaining_pts, threshold=0.03, n_iterations=1000)
        if plane is None or inlier_mask.sum() < 50:
            break

        normal, offset = plane[:3], plane[3]

        # Normalize normal direction
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            remaining_pts = remaining_pts[~inlier_mask]
            continue
        normal = normal / norm_len
        offset = offset / norm_len

        label = _classify_plane(normal)
        inlier_points = remaining_pts[inlier_mask]

        planes.append({
            "plane_id": len(planes),
            "label": label,
            "normal": normal.tolist(),
            "offset": float(offset),
            "inlier_points": inlier_points,
        })

        remaining_pts = remaining_pts[~inlier_mask]

    # Anchor objects to nearest plane
    anchoring: dict[int, int | None] = {}
    for obj_idx, obj in enumerate(objects):
        pcd = obj.get("pcd")
        if pcd is None or not planes:
            anchoring[obj_idx] = None
            continue

        centroid = np.asarray(pcd.points).mean(axis=0)
        best_plane_id = None
        best_dist = float("inf")
        for p in planes:
            normal = np.array(p["normal"])
            offset = p["offset"]
            dist = abs(np.dot(normal, centroid) + offset)
            if dist < best_dist:
                best_dist = dist
                best_plane_id = p["plane_id"]

        anchoring[obj_idx] = best_plane_id

    return planes, anchoring


def _ransac_plane(
    points: np.ndarray,
    threshold: float = 0.03,
    n_iterations: int = 1000,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Simple RANSAC plane fitting. Returns (plane_coeffs, inlier_mask)."""
    n = len(points)
    best_inliers = np.zeros(n, dtype=bool)
    best_plane = None

    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p1)

        dists = np.abs(points @ normal + d)
        inliers = dists < threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = np.array([normal[0], normal[1], normal[2], d])

    return best_plane, best_inliers


def _classify_plane(normal: np.ndarray) -> str:
    """Classify a plane as floor, wall, or ceiling by normal alignment."""
    up = np.array([0.0, 0.0, 1.0])
    dot = np.dot(normal, up)
    if abs(dot) > 0.8:
        return "floor" if dot > 0 else "ceiling"
    return "wall"


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Build oracle scene: MST edges + HPSG planes + object anchoring."""
    from conceptgraph.stages.paths import stage_paths
    from conceptgraph.stages import io as stage_io
    from conceptgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    paths["oracle_scene"].mkdir(parents=True, exist_ok=True)

    result = stage_io.load_map(paths["map"])
    if result is None:
        raise FileNotFoundError(f"map.pkl.gz not found at {paths['map']}")
    objects, map_edges, saved_cfg = result

    print(f"[oracle_finalize] Building MST edges for {len(objects)} objects...")
    mst_edges = _build_mst_edges(objects)
    print(f"[oracle_finalize] {len(mst_edges)} MST edges")

    print("[oracle_finalize] Detecting structural planes...")
    planes, anchoring = _detect_planes(objects)
    print(f"[oracle_finalize] {len(planes)} planes detected")

    for obj_idx, parent_plane_id in anchoring.items():
        if obj_idx < len(objects):
            objects[obj_idx]["parent_plane_id"] = parent_plane_id

    # Strip large inlier_points from plane records before saving
    planes_for_save = []
    for p in planes:
        planes_for_save.append({
            "plane_id": p["plane_id"],
            "label": p["label"],
            "normal": p["normal"],
            "offset": p["offset"],
        })

    stage_io.save_oracle_scene(
        paths["oracle_scene"],
        objects=objects,
        edges=map_edges,
        planes=planes_for_save,
        mst_edges=mst_edges,
        cfg=cfg,
    )
    print("[oracle_finalize] Oracle scene saved.")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
