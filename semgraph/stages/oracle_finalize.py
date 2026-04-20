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

    python -m semgraph.stages.oracle_finalize <hydra overrides>
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
    from semgraph.slam.mapping import compute_3d_bbox_iou

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
    from semgraph.stages.paths import stage_paths
    from semgraph.io import load_map, save_oracle_scene, OracleSceneRecord
    from semgraph.io.records import _PerViewMeta
    from semgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    paths["oracle"].mkdir(parents=True, exist_ok=True)

    result = load_map(paths["map"])
    if result is None:
        raise FileNotFoundError(f"oracle_map not found at {paths['map']}")
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

    planes_for_save = []
    for p in planes:
        planes_for_save.append({
            "plane_id": p["plane_id"],
            "label": p["label"],
            "normal": p["normal"],
            "offset": p["offset"],
        })

    # Build OracleSceneRecord from live objects
    obj_pcd_points = []
    obj_pcd_colors = []
    bbox_corners_list = []
    class_names = []
    parent_plane_ids = []
    per_view_meta: list[list[_PerViewMeta]] = []
    pv_clip_ft_all: list[np.ndarray] = []

    for obj_idx, obj in enumerate(objects):
        pcd = obj.get("pcd")
        pts = np.asarray(pcd.points) if pcd is not None else np.empty((0, 3))
        cols = np.asarray(pcd.colors) if pcd is not None and pcd.has_colors() else np.zeros_like(pts)
        obj_pcd_points.append(pts)
        obj_pcd_colors.append(cols)

        bbox = obj.get("bbox")
        if bbox is not None:
            bbox_corners_list.append(np.asarray(bbox.get_box_points()))
        else:
            bbox_corners_list.append(np.zeros((8, 3)))

        class_names.append(obj.get("class_name", "object"))
        parent_plane_ids.append(obj.get("parent_plane_id"))

        pvr_raw = obj.get("per_view_records", [])
        obj_pv_meta = []
        obj_pv_feats = []
        for r in pvr_raw:
            gt_iid = r.get("gt_instance_id")
            n_vis = r.get("n_visible")
            obj_pv_meta.append(_PerViewMeta(
                frame_idx=int(r.get("frame_idx", 0)),
                n_points=int(r.get("n_points", 0)),
                crop_path=r.get("crop_path", ""),
                crop_bbox=r.get("crop_bbox", []),
                gt_instance_id=int(gt_iid) if gt_iid is not None else None,
                n_visible=int(n_vis) if n_vis is not None else None,
            ))
            ft = r.get("clip_ft")
            if ft is not None:
                obj_pv_feats.append(np.asarray(ft))
        per_view_meta.append(obj_pv_meta)
        if obj_pv_feats:
            pv_clip_ft_all.append(np.stack(obj_pv_feats))
        else:
            pv_clip_ft_all.append(np.empty((0, 0), dtype=np.float32))

    record = OracleSceneRecord(
        obj_pcd_points_list=obj_pcd_points,
        obj_pcd_colors_list=obj_pcd_colors,
        obj_bbox_corners=np.stack(bbox_corners_list) if bbox_corners_list else np.empty((0, 8, 3)),
        pv_clip_ft_list=pv_clip_ft_all,
        class_names=class_names,
        parent_plane_ids=parent_plane_ids,
        planes=planes_for_save,
        mst_edges=mst_edges,
        per_view_meta=per_view_meta,
    )
    save_oracle_scene(paths["oracle"], record)
    print("[oracle_finalize] Oracle scene saved.")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
