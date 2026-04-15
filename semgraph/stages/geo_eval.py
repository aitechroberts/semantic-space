"""
Stage A5 — Geometry fidelity evaluation.

Compares oracle scene objects against GT mesh instances using voxel IoU,
completeness, accuracy, and Chamfer distance.  Runs after oracle_finalize.py.
Purely geometric — no model calls, no GPU required.

Supports two matching strategies:
  - **gt_instances** fast path: direct 1:1 mapping when instance IDs are
    preserved (skip_matching=True bypasses merge).
  - **Hungarian** general path: AABB IoU + centroid distance fallback.

Optional ICP alignment for the ``sparse`` (DUSt3R) backend where the
reconstructed point cloud may be in an arbitrary coordinate frame.

Standalone usage::

    python -m semgraph.stages.geo_eval <hydra overrides>
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GT mesh loading
# ---------------------------------------------------------------------------

def _load_gt_instances(
    cfg: Any,
) -> tuple[dict[int, o3d.geometry.PointCloud], dict[int, str]]:
    """Load GT mesh and build per-instance point clouds + class map.

    Returns (instance_pcds, class_map) where instance_pcds maps instance ID
    to an Open3D PointCloud and class_map maps instance ID to class name.
    """
    from semgraph.slam.geometry.mesh_io import load_instance_mesh

    mesh_path = cfg.mesh_path
    if not mesh_path:
        raise ValueError("geo_eval requires cfg.mesh_path to be set.")

    mesh_format = cfg.get("mesh_format", "replica")
    label_key = cfg.get("instance_label_key", "objectId")

    vertices, colors, instance_ids = load_instance_mesh(
        mesh_path, mesh_format, label_key,
    )

    instance_pcds: dict[int, o3d.geometry.PointCloud] = {}
    for iid in np.unique(instance_ids):
        mask = instance_ids == iid
        pts = vertices[mask]
        if len(pts) == 0:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        instance_pcds[int(iid)] = pcd

    class_map: dict[int, str] = {}
    class_map_path = cfg.get("instance_class_map", None)
    if class_map_path:
        with open(class_map_path) as f:
            raw = json.load(f)
        class_map = {int(k): v for k, v in raw.items()}

    return instance_pcds, class_map


# ---------------------------------------------------------------------------
# AABB IoU (replicates logic from compare_oracles.py)
# ---------------------------------------------------------------------------

def _bbox_iou_3d(
    min1: np.ndarray, max1: np.ndarray,
    min2: np.ndarray, max2: np.ndarray,
) -> float:
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(np.prod(inter_dims))
    vol1 = float(np.prod(np.maximum(0.0, max1 - min1)))
    vol2 = float(np.prod(np.maximum(0.0, max2 - min2)))
    union_vol = vol1 + vol2 - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def _aabb_from_corners(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract (min, max) from (8, 3) bbox corners."""
    return corners.min(axis=0), corners.max(axis=0)


def _aabb_from_pcd(pcd: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return np.zeros(3), np.zeros(3)
    return pts.min(axis=0), pts.max(axis=0)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _hungarian_match(
    oracle_aabbs: list[tuple[np.ndarray, np.ndarray]],
    gt_aabbs: dict[int, tuple[np.ndarray, np.ndarray]],
    iou_threshold: float,
    centroid_max_dist: float,
) -> list[tuple[int, int, float]]:
    """Hungarian matching on AABB IoU with centroid distance fallback.

    Returns list of (oracle_idx, gt_instance_id, iou).
    """
    from scipy.optimize import linear_sum_assignment

    gt_ids = list(gt_aabbs.keys())
    n_oracle = len(oracle_aabbs)
    n_gt = len(gt_ids)
    if n_oracle == 0 or n_gt == 0:
        return []

    iou_matrix = np.zeros((n_oracle, n_gt), dtype=np.float64)
    for i, (omin, omax) in enumerate(oracle_aabbs):
        for j, gid in enumerate(gt_ids):
            gmin, gmax = gt_aabbs[gid]
            iou_matrix[i, j] = _bbox_iou_3d(omin, omax, gmin, gmax)

    cost = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for r, c in zip(row_ind, col_ind):
        iou = iou_matrix[r, c]
        if iou >= iou_threshold:
            matches.append((int(r), gt_ids[c], float(iou)))
        elif iou >= iou_threshold * 0.5:
            omin, omax = oracle_aabbs[r]
            gmin, gmax = gt_aabbs[gt_ids[c]]
            o_center = (omin + omax) / 2
            g_center = (gmin + gmax) / 2
            dist = float(np.linalg.norm(o_center - g_center))
            if dist <= centroid_max_dist:
                matches.append((int(r), gt_ids[c], float(iou)))

    return matches


def _direct_match_gt_instances(
    oracle_class_names: list[str],
    gt_class_map: dict[int, str],
    gt_ids_sorted: list[int],
) -> list[tuple[int, int]] | None:
    """Attempt direct 1:1 mapping for gt_instances backend.

    Returns list of (oracle_idx, gt_instance_id) or None if counts don't match.
    """
    if len(oracle_class_names) != len(gt_ids_sorted):
        return None

    matches = []
    for oidx, gid in enumerate(gt_ids_sorted):
        gt_name = gt_class_map.get(gid, "object")
        oracle_name = oracle_class_names[oidx]
        if oracle_name != gt_name:
            logger.warning(
                "Direct mapping mismatch at oracle[%d]: oracle=%r vs gt[%d]=%r",
                oidx, oracle_name, gid, gt_name,
            )
        matches.append((oidx, gid))
    return matches


# ---------------------------------------------------------------------------
# Per-object metrics
# ---------------------------------------------------------------------------

def _voxel_metrics(
    oracle_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> dict[str, float]:
    """Compute volumetric IoU, completeness, and accuracy."""
    vg_oracle = o3d.geometry.VoxelGrid.create_from_point_cloud(oracle_pcd, voxel_size)
    vg_gt = o3d.geometry.VoxelGrid.create_from_point_cloud(gt_pcd, voxel_size)

    set_oracle = {tuple(v.grid_index) for v in vg_oracle.get_voxels()}
    set_gt = {tuple(v.grid_index) for v in vg_gt.get_voxels()}

    intersection = len(set_oracle & set_gt)
    union = len(set_oracle | set_gt)

    return {
        "vol_iou": intersection / union if union > 0 else 0.0,
        "completeness": intersection / len(set_gt) if set_gt else 0.0,
        "accuracy": intersection / len(set_oracle) if set_oracle else 0.0,
        "n_oracle_voxels": len(set_oracle),
        "n_gt_voxels": len(set_gt),
    }


def _chamfer_distances(
    oracle_pcd: o3d.geometry.PointCloud,
    gt_pcd: o3d.geometry.PointCloud,
) -> dict[str, float]:
    """Compute directional Chamfer distances (mean nearest-neighbor)."""
    if len(oracle_pcd.points) == 0 or len(gt_pcd.points) == 0:
        return {"chamfer_oracle_to_gt": float("inf"), "chamfer_gt_to_oracle": float("inf")}

    d_o2g = np.asarray(oracle_pcd.compute_point_cloud_distance(gt_pcd))
    d_g2o = np.asarray(gt_pcd.compute_point_cloud_distance(oracle_pcd))
    return {
        "chamfer_oracle_to_gt": float(d_o2g.mean()),
        "chamfer_gt_to_oracle": float(d_g2o.mean()),
    }


# ---------------------------------------------------------------------------
# ICP alignment for sparse backend
# ---------------------------------------------------------------------------

def _icp_align(
    oracle_pcds: list[np.ndarray],
    gt_pcds: list[o3d.geometry.PointCloud],
    voxel_size: float,
) -> tuple[np.ndarray, float, float]:
    """Align concatenated oracle scene to GT scene via ICP.

    Returns (4x4 transform, fitness, rmse).
    """
    oracle_pts = [p for p in oracle_pcds if len(p) > 0]
    gt_pts = [np.asarray(g.points) for g in gt_pcds if len(g.points) > 0]

    if not oracle_pts or not gt_pts:
        return np.eye(4), 0.0, float("inf")

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.concatenate(oracle_pts))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(np.concatenate(gt_pts))

    src = src.voxel_down_sample(voxel_size * 2)
    tgt = tgt.voxel_down_sample(voxel_size * 2)

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=voxel_size * 10,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return result.transformation, result.fitness, result.inlier_rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Geometry fidelity evaluation: oracle scene vs GT mesh."""
    from semgraph.stages.paths import stage_paths
    from semgraph.io import load_oracle_scene
    from semgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)

    geo_cfg = cfg.get("geo_eval", {})
    voxel_size = geo_cfg.get("voxel_size", 0.005)
    match_iou_threshold = geo_cfg.get("match_iou_threshold", 0.2)
    centroid_max_dist = geo_cfg.get("centroid_max_dist", 0.15)
    compute_chamfer = geo_cfg.get("compute_chamfer", True)
    align_icp = geo_cfg.get("align_icp", False)
    validate_gt_instances = geo_cfg.get("validate_gt_instances", True)

    # --- Load oracle scene ---
    oracle = load_oracle_scene(paths["oracle"])
    if oracle is None:
        raise FileNotFoundError(f"oracle_scene not found at {paths['oracle']}")

    n_oracle = len(oracle.obj_pcd_points_list)
    print(f"[geo_eval] Oracle: {n_oracle} objects")

    # --- Load GT mesh instances ---
    gt_pcds, gt_class_map = _load_gt_instances(cfg)
    gt_ids_sorted = sorted(gt_pcds.keys())
    n_gt = len(gt_ids_sorted)
    print(f"[geo_eval] GT mesh: {n_gt} instances")

    # --- Build AABBs ---
    oracle_aabbs = []
    for i in range(n_oracle):
        if oracle.obj_bbox_corners.shape[0] > i:
            oracle_aabbs.append(_aabb_from_corners(oracle.obj_bbox_corners[i]))
        else:
            pts = oracle.obj_pcd_points_list[i]
            if len(pts) > 0:
                oracle_aabbs.append((pts.min(axis=0), pts.max(axis=0)))
            else:
                oracle_aabbs.append((np.zeros(3), np.zeros(3)))

    gt_aabbs = {gid: _aabb_from_pcd(gt_pcds[gid]) for gid in gt_ids_sorted}

    # --- Matching ---
    match_method = "hungarian"
    agreement = None

    seg_backend = cfg.get("segmentation_backend", "")
    direct_matches = None
    if seg_backend == "gt_instances" and gt_class_map:
        direct_matches = _direct_match_gt_instances(
            oracle.class_names, gt_class_map, gt_ids_sorted,
        )
        if direct_matches is not None:
            match_method = "direct"
            print(f"[geo_eval] Direct gt_instances mapping: {len(direct_matches)} pairs")

    hungarian_matches = _hungarian_match(
        oracle_aabbs, gt_aabbs, match_iou_threshold, centroid_max_dist,
    )

    if direct_matches is not None and validate_gt_instances:
        direct_set = {(o, g) for o, g in direct_matches}
        hungarian_set = {(o, g) for o, g, _ in hungarian_matches}
        agreement = direct_set == hungarian_set
        if not agreement:
            logger.warning(
                "Direct and Hungarian mappings disagree: "
                "direct=%d pairs, hungarian=%d pairs. "
                "This may indicate merge happened despite skip_matching.",
                len(direct_set), len(hungarian_set),
            )

    if direct_matches is not None:
        final_matches = [(o, g, 1.0) for o, g in direct_matches]
    else:
        final_matches = hungarian_matches

    matched_oracle_idxs = {o for o, _, _ in final_matches}
    matched_gt_ids = {g for _, g, _ in final_matches}
    unmatched_oracle = [i for i in range(n_oracle) if i not in matched_oracle_idxs]
    unmatched_gt = [gid for gid in gt_ids_sorted if gid not in matched_gt_ids]

    print(f"[geo_eval] Matched: {len(final_matches)}, "
          f"unmatched oracle: {len(unmatched_oracle)}, "
          f"unmatched GT: {len(unmatched_gt)}")

    # --- Optional ICP alignment ---
    icp_info: dict[str, Any] = {}
    transform = None
    if align_icp:
        print("[geo_eval] Running ICP alignment...")
        transform, fitness, rmse = _icp_align(
            oracle.obj_pcd_points_list,
            [gt_pcds[gid] for gid in gt_ids_sorted],
            voxel_size,
        )
        icp_info = {"icp_fitness": float(fitness), "icp_rmse": float(rmse)}
        print(f"[geo_eval] ICP fitness={fitness:.4f}, RMSE={rmse:.6f}")

    # --- Per-object metrics ---
    per_object_results = []
    for oracle_idx, gt_id, match_iou in final_matches:
        oracle_pts = oracle.obj_pcd_points_list[oracle_idx]

        if transform is not None and len(oracle_pts) > 0:
            ones = np.ones((len(oracle_pts), 1))
            pts_h = np.hstack([oracle_pts, ones])
            oracle_pts = (transform @ pts_h.T).T[:, :3]

        o_pcd = o3d.geometry.PointCloud()
        o_pcd.points = o3d.utility.Vector3dVector(oracle_pts)
        g_pcd = gt_pcds[gt_id]

        bbox_center = oracle_pts.mean(axis=0).tolist() if len(oracle_pts) > 0 else [0, 0, 0]

        metrics = _voxel_metrics(o_pcd, g_pcd, voxel_size)

        if compute_chamfer and len(oracle_pts) > 0:
            chamfer = _chamfer_distances(o_pcd, g_pcd)
        else:
            chamfer = {"chamfer_oracle_to_gt": None, "chamfer_gt_to_oracle": None}

        per_object_results.append({
            "oracle_idx": oracle_idx,
            "bbox_center": bbox_center,
            "gt_instance_id": gt_id,
            "gt_class_name": gt_class_map.get(gt_id, "unknown"),
            "match_iou": match_iou,
            **metrics,
            **chamfer,
        })

    # --- Scene-level aggregates ---
    def _agg(key: str) -> dict[str, float]:
        vals = [r[key] for r in per_object_results if r[key] is not None]
        if not vals:
            return {"mean": 0.0, "median": 0.0}
        return {"mean": float(np.mean(vals)), "median": float(np.median(vals))}

    recall = len(final_matches) / n_gt if n_gt > 0 else 0.0
    precision = len(final_matches) / n_oracle if n_oracle > 0 else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

    scene_level = {
        "n_oracle": n_oracle,
        "n_gt": n_gt,
        "n_matched": len(final_matches),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "vol_iou": _agg("vol_iou"),
        "completeness": _agg("completeness"),
        "accuracy": _agg("accuracy"),
        **icp_info,
    }
    if compute_chamfer:
        scene_level["chamfer_oracle_to_gt"] = _agg("chamfer_oracle_to_gt")
        scene_level["chamfer_gt_to_oracle"] = _agg("chamfer_gt_to_oracle")

    # --- Write output ---
    output = {
        "scene_level": scene_level,
        "per_object": per_object_results,
        "unmatched_oracle": unmatched_oracle,
        "unmatched_gt": unmatched_gt,
        "matching": {
            "method": match_method,
            **({"agreement": agreement} if agreement is not None else {}),
        },
        "config": {
            "voxel_size": voxel_size,
            "match_iou_threshold": match_iou_threshold,
            "centroid_max_dist": centroid_max_dist,
            "compute_chamfer": compute_chamfer,
            "align_icp": align_icp,
            "mesh_path": str(cfg.get("mesh_path", "")),
            "pipeline_mode": str(cfg.get("pipeline_mode", "")),
            "segmentation_backend": seg_backend,
        },
    }

    out_dir = paths["geo_eval"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "geometry.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[geo_eval] === Results ===")
    print(f"  Matched:      {len(final_matches)}/{n_gt} GT instances (recall={recall:.3f})")
    print(f"  Precision:    {precision:.3f}  F1: {f1:.3f}")
    print(f"  Vol IoU:      mean={scene_level['vol_iou']['mean']:.3f}, "
          f"median={scene_level['vol_iou']['median']:.3f}")
    print(f"  Completeness: mean={scene_level['completeness']['mean']:.3f}, "
          f"median={scene_level['completeness']['median']:.3f}")
    print(f"  Accuracy:     mean={scene_level['accuracy']['mean']:.3f}, "
          f"median={scene_level['accuracy']['median']:.3f}")
    if compute_chamfer:
        print(f"  Chamfer O->G: mean={scene_level['chamfer_oracle_to_gt']['mean']:.4f}")
        print(f"  Chamfer G->O: mean={scene_level['chamfer_gt_to_oracle']['mean']:.4f}")
    print(f"\n[geo_eval] Results saved to {out_path}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
