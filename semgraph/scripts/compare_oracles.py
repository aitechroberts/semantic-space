"""Compare a subset oracle scene against a reference oracle for geometric quality.

Purely geometric evaluation — no encoders, no language, no GPU.  Runs in
seconds on any machine.

Usage::

    python -m semgraph.scripts.compare_oracles \
        --reference /data/office0/stages/oracle \
        --subset /data/ablations/fps_pose_30/office0/stages/oracle \
        --output results/fps_pose_30.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from semgraph.io import load_oracle_scene, OracleSceneRecord


# ---------------------------------------------------------------------------
# 3D AABB IoU
# ---------------------------------------------------------------------------

def _bbox_iou_3d(
    min1: np.ndarray, max1: np.ndarray,
    min2: np.ndarray, max2: np.ndarray,
) -> float:
    """Axis-aligned 3D bounding box IoU (intersection volume / union volume)."""
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


def _extract_aabbs(record: OracleSceneRecord) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-object AABB (min, max) from 8-corner bbox representation.

    Returns (mins, maxs) each of shape (N, 3).
    """
    corners = record.obj_bbox_corners  # (N, 8, 3)
    if corners.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    mins = corners.min(axis=1)  # (N, 3)
    maxs = corners.max(axis=1)  # (N, 3)
    return mins, maxs


# ---------------------------------------------------------------------------
# IoU matrix and matching
# ---------------------------------------------------------------------------

def _compute_iou_matrix(
    ref_mins: np.ndarray, ref_maxs: np.ndarray,
    sub_mins: np.ndarray, sub_maxs: np.ndarray,
) -> np.ndarray:
    """Compute (N_ref, N_sub) pairwise 3D AABB IoU matrix."""
    n_ref = len(ref_mins)
    n_sub = len(sub_mins)
    iou_mat = np.zeros((n_ref, n_sub), dtype=np.float64)
    for i in range(n_ref):
        for j in range(n_sub):
            iou_mat[i, j] = _bbox_iou_3d(
                ref_mins[i], ref_maxs[i], sub_mins[j], sub_maxs[j],
            )
    return iou_mat


def _compute_metrics(
    iou_matrix: np.ndarray,
    threshold: float,
) -> dict:
    """Compute recall, precision, F1, fragmentation, mean IoU from the IoU matrix."""
    n_ref, n_sub = iou_matrix.shape

    if n_ref == 0 and n_sub == 0:
        return {
            "recall": 1.0, "precision": 1.0, "f1": 1.0,
            "fragmentation_count": 0, "fragmentation_ratio": 0.0,
            "mean_matched_iou": 0.0, "n_recalled": 0, "n_precise": 0,
        }
    if n_ref == 0:
        return {
            "recall": 1.0, "precision": 0.0, "f1": 0.0,
            "fragmentation_count": 0, "fragmentation_ratio": 0.0,
            "mean_matched_iou": 0.0, "n_recalled": 0, "n_precise": 0,
        }
    if n_sub == 0:
        return {
            "recall": 0.0, "precision": 1.0, "f1": 0.0,
            "fragmentation_count": 0, "fragmentation_ratio": 0.0,
            "mean_matched_iou": 0.0, "n_recalled": 0, "n_precise": 0,
        }

    # Recall: for each ref object, best-matching subset object
    best_sub_for_ref = iou_matrix.max(axis=1)  # (N_ref,)
    recalled_mask = best_sub_for_ref >= threshold
    n_recalled = int(recalled_mask.sum())
    recall = n_recalled / n_ref

    # Precision: for each subset object, best-matching ref object
    best_ref_for_sub = iou_matrix.max(axis=0)  # (N_sub,)
    precise_mask = best_ref_for_sub >= threshold
    n_precise = int(precise_mask.sum())
    precision = n_precise / n_sub

    # F1
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    # Fragmentation: count subset objects mapping to the same ref object
    sub_to_ref = iou_matrix.argmax(axis=0)  # (N_sub,) — best ref for each sub
    sub_matched = precise_mask
    matched_ref_ids = sub_to_ref[sub_matched]
    unique_refs_hit = len(set(matched_ref_ids.tolist()))
    n_matched_sub = int(sub_matched.sum())
    fragmentation_count = max(0, n_matched_sub - unique_refs_hit)
    fragmentation_ratio = fragmentation_count / n_sub if n_sub > 0 else 0.0

    # Mean IoU of recalled pairs
    matched_ious = best_sub_for_ref[recalled_mask]
    mean_matched_iou = float(matched_ious.mean()) if len(matched_ious) > 0 else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "fragmentation_count": fragmentation_count,
        "fragmentation_ratio": fragmentation_ratio,
        "mean_matched_iou": mean_matched_iou,
        "n_recalled": n_recalled,
        "n_precise": n_precise,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare subset oracle scene vs reference (geometric metrics).",
    )
    parser.add_argument("--reference", type=Path, required=True,
                        help="Path to reference oracle directory (contains oracle_scene.npz)")
    parser.add_argument("--subset", type=Path, required=True,
                        help="Path to subset oracle directory")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to write results JSON (also prints to stdout)")
    parser.add_argument("--iou_threshold", type=float, default=0.25,
                        help="IoU threshold for matching objects (default: 0.25)")
    args = parser.parse_args()

    # --- Load oracle scenes ---
    ref = load_oracle_scene(args.reference)
    if ref is None:
        print(f"Error: could not load reference oracle from {args.reference}", file=sys.stderr)
        sys.exit(1)

    sub = load_oracle_scene(args.subset)
    if sub is None:
        print(f"Error: could not load subset oracle from {args.subset}", file=sys.stderr)
        sys.exit(1)

    # --- Extract AABBs ---
    ref_mins, ref_maxs = _extract_aabbs(ref)
    sub_mins, sub_maxs = _extract_aabbs(sub)

    n_ref = len(ref_mins)
    n_sub = len(sub_mins)

    # --- Compute IoU matrix and metrics ---
    iou_matrix = _compute_iou_matrix(ref_mins, ref_maxs, sub_mins, sub_maxs)
    metrics = _compute_metrics(iou_matrix, args.iou_threshold)

    results = {
        "ref_path": str(args.reference.resolve()),
        "subset_path": str(args.subset.resolve()),
        "iou_threshold": args.iou_threshold,
        "ref_n_objects": n_ref,
        "subset_n_objects": n_sub,
        **metrics,
    }

    # --- Output ---
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to {args.output}")

    print()
    print(f"  Reference:    {args.reference}  ({n_ref} objects)")
    print(f"  Subset:       {args.subset}  ({n_sub} objects)")
    print(f"  IoU threshold: {args.iou_threshold}")
    print(f"  ---")
    print(f"  Recall:        {metrics['recall']:.3f}  ({metrics['n_recalled']}/{n_ref})")
    print(f"  Precision:     {metrics['precision']:.3f}  ({metrics['n_precise']}/{n_sub})")
    print(f"  F1:            {metrics['f1']:.3f}")
    print(f"  Mean IoU:      {metrics['mean_matched_iou']:.3f}")
    print(f"  Fragmentation: {metrics['fragmentation_count']}  ({metrics['fragmentation_ratio']:.3f})")


if __name__ == "__main__":
    main()
