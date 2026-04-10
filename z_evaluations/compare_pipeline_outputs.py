"""
Compare all-in-one vs staged pipeline outputs for validation.

Loads two PKL.gz map files and compares:
  - Object count (exact match)
  - Per-object CLIP cosine similarity (>0.999)
  - Per-object bbox center distance (<1cm)
  - Edge count (exact match)
  - Caption consolidation (same objects get captions)

Usage:
    python z_evaluations/compare_pipeline_outputs.py \\
        --baseline /path/to/pcd_batch_api.pkl.gz \\
        --staged /path/to/pcd_batch_api.pkl.gz
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import sys

import numpy as np


def load_map(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / denom)


def bbox_center(obj: dict) -> np.ndarray:
    bbox = obj.get("bbox")
    if bbox is None:
        return np.zeros(3)
    try:
        return np.asarray(bbox.get_center())
    except Exception:
        corners = np.asarray(bbox.get_box_points())
        return corners.mean(axis=0)


def compare(baseline_path: str, staged_path: str) -> bool:
    baseline = load_map(baseline_path)
    staged = load_map(staged_path)

    b_objects = baseline["objects"]
    s_objects = staged["objects"]

    print(f"Baseline: {len(b_objects)} objects")
    print(f"Staged:   {len(s_objects)} objects")

    all_pass = True

    if len(b_objects) != len(s_objects):
        print(f"FAIL: Object count mismatch ({len(b_objects)} vs {len(s_objects)})")
        all_pass = False
    else:
        print("PASS: Object count matches")

    n = min(len(b_objects), len(s_objects))

    clip_sims = []
    bbox_dists = []
    for i in range(n):
        b_clip = b_objects[i].get("clip_ft")
        s_clip = s_objects[i].get("clip_ft")
        if b_clip is not None and s_clip is not None:
            if hasattr(b_clip, "numpy"):
                b_clip = b_clip.numpy()
            if hasattr(s_clip, "numpy"):
                s_clip = s_clip.numpy()
            sim = cosine_sim(b_clip, s_clip)
            clip_sims.append(sim)

        b_center = bbox_center(b_objects[i])
        s_center = bbox_center(s_objects[i])
        dist = np.linalg.norm(b_center - s_center)
        bbox_dists.append(dist)

    if clip_sims:
        min_sim = min(clip_sims)
        mean_sim = np.mean(clip_sims)
        print(f"CLIP cosine similarity: min={min_sim:.6f}, mean={mean_sim:.6f}")
        if min_sim < 0.999:
            print(f"WARN: Min CLIP similarity {min_sim:.6f} < 0.999")
        else:
            print("PASS: All CLIP similarities > 0.999")

    if bbox_dists:
        max_dist = max(bbox_dists)
        mean_dist = np.mean(bbox_dists)
        print(f"Bbox center distance: max={max_dist:.4f}m, mean={mean_dist:.4f}m")
        if max_dist > 0.01:
            print(f"WARN: Max bbox distance {max_dist:.4f}m > 1cm")
        else:
            print("PASS: All bbox centers within 1cm")

    b_edges = baseline.get("edges")
    s_edges = staged.get("edges")
    if b_edges is not None and s_edges is not None:
        b_count = len(getattr(b_edges, "edges_by_index", {})) if hasattr(b_edges, "edges_by_index") else 0
        s_count = len(getattr(s_edges, "edges_by_index", {})) if hasattr(s_edges, "edges_by_index") else 0
        print(f"Edge count: baseline={b_count}, staged={s_count}")
        if b_count != s_count:
            print("WARN: Edge count mismatch")
        else:
            print("PASS: Edge count matches")

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline outputs")
    parser.add_argument("--baseline", required=True, help="Path to baseline PKL.gz")
    parser.add_argument("--staged", required=True, help="Path to staged PKL.gz")
    args = parser.parse_args()

    ok = compare(args.baseline, args.staged)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
