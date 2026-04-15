"""Create a symlinked frame_data subset from a full oracle run.

Reads poses from existing frame_data files (lightweight — only the npz
pose array and JSON metadata are touched), runs a frame selector from
``semgraph.sampling``, and creates a new directory containing symlinks
to the selected frame_data files.

Usage::

    python -m semgraph.scripts.create_subset \
        --source /data/office0/stages/frame_data \
        --dest /data/ablations/fps_pose_30/office0/stages/frame_data \
        --method fps_pose --n_frames 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from semgraph.sampling import get_frame_selector


# ---------------------------------------------------------------------------
# Pose loading (lightweight — avoids deserializing full FrameDataRecords)
# ---------------------------------------------------------------------------

def _load_poses(source: Path) -> dict[int, np.ndarray]:
    """Load frame_idx -> 4x4 pose from npz+json pairs in *source*."""
    poses: dict[int, np.ndarray] = {}
    for npz_path in sorted(source.glob("*.npz")):
        json_path = npz_path.with_suffix(".json")
        if not json_path.is_file():
            continue
        with open(json_path) as f:
            meta = json.load(f)
        frame_idx = int(meta["frame_idx"])
        pose = np.load(npz_path)["pose"]
        poses[frame_idx] = pose
    return poses


# ---------------------------------------------------------------------------
# Symlink helpers
# ---------------------------------------------------------------------------

def _symlink(src: Path, dst: Path, *, dry_run: bool) -> None:
    """Create an absolute symlink from *dst* pointing to *src*."""
    target = src.resolve()
    if dry_run:
        print(f"  [dry-run] {dst} -> {target}")
        return
    os.symlink(target, dst)


def _symlink_frame_files(
    source: Path,
    dest: Path,
    frame_indices: np.ndarray,
    *,
    dry_run: bool,
) -> None:
    """Symlink .npz and .json for each selected frame."""
    for idx in frame_indices:
        stem = f"{idx:06d}"
        for suffix in (".npz", ".json"):
            src = source / f"{stem}{suffix}"
            dst = dest / f"{stem}{suffix}"
            if src.exists():
                _symlink(src, dst, dry_run=dry_run)


def _symlink_crops(
    source: Path,
    dest: Path,
    frame_indices: np.ndarray,
    *,
    dry_run: bool,
) -> None:
    """Symlink crop images for each selected frame."""
    crops_src = source.parent / "crops"
    if not crops_src.is_dir():
        print(f"  [warn] crops directory not found: {crops_src}")
        return
    crops_dst = dest.parent / "crops"
    if not dry_run:
        crops_dst.mkdir(parents=True, exist_ok=True)
    for idx in frame_indices:
        pattern = f"{idx:06d}_*.jpg"
        for crop_file in sorted(crops_src.glob(pattern)):
            _symlink(crop_file, crops_dst / crop_file.name, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a symlinked frame_data subset via frame selection.",
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="Path to oracle frame_data/ directory")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Path to subset frame_data/ directory to create")
    parser.add_argument("--method", choices=["stride", "fps_pose"], required=True,
                        help="Frame selection method")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Target frame count")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride value (stride method only)")
    parser.add_argument("--position_weight", type=float, default=1.0,
                        help="Position weight for fps_pose (default: 1.0)")
    parser.add_argument("--direction_weight", type=float, default=0.5,
                        help="Direction weight for fps_pose (default: 0.5)")
    parser.add_argument("--symlink_crops", action="store_true",
                        help="Also symlink corresponding crop images")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print actions without creating anything")
    args = parser.parse_args()

    if not args.source.is_dir():
        print(f"Error: source directory does not exist: {args.source}", file=sys.stderr)
        sys.exit(1)

    # --- Load poses ---
    print(f"Loading poses from {args.source} ...")
    poses = _load_poses(args.source)
    total = len(poses)
    if total == 0:
        print("Error: no frame_data files found in source directory.", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {total} frames.")

    # --- Build selector kwargs ---
    selector = get_frame_selector(args.method)
    kwargs: dict = {}

    if args.method == "stride":
        if args.stride is not None:
            kwargs["stride"] = args.stride
        elif args.n_frames is not None:
            kwargs["stride"] = max(1, total // args.n_frames)
        # else: StrideSelector uses its default stride=10
    elif args.method == "fps_pose":
        kwargs["position_weight"] = args.position_weight
        kwargs["direction_weight"] = args.direction_weight

    # --- Run selection ---
    result = selector.select(poses, n_frames=args.n_frames, **kwargs)
    selected = result.frame_indices
    print(f"  Selected {len(selected)} / {total} frames via '{result.method}'.")

    # --- Create dest and symlinks ---
    if not args.dry_run:
        args.dest.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  [dry-run] would create {args.dest}")

    _symlink_frame_files(args.source, args.dest, selected, dry_run=args.dry_run)

    if args.symlink_crops:
        _symlink_crops(args.source, args.dest, selected, dry_run=args.dry_run)

    # --- Write provenance metadata ---
    provenance = {
        "method": result.method,
        "n_selected": int(len(selected)),
        "total_available": total,
        "selected_frame_indices": sorted(int(i) for i in selected),
        **result.metadata,
    }
    if args.method == "stride" and "stride" in kwargs:
        provenance["computed_stride"] = kwargs["stride"]
    if args.n_frames is not None:
        provenance["requested_n_frames"] = args.n_frames

    meta_path = args.dest / "_selection_metadata.json"
    if args.dry_run:
        print(f"  [dry-run] would write {meta_path}")
    else:
        with open(meta_path, "w") as f:
            json.dump(provenance, f, indent=2)

    # --- Summary ---
    print()
    print(f"  Method:    {result.method}")
    print(f"  Selected:  {len(selected)} / {total}")
    print(f"  Dest:      {args.dest}")
    if args.dry_run:
        print("  (dry run — no files created)")


if __name__ == "__main__":
    main()
