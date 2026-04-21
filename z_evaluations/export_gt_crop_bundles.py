"""
Export per-scene ground-truth crop bundles for VLM-caption evaluation.

For each scene under ``--output_root``, reads
``<scene>/stages/oracle/oracle_scene.json`` and produces a zip containing:

  - ``crops/obj<obj_idx>_view<rank>_<orig_name>.jpg`` — one entry per view
  - ``labels.json`` — a manifest with per-object GT class names and the list
    of views referenced inside the zip

The manifest is grouped per object (``objects[i].class_name`` comes from the
oracle scene ``class_names`` array; ``objects[i].views`` carries the crop
arcnames, ``frame_idx``, ``n_points`` and ``n_visible`` for each view).

This lets you join against the VLM output at
``<scene>/stages/captions/<safe_vlm>/captions.json`` (keyed by the same
``obj_idx``) to compare predicted ``canonical_tag`` against GT class.

Usage::

    python z_evaluations/export_gt_crop_bundles.py \\
        --output_root /home/jrob/cmu-grad/neuro-experiments/ReplicaGroundTruth \\
        --scenes room0 room1 office2 office3 \\
        --out_dir outputs/gt_crop_bundles \\
        --top_k 0
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path


def _rank_views(views: list[dict]) -> list[dict]:
    """Mirror caption.py view ranking (n_points descending)."""
    return sorted(views, key=lambda v: v.get("n_points", 0), reverse=True)


def export_scene(
    scene: str,
    scene_root: Path,
    out_dir: Path,
    top_k: int,
) -> dict | None:
    """Build one zip + manifest for ``scene``. Returns stats dict or None."""
    oracle_json = scene_root / "stages" / "oracle" / "oracle_scene.json"
    if not oracle_json.is_file():
        print(f"[skip] {scene}: no oracle_scene.json at {oracle_json}", file=sys.stderr)
        return None

    data = json.loads(oracle_json.read_text())
    class_names: list[str] = data.get("class_names", [])
    per_view: list[list[dict]] = data.get("per_view_meta", [])
    if not class_names or not per_view:
        print(f"[skip] {scene}: empty class_names or per_view_meta", file=sys.stderr)
        return None

    zip_path = out_dir / f"{scene}_gt_crops.zip"
    manifest_path = out_dir / f"{scene}_gt_crops.labels.json"

    manifest: dict = {
        "scene": scene,
        "source_oracle_json": str(oracle_json),
        "top_k": top_k if top_k > 0 else None,
        "n_objects": len(class_names),
        "objects": [],
    }

    n_views_written = 0
    n_views_missing = 0
    seen_arcnames: set[str] = set()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for obj_idx, views in enumerate(per_view):
            cls = class_names[obj_idx] if obj_idx < len(class_names) else "unknown"
            ranked = _rank_views(views)
            selected = ranked if top_k <= 0 else ranked[:top_k]

            gt_inst = None
            if views:
                gt_inst = views[0].get("gt_instance_id")

            obj_rec: dict = {
                "object_idx": obj_idx,
                "class_name": cls,
                "gt_instance_id": gt_inst,
                "n_views_available": len(views),
                "n_views_included": 0,
                "views": [],
            }

            for rank, view in enumerate(selected):
                crop_path_s = view.get("crop_path", "")
                if not crop_path_s:
                    n_views_missing += 1
                    continue
                src = Path(crop_path_s)
                if not src.is_file():
                    n_views_missing += 1
                    continue

                arc = f"crops/obj{obj_idx:04d}_view{rank:02d}_{src.name}"
                if arc not in seen_arcnames:
                    zf.write(src, arcname=arc)
                    seen_arcnames.add(arc)
                    n_views_written += 1

                obj_rec["views"].append({
                    "crop": arc,
                    "frame_idx": view.get("frame_idx"),
                    "n_points": view.get("n_points"),
                    "n_visible": view.get("n_visible"),
                    "view_rank_by_n_points": rank,
                    "orig_crop_path": crop_path_s,
                })

            obj_rec["n_views_included"] = len(obj_rec["views"])
            manifest["objects"].append(obj_rec)

        manifest["n_views_written"] = n_views_written
        manifest["n_views_missing"] = n_views_missing
        zf.writestr("labels.json", json.dumps(manifest, indent=2))

    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(
        f"[ok] {scene}: {len(class_names)} objects, "
        f"{n_views_written} crops written "
        f"({n_views_missing} missing) -> {zip_path}"
    )
    return {
        "scene": scene,
        "zip": str(zip_path),
        "manifest": str(manifest_path),
        "n_objects": len(class_names),
        "n_views_written": n_views_written,
        "n_views_missing": n_views_missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("/home/jrob/cmu-grad/neuro-experiments/ReplicaGroundTruth"),
        help="Root containing <scene>/stages/oracle/oracle_scene.json",
    )
    ap.add_argument(
        "--scenes",
        nargs="+",
        default=["room0", "room1", "office2", "office3"],
        help="Scene ids to export.",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/gt_crop_bundles"),
        help="Destination for <scene>_gt_crops.zip and sidecar manifests.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=0,
        help=(
            "Keep only the top-K views per object ranked by n_points (matches "
            "caption.top_k). 0 or negative = include all views."
        ),
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for scene in args.scenes:
        scene_root = args.output_root / scene
        stats = export_scene(
            scene=scene,
            scene_root=scene_root,
            out_dir=args.out_dir,
            top_k=args.top_k,
        )
        if stats is not None:
            summary.append(stats)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": summary}, indent=2))
    print(f"[summary] wrote {summary_path} ({len(summary)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
