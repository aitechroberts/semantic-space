"""
semgraph/scripts/visualize_map_open3d.py

Open3D visualizer for oracle_map.pkl.gz (preferred) or
oracle_scene.npz + variant.npz. Keyboard callbacks mirror
ConceptGraph's visualize_cfslam_results.py.

Usage:
    python -m semgraph.scripts.visualize_map_open3d \
        --map_path outputs/replica/room0/stages/map/oracle_map.pkl.gz

    # Or from finalized artifacts (with a Phase B variant for queries):
    python -m semgraph.scripts.visualize_map_open3d \
        --oracle_dir outputs/replica/room0/stages/oracle \
        --variant_path outputs/replica/room0/stages/variants/laion-bigG.npz
"""

from __future__ import annotations

import argparse
import copy
import queue
import sys
import threading
from collections import Counter
from pathlib import Path

import distinctipy
import matplotlib
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F


def _obj_class_id(obj: dict, fallback: int) -> int:
    """Return a single scalar class_id, majority-voting if it's a list.

    ConceptGraphs-style object dicts accumulate ``class_id`` as one entry
    per contributing view; our ``oracle_map`` carries that form directly.
    The ``oracle_scene`` branch already stores a scalar, so this helper is
    a no-op there.
    """
    cid = obj.get("class_id", None)
    if cid is None:
        return fallback
    if isinstance(cid, (list, tuple, np.ndarray)):
        if len(cid) == 0:
            return fallback
        return int(Counter(list(cid)).most_common(1)[0][0])
    return int(cid)


# ── loaders ─────────────────────────────────────────────────────────────

def _load_from_oracle_map(map_path: Path):
    """Returns (objects, edges, cfg, feat_matrix, feat_space_label)."""
    from semgraph.io import load_map

    res = load_map(map_path.parent)
    if res is None:
        raise FileNotFoundError(map_path)
    objects, edges, cfg = res

    feats = []
    for o in objects:
        ft = o.get("clip_ft")
        if ft is None:
            feats.append(None)
        elif hasattr(ft, "detach"):
            feats.append(ft.detach().cpu().float().numpy())
        else:
            feats.append(np.asarray(ft, dtype=np.float32))
    dim = next((f.shape[-1] for f in feats if f is not None), 0)
    feat_matrix = np.stack([
        f if f is not None else np.zeros(dim, dtype=np.float32) for f in feats
    ]) if feats and dim > 0 else None
    return objects, edges, cfg, feat_matrix, "oracle_map"


def _load_from_oracle_scene(oracle_dir: Path, variant_path: Path | None):
    from semgraph.io import load_oracle_scene, load_variant

    rec = load_oracle_scene(oracle_dir)
    if rec is None:
        raise FileNotFoundError(oracle_dir)

    objects = []
    for i, (pts, cols) in enumerate(zip(rec.obj_pcd_points_list, rec.obj_pcd_colors_list)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None and len(cols) == len(pts):
            pcd.colors = o3d.utility.Vector3dVector(cols)
        corners = rec.obj_bbox_corners[i] if i < len(rec.obj_bbox_corners) else None
        bbox = None
        if corners is not None and corners.size > 0:
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corners),
            )
        objects.append({
            "pcd": pcd,
            "bbox": bbox,
            "class_name": rec.class_names[i] if i < len(rec.class_names) else "object",
            "class_id": i,
            "parent_plane_id": rec.parent_plane_ids[i] if i < len(rec.parent_plane_ids) else None,
        })

    feat_matrix = None
    if variant_path is not None:
        slug = variant_path.stem
        vr = load_variant(variant_path.parent, slug)
        if vr is not None and vr.clip_ft_weighted_avg.size:
            feat_matrix = vr.clip_ft_weighted_avg.astype(np.float32)

    return objects, rec, None, feat_matrix, "oracle_scene"


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_path", type=str, default=None,
                    help="Path to oracle_map.pkl.gz")
    ap.add_argument("--oracle_dir", type=str, default=None,
                    help="Directory containing oracle_scene.npz (fallback)")
    ap.add_argument("--variant_path", type=str, default=None,
                    help="Phase B variant .npz providing per-object features")
    ap.add_argument("--encoder_type", type=str, default=None,
                    help="Override encoder_type for text queries")
    ap.add_argument("--encoder_name", type=str, default=None)
    ap.add_argument("--no_text", action="store_true",
                    help="Skip loading a text encoder (disables F query)")
    ap.add_argument("--voxel_down", type=float, default=0.02,
                    help="PCD voxel downsample for smoother interaction")
    args = ap.parse_args()

    if args.map_path:
        objects, edges, cfg, feats, source = _load_from_oracle_map(Path(args.map_path))
    elif args.oracle_dir:
        objects, edges, cfg, feats, source = _load_from_oracle_scene(
            Path(args.oracle_dir),
            Path(args.variant_path) if args.variant_path else None,
        )
    else:
        raise SystemExit("Provide --map_path OR --oracle_dir")

    print(f"[vis] loaded {len(objects)} objects from {source}")

    # Downsample PCDs for faster interactive rendering
    for o in objects:
        if o.get("pcd") is not None:
            o["pcd"] = o["pcd"].voxel_down_sample(args.voxel_down)

    # Per-class colors (distinctipy) and per-instance colormap (turbo)
    class_ids = [_obj_class_id(o, i) for i, o in enumerate(objects)]
    uniq = sorted(set(class_ids))
    palette = distinctipy.get_colors(max(len(uniq), 1), pastel_factor=0.5)
    class_colors = {c: palette[i] for i, c in enumerate(uniq)}
    cmap = matplotlib.colormaps.get_cmap("turbo")

    pcds = [copy.deepcopy(o["pcd"]) for o in objects]
    bboxes = [o["bbox"] for o in objects if o.get("bbox") is not None]

    # Scene graph geometry from MapEdgeMapping
    scene_graph_geoms = []
    if edges is not None and hasattr(edges, "edges_by_index") and edges.edges_by_index:
        centers = [
            np.asarray(p.points).mean(axis=0) if len(p.points) else np.zeros(3)
            for p in pcds
        ]
        for e in edges.edges_by_index.values():
            i, j = e.obj1_idx, e.obj2_idx
            if i >= len(centers) or j >= len(centers):
                continue
            pts = np.array([centers[i], centers[j]])
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector([[0, 1]]),
            )
            ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]])
            scene_graph_geoms.append(ls)
            # Center nodes
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            s.translate(centers[i])
            s.paint_uniform_color([0, 1, 0])
            scene_graph_geoms.append(s)

    # Lazily build text encoder only if we need queries
    encoder = None
    feat_tensor = None
    if feats is not None and not args.no_text:
        enc_type = args.encoder_type
        enc_name = args.encoder_name
        if (enc_type is None or enc_name is None) and cfg is not None:
            emb = cfg.get("embed", {}) if hasattr(cfg, "get") else getattr(cfg, "embed", {})
            enc_type = enc_type or (emb.get("encoder_type") if hasattr(emb, "get") else getattr(emb, "encoder_type", None))
            enc_name = enc_name or (emb.get("encoder_name") if hasattr(emb, "get") else getattr(emb, "encoder_name", None))
        if enc_type and enc_name:
            print(f"[vis] loading text encoder {enc_type}/{enc_name} ...")
            from semgraph.encoding import get_encoder
            encoder = get_encoder(enc_type, enc_name,
                                  device="cuda" if torch.cuda.is_available() else "cpu")
            feat_tensor = torch.from_numpy(feats).to(encoder.model.device if hasattr(encoder, "model") else "cpu")
            feat_tensor = F.normalize(feat_tensor.float(), dim=-1)
        else:
            print("[vis] no encoder info — F-query disabled")

    # ── viewer ──────────────────────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="semgraph map", width=1280, height=720)
    for g in pcds + bboxes:
        vis.add_geometry(g)

    state = {"graph_on": False}

    def color_by_class(_v):
        for pcd, cid in zip(pcds, class_ids):
            col = np.asarray(class_colors[cid])
            pcd.colors = o3d.utility.Vector3dVector(np.tile(col, (len(pcd.points), 1)))
            _v.update_geometry(pcd)

    def color_by_rgb(_v):
        for pcd, o in zip(pcds, objects):
            if o["pcd"].has_colors():
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(o["pcd"].colors))
                _v.update_geometry(pcd)

    def color_by_instance(_v):
        n = len(pcds) or 1
        inst = cmap(np.linspace(0, 1, n))[:, :3]
        for pcd, c in zip(pcds, inst):
            pcd.colors = o3d.utility.Vector3dVector(np.tile(c, (len(pcd.points), 1)))
            _v.update_geometry(pcd)

    def toggle_graph(_v):
        if not scene_graph_geoms:
            print("[vis] no edges"); return
        for g in scene_graph_geoms:
            (_v.add_geometry if not state["graph_on"] else _v.remove_geometry)(g, reset_bounding_box=False)
        state["graph_on"] = not state["graph_on"]

    # Shared state between the stdin reader thread (produces colors) and
    # the Open3D animation callback (applies them to geometry).  All
    # geometry mutations MUST happen on the main thread or Open3D will
    # silently drop updates / crash.
    result_q: "queue.Queue[tuple[str, np.ndarray, np.ndarray, np.ndarray]]" = queue.Queue()

    def _compute_query(q: str) -> None:
        if encoder is None or feat_tensor is None:
            print("[vis] query disabled (no per-object features or encoder)")
            return
        txt = encoder.encode_texts([q])
        if txt is None:
            print("[vis] this encoder has no text tower")
            return
        tq = F.normalize(torch.from_numpy(txt).to(feat_tensor.device).float(), dim=-1).squeeze(0)
        sims = (feat_tensor @ tq).cpu().numpy()
        lo, hi = float(sims.min()), float(sims.max())
        norm = (sims - lo) / max(hi - lo, 1e-9)
        cols = cmap(norm)[:, :3]
        top = np.argsort(-sims)[:5]
        result_q.put((q, sims, cols, top))

    def _stdin_reader():
        if not sys.stdin or not sys.stdin.isatty():
            return
        try:
            while True:
                try:
                    q = input("Query > ").strip()
                except EOFError:
                    return
                if not q:
                    continue
                try:
                    _compute_query(q)
                except Exception as e:  # noqa: BLE001
                    print(f"[vis] query error: {e}")
        except Exception:  # noqa: BLE001
            pass

    def query_by_clip(_v):
        """F-key: prompt user via a dialog-free stdin read on a worker."""
        if encoder is None or feat_tensor is None:
            print("[vis] query disabled (no per-object features or encoder)")
            return
        # Don't call input() from the GUI thread; defer to the background
        # stdin reader that's already running.
        print("[vis] type your query in the terminal window")

    def save_view(_v):
        p = _v.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("view.json", p)
        print("[vis] wrote view.json")

    def _apply_query_results(_v):
        """Animation callback — drains pending query results on the main thread."""
        drained = False
        try:
            while True:
                q, sims, cols, top = result_q.get_nowait()
                print(f"[vis] query '{q}' → top matches:")
                for rank, idx in enumerate(top):
                    name = objects[idx].get("class_name", "?")
                    print(f"  #{rank+1}  idx={idx:>3}  sim={sims[idx]:+.3f}  {name}")
                for pcd, c in zip(pcds, cols):
                    if len(pcd.points) == 0:
                        continue
                    pcd.colors = o3d.utility.Vector3dVector(
                        np.tile(c, (len(pcd.points), 1))
                    )
                    _v.update_geometry(pcd)
                drained = True
        except queue.Empty:
            pass
        return drained

    for key, cb in [
        ("C", color_by_class), ("R", color_by_rgb), ("I", color_by_instance),
        ("F", query_by_clip), ("G", toggle_graph), ("V", save_view),
    ]:
        vis.register_key_callback(ord(key), cb)

    vis.register_animation_callback(_apply_query_results)

    # Start the stdin reader only if we actually have a text encoder.
    if encoder is not None and feat_tensor is not None:
        threading.Thread(target=_stdin_reader, daemon=True).start()
        print("CLIP query: type any text in this terminal and press Enter "
              "(focus doesn't need to be on the viewer).")

    print("Keys: C=class  R=rgb  I=instance  F=query (stdin)  "
          "G=graph  V=save view")
    vis.run()


if __name__ == "__main__":
    main()