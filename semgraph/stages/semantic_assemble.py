"""
Stage B3 — Semantic assembly: combine oracle geometry with Phase B features
and captions into the final HPSG JSON.

- LLM edge labeling: for each unlabeled MST edge, assign spatial relationship
- Scene type inference: LLM infers room type from object tag list
- Plane captions: template "This is a {plane_label} in the {scene_type}."
- Output: HPSG JSON with the explicit node schema

Standalone usage::

    python -m semgraph.stages.semantic_assemble <hydra overrides> \\
        assemble.encoder=... assemble.vlm=...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM edge labeling
# ---------------------------------------------------------------------------

EDGE_LABEL_PROMPT = """Given two objects in a 3D scene:
Object A: "{tag_a}" at position {center_a}, size {extent_a}
Object B: "{tag_b}" at position {center_b}, size {extent_b}

What is the spatial relationship between A and B?
Choose exactly one: on, supports, in, contains, next to, none
Answer with just the relationship word."""

VALID_RELATIONS = {"on", "supports", "in", "contains", "next to", "none"}


def _label_edges(
    mst_edges: list[tuple[int, int, float]],
    objects: Any,
    captions_data: dict,
    vlm_client: Any,
) -> list[dict]:
    """Label each MST edge with a spatial relationship via LLM."""
    labeled = []
    for src, tgt, iou_score in mst_edges:
        cap_a = captions_data.get(src, {})
        cap_b = captions_data.get(tgt, {})
        tag_a = cap_a.get("canonical_tag", "object")
        tag_b = cap_b.get("canonical_tag", "object")

        bbox_a = objects[src].get("bbox") if src < len(objects) else None
        bbox_b = objects[tgt].get("bbox") if tgt < len(objects) else None

        center_a = _bbox_center(bbox_a)
        center_b = _bbox_center(bbox_b)
        extent_a = _bbox_extent(bbox_a)
        extent_b = _bbox_extent(bbox_b)

        relation = "none"
        if vlm_client is not None:
            prompt = EDGE_LABEL_PROMPT.format(
                tag_a=tag_a, center_a=center_a, extent_a=extent_a,
                tag_b=tag_b, center_b=center_b, extent_b=extent_b,
            )
            try:
                resp = vlm_client.generate(prompt=prompt)
                if resp:
                    resp_clean = resp.strip().lower()
                    for valid in VALID_RELATIONS:
                        if valid in resp_clean:
                            relation = valid
                            break
            except Exception as exc:
                logger.debug("Edge labeling failed for (%d, %d): %s", src, tgt, exc)

        labeled.append({
            "source": src,
            "target": tgt,
            "relationship": relation,
            "iou_score": iou_score,
        })

    return labeled


# ---------------------------------------------------------------------------
# Scene type inference
# ---------------------------------------------------------------------------

SCENE_TYPE_PROMPT = """Given these objects in a room: {object_tags}
What type of room is this most likely? Choose one:
office, bedroom, living room, kitchen, bathroom, dining room, lab, classroom, hallway.
Answer with just the room type."""


def _infer_scene_type(captions_data: dict, vlm_client: Any) -> str:
    """Infer room type from the list of all canonical tags."""
    tags = [v.get("canonical_tag", "object") for v in captions_data.values() if v.get("canonical_tag")]
    if not tags or vlm_client is None:
        return "unknown"

    prompt = SCENE_TYPE_PROMPT.format(object_tags=", ".join(tags[:50]))
    try:
        resp = vlm_client.generate(prompt=prompt)
        if resp:
            return resp.strip().lower()
    except Exception as exc:
        logger.warning("Scene type inference failed: %s", exc)

    return "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _BBoxProxy:
    """Lightweight stand-in for o3d bounding boxes, reconstructed from 8 corners."""

    def __init__(self, corners: np.ndarray):
        self.min_bound = corners.min(axis=0)
        self.max_bound = corners.max(axis=0)


def _make_bbox_proxy(corners: np.ndarray) -> _BBoxProxy | None:
    if corners is None or len(corners) == 0 or np.all(corners == 0):
        return None
    return _BBoxProxy(corners)


def _bbox_center(bbox) -> list[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0]
    mn = np.asarray(bbox.min_bound)
    mx = np.asarray(bbox.max_bound)
    return ((mn + mx) / 2.0).tolist()


def _bbox_extent(bbox) -> list[float]:
    if bbox is None:
        return [0.0, 0.0, 0.0]
    mn = np.asarray(bbox.min_bound)
    mx = np.asarray(bbox.max_bound)
    return (mx - mn).tolist()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(
    objects: Any,
    planes: list[dict],
    mst_edges: list[tuple[int, int, float]],
    embed_variant: dict,
    caption_variant: dict,
    vlm_client: Any,
) -> dict:
    """Combine oracle geometry + encoder features + captions into HPSG."""

    captions_data = caption_variant.get("captions", {})
    # Ensure keys are ints
    captions_data = {int(k): v for k, v in captions_data.items()}

    obj_features = embed_variant.get("objects_features", {})
    obj_features = {int(k): v for k, v in obj_features.items()}

    # Label edges
    labeled_edges = _label_edges(mst_edges, objects, captions_data, vlm_client)

    # Scene type
    scene_type = _infer_scene_type(captions_data, vlm_client)

    # Plane captions
    for p in planes:
        p["caption"] = f"This is a {p.get('label', 'surface')} in the {scene_type}."

    # Build nodes
    nodes = []
    for idx, obj in enumerate(objects):
        cap = captions_data.get(idx, {})
        feat = obj_features.get(idx, {})

        bbox = obj.get("bbox")
        extent = _bbox_extent(bbox)
        center = _bbox_center(bbox)

        nodes.append({
            "id": obj.get("id", idx),
            "bbox_extent": extent,
            "bbox_center": center,
            "object_tag": cap.get("canonical_tag", obj.get("class_name", "unknown")),
            "caption": cap.get("summary", ""),
            "color": cap.get("color", ""),
            "material": cap.get("material", ""),
            "candidate_tags": cap.get("candidate_tags", []),
            "best_entropy": float(feat.get("best_entropy", 0.0)),
            "n_views": int(len(obj.get("per_view_records", []))),
            "parent_plane_id": obj.get("parent_plane_id"),
        })

    plane_records = []
    for p in planes:
        plane_records.append({
            "plane_id": p.get("plane_id"),
            "label": p.get("label", ""),
            "caption": p.get("caption", ""),
            "normal": [float(x) for x in p.get("normal", [0, 0, 0])],
            "offset": float(p.get("offset", 0.0)),
        })

    return {
        "scene_type": scene_type,
        "objects": nodes,
        "planes": plane_records,
        "edges": labeled_edges,
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Combine oracle + features + captions, produce HPSG JSON."""
    from semgraph.stages.paths import stage_paths
    from semgraph.io import (
        load_oracle_scene,
        load_variant,
        load_captions,
    )
    from semgraph.stages.caption import init_vlm_client
    from semgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)

    assemble_cfg = cfg.get("assemble", {}) if hasattr(cfg, "get") else {}
    encoder_name = assemble_cfg.get("encoder", "laion_CLIP-ViT-bigG-14-laion2B-39B-b160k")
    vlm_name = assemble_cfg.get("vlm", "Qwen_Qwen3-VL-2B-Instruct")

    oracle = load_oracle_scene(paths["oracle"])
    if oracle is None:
        raise FileNotFoundError(f"oracle_scene not found at {paths['oracle']}")

    planes = oracle.planes
    mst_edges = oracle.mst_edges

    safe_enc = encoder_name.replace("/", "_")
    safe_vlm = vlm_name.replace("/", "_")

    embed_variant = load_variant(paths["variants"], f"embed_{safe_enc}")
    embed_dict: dict[str, Any] = {"objects_features": {}}
    if embed_variant is not None:
        for i in range(len(embed_variant.best_entropy)):
            embed_dict["objects_features"][i] = {
                "best_entropy": embed_variant.best_entropy[i],
            }

    captions_dir = paths["captions"] / safe_vlm
    captions_record = load_captions(captions_dir)
    captions_dict: dict[int, dict] = {}
    if captions_record is not None:
        captions_dict = captions_record.entries

    # Build a lightweight object-list proxy from oracle arrays for the assemble function
    objects_proxy = []
    for obj_idx in range(len(oracle.class_names)):
        bbox_corners = oracle.obj_bbox_corners[obj_idx] if obj_idx < len(oracle.obj_bbox_corners) else np.zeros((8, 3))
        pv_meta = oracle.per_view_meta[obj_idx] if obj_idx < len(oracle.per_view_meta) else []
        objects_proxy.append({
            "id": obj_idx,
            "class_name": oracle.class_names[obj_idx],
            "parent_plane_id": oracle.parent_plane_ids[obj_idx] if obj_idx < len(oracle.parent_plane_ids) else None,
            "per_view_records": [{"n_points": pm.n_points} for pm in pv_meta],
            "bbox": _make_bbox_proxy(bbox_corners),
        })

    vlm_client = init_vlm_client(cfg)

    print(f"[semantic_assemble] Assembling: encoder={encoder_name}, vlm={vlm_name}")
    caption_variant_compat = {"captions": captions_dict}
    scene_graph = assemble(objects_proxy, planes, mst_edges, embed_dict, caption_variant_compat, vlm_client)

    # Save HPSG JSON
    assembled_dir = paths["assembled"] / f"{safe_enc}_{safe_vlm}"
    assembled_dir.mkdir(parents=True, exist_ok=True)
    output_json = assembled_dir / "scene_graph.json"
    with open(output_json, "w") as f:
        json.dump(scene_graph, f, indent=2)

    if vlm_client is not None:
        try:
            vlm_client.cleanup()
        except Exception:
            pass

    print(f"[semantic_assemble] Done. Scene graph saved to {output_json}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
