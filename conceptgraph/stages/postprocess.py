"""
Stage 4 — Post-processing: caption consolidation + final serialization + Rerun.

Reads ``map.pkl.gz`` from build_map.py, optionally consolidates captions
via VLM, then saves the final map artifacts.

Standalone usage::

    python -m conceptgraph.stages.postprocess <hydra overrides>
"""

from __future__ import annotations

from typing import Any

from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList


# ---------------------------------------------------------------------------
# Caption consolidation
# ---------------------------------------------------------------------------

def consolidate_map_captions(
    objects: MapObjectList,
    vlm_client: Any,
) -> None:
    """Consolidate accumulated per-detection captions into a single caption per object."""
    from conceptgraph.utils.vlms.vlm_api import consolidate_captions

    for obj in objects:
        obj_captions = obj.get("captions", [])[:20]
        if obj_captions:
            obj["consolidated_caption"] = consolidate_captions(vlm_client, obj_captions)
        else:
            obj["consolidated_caption"] = obj.get("class_name", "unknown object")


# ---------------------------------------------------------------------------
# Final serialization
# ---------------------------------------------------------------------------

def finalize(
    objects: MapObjectList,
    map_edges: MapEdgeMapping,
    vlm_client: Any | None,
    cfg: Any,
    obj_classes: Any | None = None,
) -> None:
    """Run caption consolidation and save all final artifacts."""
    from conceptgraph.utils.general_utils import (
        ObjectClasses,
        cfg_to_dict,
        save_obj_json,
        save_pointcloud,
    )
    from conceptgraph.stages.paths import _resolve_output_base, _build_exp_path

    if cfg.make_edges and vlm_client is not None:
        print(f"[postprocess] Consolidating captions...")
        consolidate_map_captions(objects, vlm_client)

    if obj_classes is None:
        det_cfg = cfg_to_dict(cfg)
        obj_classes = ObjectClasses(
            classes_file_path=det_cfg["classes_file"],
            bg_classes=det_cfg["bg_classes"],
            skip_bg=det_cfg["skip_bg"],
        )

    output_base = _resolve_output_base(cfg)
    exp_out_path = _build_exp_path(output_base, cfg.scene_id, cfg.exp_suffix, create=True)

    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges,
        )

    if cfg.get("save_semantic_snapshot", False):
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=None,
            create_symlink=False,
            edges=map_edges,
            include_geometry=False,
            artifact_prefix="semantic",
        )

    if cfg.save_json:
        save_obj_json(exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, objects=objects)
        from conceptgraph.utils.general_utils import save_edge_json
        save_edge_json(
            exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path,
            objects=objects, edges=map_edges,
        )


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Standalone postprocess stage — loads map, consolidates, saves."""
    from conceptgraph.stages.paths import stage_paths
    from conceptgraph.stages import io as stage_io
    from conceptgraph.stages.caption import init_vlm_client
    from conceptgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)

    result = stage_io.load_map(paths["map"])
    if result is None:
        raise FileNotFoundError(f"map.pkl.gz not found at {paths['map']}")
    objects, map_edges, _saved_cfg = result

    vlm_client = init_vlm_client(cfg)

    finalize(objects, map_edges, vlm_client, cfg)

    if vlm_client is not None:
        vlm_client.cleanup()

    print("[postprocess] Done.")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
