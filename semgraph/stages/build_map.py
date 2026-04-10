"""
Stage A3 — Incremental map construction.

Reads ``frame_data/*.npz`` (+ ``.json``), runs the matching/merging loop
to build a MapObjectList, writes ``map/oracle_map``.

**Must always run from frame 0** — the matching/merging loop is incremental
and frame N depends on accumulated state from frames 0 through N-1.

Standalone usage::

    python -m semgraph.stages.build_map <hydra overrides>
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from semgraph.slam.slam_classes import MapEdgeMapping, MapObjectList

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption merging
# ---------------------------------------------------------------------------

def merge_captions_into_detections(
    det_list: list,
    captions_data: dict | None,
    surviving_indices: np.ndarray | None,
) -> list:
    """Attach VLM captions to each detection using surviving_indices.

    ``surviving_indices[k]`` maps filtered detection k to raw detection index,
    which is used to look up the caption from captions_data.
    """
    if captions_data is None or not det_list:
        return det_list

    captions = captions_data.get("captions", [])
    edges = captions_data.get("edges", [])
    labels = captions_data.get("labels", [])

    for k, det in enumerate(det_list):
        raw_idx = k
        if surviving_indices is not None and k < len(surviving_indices):
            raw_idx = int(surviving_indices[k])

        if raw_idx < len(captions):
            det["captions"] = [captions[raw_idx]]
        else:
            logger.warning(
                "surviving_indices[%d]=%d >= len(captions)=%d, using empty caption",
                k, raw_idx, len(captions),
            )
            det["captions"] = [""]

        if raw_idx < len(labels):
            det["labels"] = labels[raw_idx]

    return det_list


# ---------------------------------------------------------------------------
# Map update (per-frame matching + merging)
# ---------------------------------------------------------------------------

def update_map(
    det_list: list,
    objects: MapObjectList,
    map_edges: MapEdgeMapping,
    cfg: Any,
    frame_ctx_or_metadata: Any,
    gobs: dict | None = None,
) -> tuple[MapObjectList, MapEdgeMapping]:
    """Run matching + merging for one frame's detections."""
    from semgraph.slam.mapping import (
        aggregate_similarities,
        compute_spatial_similarities,
        compute_visual_similarities,
        match_detections_to_objects,
        merge_obj_matches,
    )
    from semgraph.slam.utils import process_edges

    skip_matching = getattr(frame_ctx_or_metadata, "skip_matching", False)
    if isinstance(frame_ctx_or_metadata, dict):
        skip_matching = frame_ctx_or_metadata.get("skip_matching", False)
        frame_idx = frame_ctx_or_metadata.get("frame_idx", 0)
    else:
        frame_idx = getattr(frame_ctx_or_metadata, "frame_idx", 0)

    if skip_matching:
        objects.extend(det_list)
        match_indices = list(range(len(objects) - len(det_list), len(objects)))
        if gobs is not None:
            map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        return objects, map_edges

    if len(objects) == 0:
        objects.extend(det_list)
        return objects, map_edges

    spatial_sim = compute_spatial_similarities(
        spatial_sim_type=cfg["spatial_sim_type"],
        detection_list=det_list,
        objects=objects,
        downsample_voxel_size=cfg["downsample_voxel_size"],
    )
    visual_sim = compute_visual_similarities(det_list, objects)
    agg_sim = aggregate_similarities(
        match_method=cfg["match_method"],
        phys_bias=cfg["phys_bias"],
        spatial_sim=spatial_sim,
        visual_sim=visual_sim,
    )
    match_indices = match_detections_to_objects(
        agg_sim=agg_sim,
        detection_threshold=cfg["sim_threshold"],
        detection_list=det_list,
        objects=objects,
        iou_merge_kappa=cfg.get("iou_merge_kappa", 0.0),
    )
    objects = merge_obj_matches(
        detection_list=det_list,
        objects=objects,
        match_indices=match_indices,
        downsample_voxel_size=cfg["downsample_voxel_size"],
        dbscan_remove_noise=cfg["dbscan_remove_noise"],
        dbscan_eps=cfg["dbscan_eps"],
        dbscan_min_points=cfg["dbscan_min_points"],
        spatial_sim_type=cfg["spatial_sim_type"],
        device=cfg["device"],
    )

    seg_backend = cfg.get("segmentation_backend", "sam_auto")
    if seg_backend == "yolo_sam":
        from semgraph.utils.general_utils import ObjectClasses, cfg_to_dict
        vocab_cfg = cfg_to_dict(cfg)
        obj_classes = ObjectClasses(
            classes_file_path=vocab_cfg["classes_file"],
            bg_classes=vocab_cfg["bg_classes"],
            skip_bg=vocab_cfg["skip_bg"],
        )
        vocab = obj_classes.get_classes_arr()
        for obj in objects:
            most_common = Counter(obj["class_id"]).most_common(1)[0][0]
            if 0 <= most_common < len(vocab):
                name = vocab[most_common]
                if obj["class_name"] != name:
                    obj["class_name"] = name

    if gobs is not None:
        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)

    return objects, map_edges


# ---------------------------------------------------------------------------
# Periodic maintenance
# ---------------------------------------------------------------------------

def run_maintenance(
    objects: MapObjectList,
    map_edges: MapEdgeMapping,
    cfg: Any,
    frame_idx: int,
    is_final_frame: bool,
) -> tuple[MapObjectList, MapEdgeMapping]:
    """Run denoise / filter / merge maintenance as configured."""
    from semgraph.slam.utils import (
        denoise_objects,
        filter_objects,
        merge_objects,
        processing_needed,
    )
    from semgraph.utils.general_utils import measure_time

    # Edge pruning
    edges_to_delete = []
    for curr_map_edge in map_edges.edges_by_index.values():
        if (frame_idx - curr_map_edge.first_detected > 5) and curr_map_edge.num_detections < 2:
            edges_to_delete.append((curr_map_edge.obj1_idx, curr_map_edge.obj2_idx))
    for e in edges_to_delete:
        map_edges.delete_edge(e[0], e[1])

    if processing_needed(cfg["denoise_interval"], cfg["run_denoise_final_frame"], frame_idx, is_final_frame):
        objects = measure_time(denoise_objects)(
            downsample_voxel_size=cfg["downsample_voxel_size"],
            dbscan_remove_noise=cfg["dbscan_remove_noise"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
            spatial_sim_type=cfg["spatial_sim_type"],
            device=cfg["device"],
            objects=objects,
        )

    if processing_needed(cfg["filter_interval"], cfg["run_filter_final_frame"], frame_idx, is_final_frame):
        objects = filter_objects(
            obj_min_points=cfg["obj_min_points"],
            obj_min_detections=cfg["obj_min_detections"],
            objects=objects,
            map_edges=map_edges,
        )

    if processing_needed(cfg["merge_interval"], cfg["run_merge_final_frame"], frame_idx, is_final_frame):
        if cfg["make_edges"]:
            objects, map_edges = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=True,
                map_edges=map_edges,
            )
        else:
            objects = measure_time(merge_objects)(
                merge_overlap_thresh=cfg["merge_overlap_thresh"],
                merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                objects=objects,
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                do_edges=False,
                map_edges=None,
            )

    return objects, map_edges


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Standalone build_map stage — reads frame_data + captions, writes map."""
    from tqdm import tqdm
    from semgraph.stages.paths import stage_paths, SerializedDetection
    from semgraph.io import (
        list_frame_indices,
        load_frame_data,
        deserialize_detection,
        save_map,
    )
    from semgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    paths["map"].mkdir(parents=True, exist_ok=True)

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    frame_indices = list_frame_indices(paths["frame_data"])
    n_frames = len(frame_indices)
    print(f"[build_map] Processing {n_frames} frames (always from frame 0)")

    for loop_idx, frame_idx in enumerate(tqdm(frame_indices, desc="build_map")):
        frame_record = load_frame_data(paths["frame_data"], frame_idx)
        if frame_record is None:
            continue

        # Reconstruct SerializedDetection dicts from the FrameDataRecord
        det_list = []
        for i in range(frame_record.n_detections):
            dm = frame_record.det_meta[i]
            sd = SerializedDetection(
                pcd_points=frame_record.pcd_points_list[i],
                pcd_colors=frame_record.pcd_colors_list[i],
                bbox_corners=frame_record.bbox_corners[i] if i < len(frame_record.bbox_corners) else np.zeros((8, 3)),
                bbox_type=dm.bbox_type,
                class_name=dm.class_name,
                class_id=dm.class_id,
                inst_id=dm.inst_id,
                n_points=dm.n_points,
                crop_path=dm.crop_path,
                clip_ft=frame_record.clip_ft[i] if frame_record.clip_ft is not None and i < len(frame_record.clip_ft) else None,
                text_ft=frame_record.text_ft[i] if frame_record.text_ft is not None and i < len(frame_record.text_ft) else None,
                vlm_vit_ft=None,
                vlm_proj_ft=None,
            )
            det_list.append(deserialize_detection(sd, cfg.device))

        if det_list and len(det_list) > 0:
            metadata = {
                "frame_idx": frame_record.frame_idx,
                "skip_matching": frame_record.skip_matching,
            }
            objects, map_edges = update_map(det_list, objects, map_edges, cfg, metadata)

        is_final = loop_idx == n_frames - 1
        objects, map_edges = run_maintenance(objects, map_edges, cfg, frame_idx, is_final)

    save_map(paths["map"], objects, map_edges, cfg)
    print(f"[build_map] Done. {len(objects)} objects in map.")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
