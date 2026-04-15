"""
Stage A3 — Map construction (incremental or batch).

Reads ``frame_data/*.npz`` (+ ``.json``), runs matching/merging to build a
MapObjectList, writes ``map/oracle_map``.

Two matching modes (``build_map.matching_mode``):

* **incremental** (default) — frame-by-frame matching against accumulated
  objects.  Order-dependent; designed for sequential trajectory data with
  high temporal adjacency.
* **batch** — loads all detections, computes a full D x D pairwise
  similarity matrix, clusters via connected components, then merges each
  cluster.  Order-invariant; designed for sparse/DUSt3R mode or
  FPS-selected diverse viewpoints.

Standalone usage::

    python -m semgraph.stages.build_map <hydra overrides>
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np

from semgraph.slam.slam_classes import DetectionList, MapEdgeMapping, MapObjectList

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
    if seg_backend.startswith("detect_"):
        # Majority-vote relabeling for all detector-first backends.
        # NOTE: This assumes class_id values index into the global
        # ObjectClasses vocabulary, which is true for vocab-driven detectors
        # (YOLOE, YOLO-World, GroundingDINO) but not for Florence-2 whose
        # class_ids index per-frame ad-hoc vocabularies.  A full fix would
        # require voting on label strings rather than integer IDs.
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
        if cfg["obj_min_points"] > 0 or cfg["obj_min_detections"] > 1:
            pre_count = len(objects)
            objects = filter_objects(
                obj_min_points=cfg["obj_min_points"],
                obj_min_detections=cfg["obj_min_detections"],
                objects=objects,
                map_edges=map_edges,
            )
            if len(objects) < pre_count:
                print(f"[build_map] Pre-merge filter: {pre_count} -> {len(objects)} objects")

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
# Detection augmentation for merge compatibility
# ---------------------------------------------------------------------------

def _prepare_detection_for_merge(det: dict, frame_idx: int, det_idx: int) -> dict:
    """Add keys that ``merge_obj2_into_obj1`` expects but ``deserialize_detection`` omits."""
    import uuid

    det.setdefault("id", str(uuid.uuid4()))
    det.setdefault("image_idx", [frame_idx])
    det.setdefault("mask_idx", [det_idx])
    det.setdefault("color_path", [""])
    det.setdefault("mask", [])
    det.setdefault("xyxy", [])
    det.setdefault("conf", [])
    det.setdefault("contain_number", [0])
    det.setdefault("captions", [""])
    det.setdefault("num_detections", 1)
    det.setdefault("num_obj_in_class", 1)
    det.setdefault("is_background", False)
    det.setdefault("new_counter", 0)
    det.setdefault("curr_obj_num", det_idx)
    det.setdefault("inst_color", None)
    det.pop("inst_id", None)
    return det


# ---------------------------------------------------------------------------
# Shared deserialization helper
# ---------------------------------------------------------------------------

def _load_detections_from_record(frame_record, cfg, SerializedDetection, deserialize_detection):
    """Deserialize all detections from a single FrameDataRecord."""
    det_list = DetectionList()
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
    return det_list


# ---------------------------------------------------------------------------
# Incremental matching (original algorithm, extracted from main_standalone)
# ---------------------------------------------------------------------------

def build_map_incremental(frame_indices, paths, cfg):
    """Frame-by-frame incremental matching loop.

    This is the original build_map algorithm: each frame's detections are
    matched against accumulated objects, then maintenance is run
    periodically.  Order-dependent.
    """
    from tqdm import tqdm
    from semgraph.stages.paths import SerializedDetection
    from semgraph.io import load_frame_data, deserialize_detection

    objects = MapObjectList()
    map_edges = MapEdgeMapping(objects)
    n_frames = len(frame_indices)

    for loop_idx, frame_idx in enumerate(tqdm(frame_indices, desc="build_map")):
        frame_record = load_frame_data(paths["frame_data"], frame_idx)
        if frame_record is None:
            continue

        det_list = _load_detections_from_record(
            frame_record, cfg, SerializedDetection, deserialize_detection,
        )
        for det_idx, det in enumerate(det_list):
            _prepare_detection_for_merge(det, frame_record.frame_idx, det_idx)

        if det_list and len(det_list) > 0:
            metadata = {
                "frame_idx": frame_record.frame_idx,
                "skip_matching": frame_record.skip_matching,
            }
            objects, map_edges = update_map(det_list, objects, map_edges, cfg, metadata)

        is_final = loop_idx == n_frames - 1
        objects, map_edges = run_maintenance(objects, map_edges, cfg, frame_idx, is_final)

    return objects, map_edges


# ---------------------------------------------------------------------------
# Batch matching (all-pairs connected-components algorithm)
# ---------------------------------------------------------------------------

def build_map_batch(frame_indices, paths, cfg):
    """Order-invariant batch matching via pairwise similarity and clustering.

    All detections are loaded at once, a full D x D similarity matrix is
    computed, connected components define clusters, and each cluster is
    merged into a single map object using the same ``merge_obj2_into_obj1``
    logic as the incremental path.
    """
    from tqdm import tqdm
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    from semgraph.stages.paths import SerializedDetection
    from semgraph.io import load_frame_data, deserialize_detection
    from semgraph.slam.mapping import (
        aggregate_similarities,
        compute_spatial_similarities,
        compute_visual_similarities,
        compute_3d_bbox_iou,
    )
    from semgraph.slam.utils import merge_obj2_into_obj1

    # -- Step 1: Load all detections into a flat MapObjectList --------------
    all_dets = MapObjectList()
    global_det_idx = 0

    for frame_idx in tqdm(frame_indices, desc="build_map(batch) load"):
        frame_record = load_frame_data(paths["frame_data"], frame_idx)
        if frame_record is None:
            continue
        det_list = _load_detections_from_record(
            frame_record, cfg, SerializedDetection, deserialize_detection,
        )
        for det in det_list:
            _prepare_detection_for_merge(det, frame_idx, global_det_idx)
            all_dets.append(det)
            global_det_idx += 1

    D = len(all_dets)
    if D == 0:
        objects = MapObjectList()
        map_edges = MapEdgeMapping(objects)
        return objects, map_edges

    if D > 10_000:
        logger.warning(
            "Batch mode with %d detections — expect high memory usage "
            "and slow similarity computation (D^2 = %s).",
            D, f"{D * D:,}",
        )

    print(f"[build_map] Batch mode: {D} total detections from {len(frame_indices)} frames")

    # -- Step 2: Full D x D pairwise similarity -----------------------------
    spatial_sim = compute_spatial_similarities(
        spatial_sim_type=cfg["spatial_sim_type"],
        detection_list=all_dets,
        objects=all_dets,
        downsample_voxel_size=cfg["downsample_voxel_size"],
    )
    visual_sim = compute_visual_similarities(all_dets, all_dets)
    agg_sim = aggregate_similarities(
        match_method=cfg["match_method"],
        phys_bias=cfg["phys_bias"],
        spatial_sim=spatial_sim,
        visual_sim=visual_sim,
    )

    # -- Step 3: Connected-components clustering ----------------------------
    sim_threshold = cfg["sim_threshold"]
    adj = (agg_sim > sim_threshold).cpu().numpy()
    np.fill_diagonal(adj, False)

    # IoU fallback: only for near-threshold pairs to avoid O(D^2) bbox calls
    iou_merge_kappa = cfg.get("iou_merge_kappa", 0.0)
    if iou_merge_kappa > 0:
        near = (agg_sim.cpu().numpy() > 0.5 * sim_threshold) & ~adj
        np.fill_diagonal(near, False)
        rows, cols = np.where(np.triu(near))
        for i, j in zip(rows, cols):
            iou = compute_3d_bbox_iou(all_dets[i]["bbox"], all_dets[j]["bbox"])
            if iou > iou_merge_kappa:
                adj[i, j] = adj[j, i] = True

    n_components, labels = connected_components(
        sp.csr_matrix(adj.astype(np.bool_)), directed=False,
    )
    print(f"[build_map] {n_components} clusters from {D} detections")

    # -- Step 4: Merge each cluster into a single map object ----------------
    objects = MapObjectList()
    map_edges = MapEdgeMapping(objects)

    for cluster_id in range(n_components):
        members = np.where(labels == cluster_id)[0]
        seed = all_dets[int(members[0])]
        for m_idx in members[1:]:
            seed = merge_obj2_into_obj1(
                obj1=seed,
                obj2=all_dets[int(m_idx)],
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                run_dbscan=False,
            )
        objects.append(seed)

    # -- Step 5: Final post-processing (denoise + filter + merge) -----------
    if len(frame_indices) > 0:
        objects, map_edges = run_maintenance(
            objects, map_edges, cfg, frame_indices[-1], is_final_frame=True,
        )

    return objects, map_edges


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Standalone build_map stage — reads frame_data, writes map."""
    from semgraph.stages.paths import stage_paths
    from semgraph.io import list_frame_indices, save_map
    from semgraph.slam.utils import process_cfg

    cfg = process_cfg(cfg)
    paths = stage_paths(cfg)
    paths["map"].mkdir(parents=True, exist_ok=True)

    frame_indices = list_frame_indices(paths["frame_data"])
    print(f"[build_map] Processing {len(frame_indices)} frames")

    seg_backend = cfg.get("segmentation_backend", "")
    if seg_backend in ("sam_auto", "sam3_auto"):
        n_frames = len(frame_indices)
        preset_name = cfg.get("auto_filter_preset", None)
        if not preset_name:
            preset_name = "dense" if n_frames > 100 else "moderate" if n_frames > 30 else "sparse"
        presets = cfg.get("auto_filter_presets", {})
        if preset_name in presets:
            preset = presets[preset_name]
            cfg["obj_min_points"] = preset["obj_min_points"]
            cfg["obj_min_detections"] = preset["obj_min_detections"]
            print(f"[build_map] Auto-filter preset: {preset_name} "
                  f"(min_det={cfg['obj_min_detections']}, min_pts={cfg['obj_min_points']})")

    matching_mode = cfg.get("build_map", {}).get("matching_mode", "incremental")
    if matching_mode == "batch":
        objects, map_edges = build_map_batch(frame_indices, paths, cfg)
    else:
        objects, map_edges = build_map_incremental(frame_indices, paths, cfg)

    save_map(paths["map"], objects, map_edges, cfg)
    print(f"[build_map] Done. {len(objects)} objects in map.")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
