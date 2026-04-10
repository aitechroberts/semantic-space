"""
Batch VLM mapping — all-in-one mode.

Thin wrapper that delegates to the stage modules (detect, caption,
build_map, postprocess) while running in a single process with no
disk I/O between stages.

MappingTracker and OptionalWandB remain here — stage modules do not
import or use them.
"""

import os
import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import trange
import hydra
from omegaconf import DictConfig

from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    cfg_to_dict,
    check_run_detections,
    get_det_out_path,
    load_saved_detections,
    save_detection_results,
    save_hydra_config,
    save_pointcloud,
    should_exit_early,
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import process_cfg, processing_needed
from conceptgraph.slam.geometry import get_geometry_backend

from conceptgraph.stages import detect as detect_stage
from conceptgraph.stages import caption as caption_stage
from conceptgraph.stages import build_map as build_map_stage
from conceptgraph.stages import postprocess as postprocess_stage
from conceptgraph.stages.paths import _resolve_output_base, _build_exp_path

torch.set_grad_enabled(False)


@hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
def main(cfg: DictConfig):
    tracker = MappingTracker()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", config=cfg_to_dict(cfg))

    cfg = process_cfg(cfg)

    pipeline_mode = cfg.get("pipeline_mode", "trajectory")
    backend = get_geometry_backend(pipeline_mode)
    geo_ctx = backend.load(cfg)
    total_iterations = backend.num_iterations(geo_ctx)

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    output_base = _resolve_output_base(cfg)
    exp_out_path = _build_exp_path(output_base, cfg.scene_id, cfg.exp_suffix, create=True)
    det_exp_path = _build_exp_path(output_base, cfg.scene_id, cfg.detections_exp_suffix, create=False)

    det_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=det_cfg["classes_file"],
        bg_classes=det_cfg["bg_classes"],
        skip_bg=det_cfg["skip_bg"],
    )

    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)

    seg_backend = cfg.get("segmentation_backend", "sam_auto")
    if seg_backend == "gt_instances":
        run_detections = False

    # Load detection models
    models = detect_stage.load_models(cfg)
    if seg_backend == "yolo_sam" and models.detection_model is not None:
        models.detection_model.set_classes(obj_classes.get_classes_arr())

    # VLM client for captioning
    vlm_client = caption_stage.init_vlm_client(cfg)

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(det_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0
    geo_iter = backend.get_iterator(geo_ctx)

    for loop_idx in trange(total_iterations):
        frame_ctx = next(geo_iter)
        frame_idx = frame_ctx.frame_idx
        tracker.curr_frame_idx = frame_idx
        counter += 1

        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True
        if exit_early_flag and loop_idx < total_iterations - 1:
            continue

        # --- Detect ---
        if frame_ctx.skip_segmentation or run_detections:
            raw_gobs, detection_list, surviving_idx = detect_stage.process_frame(
                frame_ctx, models, cfg, backend, obj_classes
            )
        else:
            color_path = frame_ctx.color_path
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx}"
                )
            detection_list = None
            surviving_idx = None

        if raw_gobs is None:
            continue

        if cfg.save_detections and not frame_ctx.skip_segmentation and run_detections:
            save_detection_results(det_exp_pkl_path / frame_ctx.color_path.stem, raw_gobs)

        tracker.increment_total_detections(len(raw_gobs.get("xyxy", [])))

        # --- Caption (inline, no disk) ---
        if vlm_client is not None and not frame_ctx.skip_segmentation:
            import cv2
            image = cv2.imread(str(frame_ctx.color_path))
            raw_gobs_for_caption = dict(raw_gobs)
            raw_gobs_for_caption["_image_rgb"] = image
            captions_data = caption_stage.caption_frame(raw_gobs_for_caption, vlm_client, cfg)
            raw_gobs["captions"] = captions_data.get("captions", raw_gobs.get("captions", []))
            raw_gobs["edges"] = captions_data.get("edges", raw_gobs.get("edges", []))
            raw_gobs["labels"] = captions_data.get("labels", raw_gobs.get("labels", []))

        # Re-run filter + lift if we got raw_gobs from cache (detection_list is None)
        if detection_list is None and raw_gobs is not None:
            from conceptgraph.slam.utils import filter_gobs, resize_gobs
            from conceptgraph.utils.ious import mask_subtract_contained
            from conceptgraph.slam.utils import (
                get_bounding_box, init_process_pcd, make_detection_list_from_pcd_and_gobs,
            )
            from conceptgraph.utils.general_utils import measure_time

            image_rgb = frame_ctx.image_rgb
            resized_gobs = resize_gobs(raw_gobs, image_rgb)
            filtered_gobs = filter_gobs(
                resized_gobs, image_rgb,
                skip_bg=cfg.skip_bg,
                BG_CLASSES=obj_classes.get_bg_classes_arr(),
                mask_area_threshold=cfg.mask_area_threshold,
                max_bbox_area_ratio=cfg.max_bbox_area_ratio,
                mask_conf_threshold=cfg.mask_conf_threshold,
            )
            if len(filtered_gobs["mask"]) == 0:
                continue
            filtered_gobs["mask"] = mask_subtract_contained(filtered_gobs["xyxy"], filtered_gobs["mask"])
            obj_pcds_and_bboxes = measure_time(backend.lift_to_3d)(
                filtered_gobs["mask"], frame_ctx, cfg,
            )
            for obj in obj_pcds_and_bboxes:
                if obj:
                    obj["pcd"] = init_process_pcd(
                        pcd=obj["pcd"],
                        downsample_voxel_size=cfg["downsample_voxel_size"],
                        dbscan_remove_noise=cfg["dbscan_remove_noise"],
                        dbscan_eps=cfg["dbscan_eps"],
                        dbscan_min_points=cfg["dbscan_min_points"],
                    )
                    obj["bbox"] = get_bounding_box(
                        spatial_sim_type=cfg["spatial_sim_type"], pcd=obj["pcd"],
                    )
            detection_list = make_detection_list_from_pcd_and_gobs(
                obj_pcds_and_bboxes, filtered_gobs, frame_ctx.color_path, obj_classes, frame_idx
            )
            surviving_idx = None

        if detection_list is None or len(detection_list) == 0:
            continue

        # --- Build map (inline, no disk) ---
        objects, map_edges = build_map_stage.update_map(
            detection_list, objects, map_edges, cfg, frame_ctx,
            gobs=raw_gobs,
        )

        is_final_frame = loop_idx == total_iterations - 1
        objects, map_edges = build_map_stage.run_maintenance(
            objects, map_edges, cfg, frame_idx, is_final_frame
        )

        if cfg.save_objects_all_frames:
            from conceptgraph.utils.general_utils import save_objects_for_frame
            save_objects_for_frame(
                obj_all_frames_out_path, frame_idx, objects,
                cfg.obj_min_detections, frame_ctx.pose, frame_ctx.color_path,
            )

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            save_pointcloud(
                exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path,
                cfg=cfg, objects=objects, obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath, create_symlink=True,
            )

        owandb.log({
            "frame_idx": frame_idx, "counter": counter,
            "exit_early_flag": exit_early_flag, "is_final_frame": is_final_frame,
        })
        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log({
            "total_objects": tracker.get_total_objects(),
            "objects_this_frame": len(objects),
            "total_detections": tracker.get_total_detections(),
            "detections_this_frame": len(detection_list),
        })

    # --- Postprocess ---
    postprocess_stage.finalize(objects, map_edges, vlm_client, cfg, obj_classes)

    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / "meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                "cfg": cfg,
                "class_names": obj_classes.get_classes_arr(),
                "class_colors": obj_classes.get_class_color_dict_by_index(),
            }, f)

    if vlm_client is not None:
        vlm_client.cleanup()
    if models.vlm_encoder is not None:
        models.vlm_encoder.cleanup()

    owandb.finish()


if __name__ == "__main__":
    main()
