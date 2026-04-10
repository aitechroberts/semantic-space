## Mapping Workflow (based on `conceptgraph/slam/batch_test_local.py`)

run_slam_rgb.py → generate_gsa_results.py → cfslam_pipeline_batch.py

### Replica Scenes:
hotel0
room0
office3
apartment1

### ScanNet Scenes:
0046_00
0222_00
0389_00
0435_00

### System-Level Signal Flow

The mapping pipeline combines two complementary semantic sources:

- **CLIP embeddings** are always computed for every detection. They provide dense, comparable vectors
  for both images and text, powering the “geometry + semantics” matcher and the default retrieval
  path (vector similarity against stored object embeddings). This works even when the VLM features
  are disabled.

- The **VLM layer** (GPT-4o vision in this setup) adds human-readable metadata by describing each
  detection in plain language and inferring lightweight spatial relationships (“object 1 on top of
  object 2”). Those strings become node captions and edge labels, optionally consolidated into a
  single caption per object. They do not replace CLIP; they augment the graph for UI, filtering, or
  downstream LLM consumption.

When querying the graph you can:

1. **Run pure CLIP similarity** — (vector search) the default method concept-graphs uses for open-ended text queries (“find the lamp near the window”). No LLM needed; your query text is encoded with CLIP and compared against the stored CLIP embeddings.
2. **Filter by metadata/graph structure** — filtering by the VLM captions (“objects whose consolidated caption contains ‘trash bin’”) or by the VLM-inferred relationships (“things on top of desks”). That doesn’t require CLIP similarity if you’ve already captured the structural info you need; it’s just matching against those stored strings.
3. **Adopt a hybrid approach** — combine CLIP scores with the VLM labels/captions as context if you
   later call an LLM to answer richer questions. The current repo doesn’t force this; it simply
   provides the artifacts so downstream services can choose.

In short, VLM annotations enrich the graph with readable context, while CLIP embeddings remain the
core mechanism for semantic retrieval unless you explicitly design a different query path.

This document walks through every function defined in `batch_test_local.py` in the order they
participate in the workflow, detailing their required inputs, produced outputs, and the role each
plays in the overall mapping pipeline.

---

### 1. `_resolve_output_base(cfg: DictConfig) -> Path`

**Inputs (plain English)**
- The full Hydra configuration object (`cfg`) passed into the mapping run. Key fields include:
  - `dataset_root`, `dataset_config`, `scene_id`: tell the system which dataset to open, where the
    dataconfig YAML lives, and which Replica scene folder to traverse.
  - `start`, `end`, `stride`: define how many frame indices are enumerated (e.g., `0–10` with stride
    `1` means “process the first 10 consecutive frames”). Internally the dataset keeps only every
    `stride`‑th frame (`[start:end:stride]`), so changing stride simply thins out the frame list and
    all associated data (poses, depth, embeddings) before the mapping loop begins.
  - `exp_suffix`, `detections_exp_suffix`: determine the names of the `scene/exps/<suffix>` folders
    that will hold mapping outputs and detection artifacts, respectively.
  - `make_edges`, `force_detection`, `save_json`, `save_pcd`, `save_detections`: switches that tell
    the workflow whether to call OpenAI for VLM edges/captions, rerun YOLO/SAM even if caches exist,
    and which serialized artifacts (JSON, point clouds, detection caches) to emit.
  - `output_root`: optional override that instructs the run to redirect **all** experiment folders to
    a specific base directory (typically the `/mnt/local-output` bind mount during local tests). If
    omitted, artifacts default to `dataset_root`.
  - Environment-derived variables (e.g., `DATASET_ROOT`, `SCENE_ID`, `OUTPUT_ROOT`) are merged into
    `cfg` by Hydra before `_resolve_output_base` is called, so this helper sees the final values.
  - Environment-derived variables (e.g., `DATASET_ROOT`, `SCENE_ID`, `OUTPUT_ROOT`) are merged into
    `cfg` by Hydra before `_resolve_output_base` is called, so this helper sees the final values.

**Outputs**
- A `Path` object representing the root directory where experiment artifacts (e.g., `scene/exps/...`)
  should be written. This path will be used later when constructing the actual experiment folders.

**What it does / why it matters**
- Local smoke tests often want to separate input data (Replica scenes) from the directories where
  checkpoints and outputs are written. This helper inspects the configuration and environment to
  decide whether to redirect outputs to a dedicated mount (e.g., `/mnt/local-output`) instead of
  the dataset root. By centralizing this logic, the rest of the workflow can continue to assume
  “write under `<output_base>/<scene>/exps/...`” regardless of whether we are running locally or on
  AWS Batch.

---

### 2. `_build_exp_path(base_root: Path, scene_id: str, exp_suffix: str, create: bool = True) -> Path`

**Inputs**
- `base_root`: The filesystem root where all experiment results for this run should live (typically
  the value returned by `_resolve_output_base`).
- `scene_id`: The name of the Replica scene (e.g., `office0`) that scopes the experiment.
- `exp_suffix`: The human-readable label that distinguishes one experiment from another (for example,
  `batch_vlm_local` or `s_detections_batch`).
- `create`: Whether the function should create the directory hierarchy if it does not already exist.

**Outputs**
- A `Path` pointing to `<base_root>/<scene_id>/exps/<exp_suffix>`. Optionally ensures the directory
  exists, depending on `create`.

**What it does / why it matters**
- The mapping scripts expect every experiment—whether it is the main mapping run or a detection-only
  pass—to have its own folder under `scene/exps/<suffix>`. This helper builds those paths while
  honoring the caller’s chosen root, keeping the folder layout identical between local smoke tests
  and cloud runs. The mapping code uses it twice: once for the mapping experiment (`exp_suffix`) and
  once for the detections experiment (`detections_exp_suffix`).

---

### 3. `main(cfg: DictConfig)`

**Inputs**
- The Hydra configuration assembled from `batch_test_local.yaml` (defaults) plus any overrides
  supplied via CLI arguments or environment variables (e.g., `SCENE_ID`, `START`, `END`,
  `OUTPUT_ROOT`). In plain language, this contains:
  - Dataset location (`dataset_root`, `dataset_config`, `scene_id`, frame range, stride).
  - Experiment labels and behavioral flags (whether to run detections, save JSON/PCD, make edges,
    etc.).
  - Paths/mounts for checkpoints, FSx-style directories, and optional output redirection.
  - Low-level processing parameters (mask thresholds, merging/denoising intervals, etc.).

**Outputs**
- No explicit return value. Side effects include:
  - Writing raw detections, mapping artifacts (point clouds, JSON descriptions, optional videos),
    and logs into the experiment folders.
  - Logging metrics to Weights & Biases if enabled.
  - Printing progress/status to stdout.

**What it does / workflow role (step-by-step)**

1. **Run bookkeeping & logging setup**
   - Instantiates `MappingTracker` to accumulate detection/object counts.
   - Configures the optional Weights & Biases wrapper (`OptionalWandB`) based on `cfg.use_wandb`.

2. **Dataset initialization**
   - Calls `get_dataset` with the Replica config, frame range (`start`, `end`) and stride. This
     produces an iterable of RGB frames, depth maps, camera poses, and intrinsics that the rest of
     the pipeline consumes.

3. **Output directory preparation**
   - Uses `_resolve_output_base` and `_build_exp_path` to decide where mapping and detection
     artifacts will be written (respecting local output mounts or FSx paths).
   - Persists the resolved Hydra configs (`save_hydra_config`) so each experiment folder records the
     settings it was run with.

4. **Object-class metadata & detection cache setup**
   - Builds an `ObjectClasses` helper from the config so downstream code can translate class IDs to
     labels and apply background filtering.
   - Determines whether detections must be (re)computed (`check_run_detections`). If so, prepares
     the detection output directories (`get_det_out_path`, `get_vis_out_path`) and instantiates
     YOLO, SAM, and CLIP models on the GPU. If not, the run will load previously saved detections.

5. **Optional VLM client initialization**
   - If `cfg.make_edges` is `True`, creates an OpenAI client via `get_openai_client` so the run can
     request object relationships and captions per frame. Otherwise the VLM calls are skipped.

6. **Per-frame processing loop**  
   Each iteration performs the following steps in order:
   1. **Frame fetch**: Pulls RGB, depth, intrinsics, and pose tensors from the dataset iterator.
   2. **Detections**  
      - If `run_detections=True`:  
        a. Convert the RGB numpy array for OpenCV/SAM compatibility.  
        b. Run YOLO (`detection_model.predict`) to obtain bounding boxes, class IDs, and confidences.  
        c. Run SAM (`sam_predictor.predict`) on the YOLO boxes to get instance masks, trimming any
           mismatch between the number of boxes and masks.  
        d. Call `make_vlm_edges_and_captions` when `make_edges=True`: writes annotated frames,
           invokes OpenAI for object relationships/captions, and returns filtered labels/edges.  
        e. Call `compute_clip_features_batched` to extract CLIP embeddings for every detection.  
        f. Save detection visualizations and metadata via `save_detection_results` if requested.
      - Else (cached detections): load the serialized pickles for the current frame using
        `load_saved_detections`.
   3. **Geometric prep**  
      - Read the camera pose for this frame and convert it to numpy.  
      - Resize/filter detections (`resize_gobs`, `filter_gobs`) and subtract nested masks.  
      - Lift detections to 3D (`detections_to_obj_pcd_and_bbox`) using depth + intrinsics + pose,
        then denoise each detection point cloud (`init_process_pcd`) and compute bounding boxes.
   4. **Detection list creation**: Package each detection (point cloud, mask, metadata) into the
      unified format via `make_detection_list_from_pcd_and_gobs`.
   5. **Map update**  
      - If no objects exist yet, seed the map with the current detections.  
      - Otherwise:
        - Compute spatial similarities (`compute_spatial_similarities`) and visual similarities
          (`compute_visual_similarities`) between current detections and existing objects.  
        - Merge them with `aggregate_similarities`.  
        - Determine matches via `match_detections_to_objects`.  
        - Update objects with `merge_obj_matches`, incorporating new measurements.  
        - Re-label objects by most common class and update the edge graph via `process_edges`.  
        - Remove stale edges lacking support.
   6. **Maintenance passes**  
      - Conditionally run `denoise_objects`, `filter_objects`, and `merge_objects` depending on the
        configured intervals and whether this is the final frame (`processing_needed`).  
      - Optionally save per-frame object snapshots, renders, or periodic point clouds.  
      - Log frame-level metrics through `MappingTracker` and `OptionalWandB`.

7. **Post-loop consolidation**
   - **Caption consolidation**: For each object (when VLM edges/captions were enabled), crop the list
     of per-frame captions and call `consolidate_captions` to produce one final description stored on
     the object record.
   - **Final artifacts**:  
     - Write the point cloud(s) via `save_pointcloud`, optionally including edge information.  
     - Serialize the object list (`save_obj_json`) and the edge list (`save_edge_json`).  
     - If `save_objects_all_frames=True`, finalize the per-frame object metadata (`saved_obj_all_frames`).  
     - If detections were run and `save_video=True`, call `save_video_detections`.
   - **Cleanup**: Log completion metrics and close the WandB session with `owandb.finish()`.

**How it fits into the larger workflow**
- `main` is the entire mapping pipeline: it consumes raw Replica data, produces cleaned object/edge
  representations, and emits the artifacts that downstream evaluation or AWS Batch jobs need. The
  helper functions `_resolve_output_base` and `_build_exp_path` simply support `main` by ensuring
  the outputs go to the correct filesystem location for the current environment (local smoke test vs.
  FSx-backed Batch run).

---

## Imported Helper Functions & Utilities

The script relies heavily on utilities from `conceptgraph.utils` and `conceptgraph.slam`. Below is a
function-by-function reference for the helpers it calls, grouped by module.

### A. `conceptgraph.utils.general_utils`

| Function | Inputs (plain English) | Outputs | Role in workflow |
| --- | --- | --- | --- |
| `ObjectClasses(classes_file_path, bg_classes, skip_bg)` | Paths to class metadata, list of class names considered “background,” and a flag indicating whether to skip background objects | An object that can translate between class IDs/names, supply background lists, and provide class colors | Centralized knowledge about which classes exist and which should be filtered out when building the map |
| `get_det_out_path(exp_out_path, make_dir=True)` | Base experiment folder for detections and whether to create the directory | Path to `<exp_out_path>/detections` | Ensures serialized detections (PKL/JSON) have a consistent location |
| `get_vis_out_path(exp_out_path)` | Base experiment folder | Path to `<exp_out_path>/vis` (created if missing) | Where per-frame annotated images and depth renders are written |
| `handle_rerun_saving(use_rerun, save_rerun, exp_suffix, exp_out_path)` | Flags describing whether ReRun visualization should be persisted | None | Stub retained for compatibility; in this workflow it keeps ReRun disabled |
| `load_saved_detections(path)` | Filesystem path to a previously saved detection pickle | In-memory dictionary of detections, masks, features, etc. | Allows reusing detections across runs so you can skip YOLO/SAM if already computed |
| `load_saved_hydra_json_config(exp_out_path)` | Experiment folder containing `config_params*.json` | Python dictionary representing the saved config | Lets you compare a past run’s configuration with the current one |
| `make_vlm_edges_and_captions(image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, make_edges_flag, openai_client)` | The raw RGB image, detection results, object-class metadata, output directory, the frame’s path, a flag toggling edge creation, and an OpenAI client when edges are enabled | Filtered labels, list of VLM edges (if computed), rendered edge image, and per-object captions | Orchestrates the entire VLM workflow: filtering detections, annotating the frame, making OpenAI calls for relationships/captions, saving visualization assets |
| `measure_time(func)` | Any callable | Wrapped callable that logs execution time | Decorator used on expensive routines (YOLO load, detections-to-point-cloud conversion, etc.) so slow stages are visible in logs |
| `save_detection_results(path, payload)` | Target filename and detection payload | None | Persists detection results (masks, features, captions) for later reuse |
| `save_hydra_config(cfg, exp_out_path, is_detection_config=False)` | Hydra config object and destination experiment folder | None | Records the exact configuration used for this run inside the experiment directory |
| `save_obj_json(exp_suffix, exp_out_path, objects)` | Experiment label, output path, and the final list of objects | Writes `<exp_out_path>/obj_json_<exp_suffix>.json` | Provides a structured JSON representation of every object (poses, classes, captions) so downstream workflows don’t have to parse binary formats |
| `save_pointcloud(exp_suffix, exp_out_path, cfg, objects, obj_classes, latest_pcd_filepath, create_symlink, edges=None)` | Experiment metadata plus the final object list and optional edges | Writes `.pcd` files and (optionally) updates a “latest” symlink | Produces the mesh/point-cloud artifact consumed by visualization or evaluation tools |
| `should_exit_early(path)` | Path to the `early_exit.json` control file | Boolean indicating whether an early exit was requested | Enables the “flip a flag to skip to the final frame” mechanism during long runs |
| `cfg_to_dict(cfg)` | Hydra config object | Plain Python dictionary with only serializable types | Needed for logging configs to WandB and writing them to disk |
| `check_run_detections(force_detection, det_exp_path)` | Boolean toggle and the detections folder path | Boolean: run detections or reuse cache | Determines whether YOLO/SAM should run this time or whether cached detections suffice |

### B. `conceptgraph.utils.vlm`

| Function | Inputs | Outputs | Role |
| --- | --- | --- | --- |
| `get_openai_client()` | Reads `OPENAI_API_KEY` from env | OpenAI client object | Used once per run when VLM edges/captions are enabled |
| `consolidate_captions(openai_client, captions)` | OpenAI client plus up to 20 short captions collected for a single object | A single summarized caption string | Post-processing step that collapses per-frame captions into one human-readable description |

### C. `conceptgraph.utils.logging_metrics`

| Function/Class | Inputs | Outputs | Role |
| --- | --- | --- | --- |
| `MappingTracker()` | None | Object with counters (total detections/objects) | Simplifies logging of cumulative statistics to the console and WandB |

### D. `conceptgraph.slam.utils`

These helpers focus on processing detections, point clouds, and map maintenance.

| Function | Inputs | Outputs | Role |
| --- | --- | --- | --- |
| `process_cfg(cfg)` | Hydra config | Potentially modified config | Applies any derived settings or sanity checks before the run starts |
| `resize_gobs(raw_gobs, image_rgb)` | Raw detection dictionary and the RGB frame | Resized detection data | Adjusts detection masks/bboxes to the exact RGB resolution used downstream |
| `filter_gobs(gobs, image_rgb, skip_bg, BG_CLASSES, mask_area_threshold, max_bbox_area_ratio, mask_conf_threshold)` | Detection list, RGB frame, background settings, and thresholds | Filtered detection dictionary | Eliminates detections that are too small, low-confidence, or belong to background classes |
| `mask_subtract_contained(xyxy, mask)` *(imported earlier)* | Bounding boxes and masks | Cleaned mask array | Removes nested masks so overlapping detections don’t double-count pixels |
| `detections_to_obj_pcd_and_bbox(...)` | Depth map, masks, camera intrinsics, RGB frame, poses, and various thresholds | List of per-detection point clouds + bounding boxes | Lifts each 2D detection into 3D using depth + pose, producing the data structures used for matching and merging |
| `init_process_pcd(pcd, downsample_voxel_size, dbscan_remove_noise, dbscan_eps, dbscan_min_points)` | Raw point cloud for a detection | Filtered/normalized point cloud | Applies voxel downsampling and DBSCAN denoising to stabilize object clouds |
| `make_detection_list_from_pcd_and_gobs(obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx)` | Processed detections plus metadata | List of detection objects ready for matching | Wraps per-detection data into a structured format used by the matcher |
| `filter_objects(obj_min_points, obj_min_detections, objects, map_edges)` | Post-merge object list plus thresholds | Filtered object list | Removes small or weakly supported objects to keep the map clean |
| `get_bounding_box(spatial_sim_type, pcd)` | Spatial-similarity mode and the detection’s point cloud | 3D bounding box object | Provides geometric envelopes used during matching and visualization |
| `denoise_objects(...)`, `merge_objects(...)`, `process_edges(...)`, `processing_needed(...)` | Various combinations of objects, configuration thresholds, and frame counters | Updated objects/edges | These functions implement the maintenance chores executed every N frames (denoise, merge, edge cleanup) so the map doesn’t degrade over time |

### E. `conceptgraph.slam.mapping`

| Function | Inputs | Outputs | Role |
| --- | --- | --- | --- |
| `compute_spatial_similarities(spatial_sim_type, detection_list, objects, downsample_voxel_size)` | Current detections and existing objects | Matrix of spatial similarity scores | Quantifies how close each detection is to existing objects in 3D space |
| `compute_visual_similarities(detection_list, objects)` | Detections + existing objects | Visual similarity scores | Uses CLIP features (and other cues) to see which objects look alike |
| `aggregate_similarities(match_method, phys_bias, spatial_sim, visual_sim)` | Spatial + visual matrices, matching strategy | Combined similarity matrix | Applies the configured matching strategy (sum of scores or separate thresholds) |
| `match_detections_to_objects(agg_sim, detection_threshold)` | Aggregate similarity matrix and a threshold | Mapping from detection indices to object indices | Decides which detections correspond to existing objects vs. new ones |
| `merge_obj_matches(detection_list, objects, match_indices, ...)` | Detection list, existing object list, match indices, and processing thresholds | Updated object list | Integrates new measurements into the map, creating new objects when needed and updating existing ones otherwise |

### F. Other utilities

| Function/Class | Inputs | Outputs | Role |
| --- | --- | --- | --- |
| `OptionalWandB()` | None | Lightweight wrapper over wandb | Allows the script to call `owandb.log`, `owandb.finish` without crashing when wandb isn’t installed |
| `get_openai_client()` *(mentioned above)* | Reads API key from environment | OpenAI client | Provides the interface used by VLM helpers |

---

## Hydra Config Stack Overview

Calling `batch_test_local.py` (or the Docker image that references it) loads a layered Hydra config.
Understanding each layer helps you know where defaults come from and which file to edit when
changing behavior.

1. **`base.yaml`**  
   - Universal toggles: whether to use WandB, ReRun, etc.  
   - Minimal settings shared by all scripts.

2. **`base_mapping.yaml`**  
   - Core mapping parameters (mask thresholds, DBSCAN settings, merge intervals, whether to save
     videos/PCDs, etc.).  
   - Defines dataset placeholders such as `dataset_root`, `scene_id`, `detections_exp_suffix`.  
   - Introduces the `exit_early_file` control path.

3. **`replica.yaml`**  
   - Specifies Replica-specific dataset paths (`dataset_root`, `dataset_config`,
     `render_camera_path`).  
   - Sets default `scene_id` (e.g., `room0`) and ensures the camera intrinsics file is available.

4. **`sam.yaml`**  
   - Points to SAM checkpoint locations and variants so the detection stack knows which model
     weights to load.

5. **`classes.yaml`**  
   - Provides the class list (`scannet200_classes.txt`) and indicates which classes count as
     background.  
   - Supplies `bg_classes` / `skip_bg` defaults used by `ObjectClasses`.

6. **`logging_level.yaml`**  
   - Adjusts Hydra/console verbosity. Useful when you need extra logging during debugging.

7. **`batch_test_local.yaml`** *(local-only)*  
   - Overrides to make local smoke tests fast and deterministic:
     - `make_edges: false`, `end: 10`, `exp_suffix: batch_vlm_local`.  
     - `output_root` wired to the `OUTPUT_ROOT` environment variable when present, ensuring outputs
       flow into the bind-mounted directory (`/mnt/local-output`).  
     - Keeps detection saving enabled so subsequent runs can skip YOLO/SAM if desired.

Because Hydra composes these files in order, editing `batch_test_local.yaml` is the easiest way to
tailor the local workflow (e.g., enabling edges, changing `exp_suffix`, or redirecting outputs),
while shared behavior (mask thresholds, dataset metadata, etc.) lives in the base configs.

