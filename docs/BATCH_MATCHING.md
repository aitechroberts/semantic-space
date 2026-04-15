# Batch Matching Mode

> **Last updated:** 2026-04-13
>
> Reference for the batch matching mode in `build_map.py`.  For the
> pipeline overview, see [STAGED_PIPELINE.md](STAGED_PIPELINE.md).  For the
> detection stage, see [STAGE_DETECTION.md](STAGE_DETECTION.md).

---

## When to Use Batch vs Incremental

| Criterion | Incremental (default) | Batch |
|-----------|----------------------|-------|
| Frame source | Sequential trajectory (RGBD, high temporal overlap) | Sparse/DUSt3R or FPS-selected diverse viewpoints |
| Frame-to-frame overlap | High (adjacent frames see mostly the same objects) | Low (large spatial gaps between selected views) |
| Order dependence | Yes — frame N depends on accumulated state from 0..N-1 | No — result is the same regardless of frame ordering |
| Typical frame count | Hundreds to thousands (stride subsampled) | 10--200 (FPS-selected or sparse reconstruction) |

**Rule of thumb:** if frames are temporally adjacent and overlap heavily,
use incremental.  If frames were selected for viewpoint diversity (e.g.,
`fps_pose` selector) or come from a sparse reconstruction backend, use
batch.

---

## How It Works

### Incremental (existing, default)

For each frame in order:

1. Deserialize detections from `frame_data/`.
2. Match new detections against accumulated map objects using spatial
   similarity (3D bbox IoU), visual similarity (CLIP cosine), and the
   aggregated threshold (`sim_threshold`).
3. Merge matched detections into existing objects; create new objects for
   unmatched detections.
4. Run periodic maintenance (denoise, filter, merge-overlapping).

### Batch

1. **Load all detections** from every frame into a single flat list (D
   detections total).
2. **Compute the full D x D pairwise similarity matrix** using the same
   `compute_spatial_similarities`, `compute_visual_similarities`, and
   `aggregate_similarities` functions as the incremental path.
3. **Build an adjacency graph** where an edge exists between detections
   `i` and `j` if `agg_sim[i,j] > sim_threshold` (or, as a fallback,
   if their 3D bbox IoU exceeds `iou_merge_kappa`).
4. **Find connected components** via `scipy.sparse.csgraph.connected_components`.
   Each component is a cluster of detections that should belong to the
   same map object.
5. **Merge each cluster** into a single object using `merge_obj2_into_obj1`
   (the same function the incremental path uses).  Merge order within a
   cluster does not affect the final result.
6. **Run final maintenance** (denoise + filter + merge-overlapping), identical
   to the incremental path's final-frame cleanup.

---

## Why It Exists

The incremental loop assumes high frame-to-frame overlap for matching.
When two frames observe the same object from very different angles (common
with FPS-selected diverse viewpoints or sparse reconstruction), the
pairwise spatial IoU between their detections can be low, and CLIP cosine
similarity may also be weak.  The incremental loop creates separate map
objects for these observations.

Batch mode finds the **global connected components** of the similarity
graph.  Even if no single pair of viewpoints exceeds the threshold, the
transitive closure through intermediate viewpoints can connect them.  For
example, if viewpoints A and B each overlap with viewpoint C but not with
each other, batch mode merges all three into one object (A--C--B), whereas
incremental mode (processing in order A, B, C) might create two objects
depending on ordering.

---

## Performance

The batch algorithm is O(D^2) in total detections D, where D is the sum
of detections across all frames.

| Frames | Detections/frame | D | Matrix size | Approx. memory |
|--------|-----------------|---|-------------|----------------|
| 10 | 30 | 300 | 90K | < 1 MB |
| 50 | 30 | 1,500 | 2.25M | ~9 MB |
| 200 | 30 | 6,000 | 36M | ~144 MB |
| 500 | 30 | 15,000 | 225M | ~900 MB |

A warning is printed if D > 10,000.  For large D, use incremental mode
or reduce the frame count.

The IoU fallback (`iou_merge_kappa > 0`) is optimized to only compute 3D
bbox IoU for pairs whose aggregated similarity exceeds half the threshold
(`0.5 * sim_threshold`), avoiding the full O(D^2) sweep.

---

## Configuration

```yaml
# In batch_vlm_mapping_api.yaml or via Hydra override:
build_map:
  matching_mode: batch   # "incremental" (default) or "batch"
```

Via CLI override:

```bash
python -m semgraph.stages.build_map \
    build_map.matching_mode=batch \
    <other overrides>
```

All other config parameters (`sim_threshold`, `iou_merge_kappa`,
`spatial_sim_type`, `match_method`, denoise/filter/merge intervals, etc.)
are shared between both modes and behave identically.

---

## Interaction with the Min-Frames Experiment

The `run_min_frames_experiment.sh` script uses **incremental** mode for
all ablation subsets by default, since those subsets are stride-sampled
or FPS-sampled from a sequential trajectory and the incremental path
handles them correctly.

For future **sparse/DUSt3R work** where frames have no sequential
ordering, use `build_map.matching_mode=batch`.

The `compare_oracles.py` evaluation script compares oracle scenes
geometrically regardless of which matching mode produced them.
