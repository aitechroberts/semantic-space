# Ablation Subset Crop Path Dependency

When running the minimum-frames ablation experiment, `create_subset.py`
symlinks `frame_data/*.npz` and `*.json` files from the oracle stride=1
directory into per-subset experiment directories. This makes each subset
appear self-contained, but there is a hidden dependency on the oracle
crops directory.

## The issue

`detect.py` stores each detection's crop image path as an **absolute path**
in the `frame_data` JSON metadata (via `_DetectionMeta.crop_path`). For
example:

```
/home/jrob/cmu-grad/neuro-experiments/min-frames/office0/stages/crops/000042_003.jpg
```

When `create_subset.py` symlinks the JSON file, the `crop_path` values
inside it still point to the **original oracle crops directory** — not
anywhere inside the subset's directory tree.

## When this matters

- **Geometric-only ablation (build_map + oracle_finalize + compare_oracles):**
  No impact. These stages read point clouds and bounding boxes from the npz
  arrays and never open crop images.

- **Running embed.py on a subset:** `embed.py` opens each crop image to
  encode it with CLIP. It reads `crop_path` from the JSON and follows the
  absolute path to the original oracle crops directory. This works
  transparently — as long as the oracle crops still exist at that path.

## What can go wrong

If you move, rename, or delete the oracle crops directory after creating
subsets, `embed.py` on any subset will fail with file-not-found errors even
though the subset's `frame_data/` directory looks intact.

## Recommendations

1. **Do not delete the oracle run's crops directory** until all subset
   experiments (including any future semantic evals) are complete.

2. If you need portable subsets (e.g. copying to another machine), either:
   - Use `create_subset.py --symlink_crops` to also symlink crop files into
     the subset directory, then `cp -L` (dereference symlinks) when copying.
   - Or re-run `detect.py` on the subset frames directly (expensive, defeats
     the purpose of symlinking).

3. If the crop paths become stale, you can patch them with a one-liner that
   rewrites the `crop_path` values in each JSON to point to the new location.
