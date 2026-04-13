# Running the Batch Pipeline

> **Last updated:** 2026-04-10

## Prerequisites

1. **ScanNet `.sens` files must be extracted** before running ScanNet scenes:

```bash
cd ~/cmu-grad/neuro-nav/semgraph/scripts/scannet_process
for SCENE in scene0046_00 scene0222_00 scene0389_00 scene0435_00; do
  uv run --project ~/cmu-grad/neuro-nav python reader.py \
    --filename ~/cmu-grad/neuro-data/ScanNet/scans/${SCENE}/${SCENE}.sens \
    --output_path ~/cmu-grad/neuro-data/ScanNet/scans/${SCENE} \
    --export_depth_images --export_color_images --export_poses --export_intrinsics
done
```

2. **Data layout** after extraction:

```
~/cmu-grad/neuro-data/
├── Replica/
│   ├── room0/          # Replica scenes (ready to use)
│   ├── room1/
│   ├── office2/
│   └── office3/
└── ScanNet/scans/
    ├── scene0046_00/   # ScanNet scenes (extracted from .sens)
    │   ├── color/      # *.jpg
    │   ├── depth/      # *.png (16-bit, shift 1000)
    │   ├── pose/       # 0.txt, 1.txt, ... (4x4 c2w matrices)
    │   └── intrinsic/  # intrinsic_color.txt
    └── ...
```

---

## Pipeline Modes

`run_vllm_batch.sh` supports two execution modes via the `PIPELINE_MODE` variable:

### Staged mode (recommended)

```bash
PIPELINE_MODE=staged ./shells/run_vllm_batch.sh
```

Runs the two-phase staged pipeline. Phase A (detection, embedding, map building, oracle finalization) produces an immutable geometric oracle scene. Phase B (re-embedding, captioning, assembly, evaluation) runs against that oracle with the specified encoder/VLM combination. See [STAGED_PIPELINE.md](STAGED_PIPELINE.md) for details.

Staged mode delegates to `shells/run_staged_pipeline.sh`, which calls each stage as an independent `python -m semgraph.stages.*` module.

### All-in-one mode (legacy)

```bash
PIPELINE_MODE=all-in-one ./shells/run_vllm_batch.sh
```

Runs the monolith at `semgraph/slam/vlm_run/batch_vlm_mapping_api.py`. Processes everything in a single frame-sequential loop within one Python process. Retained for backward compatibility and single-process debugging.

---

## Quick Start

### Option A: Run vLLM in a separate terminal (recommended)

This gives you full live logs from vLLM and lets you restart scenes without reloading the model.

```bash
# Terminal 1 — start the vLLM server
cd ~/cmu-grad/neuro-nav
VLLM_USE_V1=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  --port 8000 --gpu-memory-utilization 0.75 --max-model-len 3072 \
  --trust-remote-code --dtype auto
```

```bash
# Terminal 2 — run the staged pipeline
cd ~/cmu-grad/neuro-nav
PIPELINE_MODE=staged VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

The script checks `localhost:8000/health` before startup. If a server is already running, it uses it and does not manage its lifecycle.

### Option B: Let the script manage vLLM

```bash
cd ~/cmu-grad/neuro-nav
VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

If no server is detected, the script starts vLLM in the background, monitors its health between scenes, auto-restarts if it crashes, and cleans up on exit.

---

## Running the Staged Pipeline Directly

You can bypass `run_vllm_batch.sh` and call the staged pipeline directly:

```bash
# Phase A only (geometry, no VLM needed)
python -m semgraph.stages.detect dataset_root=... scene_id=room0
python -m semgraph.stages.embed dataset_root=... scene_id=room0
python -m semgraph.stages.build_map dataset_root=... scene_id=room0
python -m semgraph.stages.oracle_finalize dataset_root=... scene_id=room0

# Phase B (needs a running vLLM server for caption + semantic_assemble)
python -m semgraph.stages.embed embed.mode=re_embed embed.encoder_name=openai/clip-vit-large-patch14 ...
python -m semgraph.stages.caption caption.vlm_name=Qwen/Qwen3-VL-2B-Instruct ...
python -m semgraph.stages.semantic_assemble assemble.encoder=... assemble.vlm=... ...
python -m semgraph.stages.eval eval.encoder=... eval.vlm=... ...
```

Or use the shell orchestrator which handles both phases:

```bash
./shells/run_staged_pipeline.sh dataset_root=... scene_id=room0

# Evaluate with multiple encoders:
ENCODER="openai/clip-vit-large-patch14" VLM="Qwen/Qwen3-VL-2B-Instruct" \
  ./shells/run_staged_pipeline.sh dataset_root=... scene_id=room0
```

Phase A does not need to be re-run when sweeping different encoder/VLM combinations — only Phase B stages execute.

---

## Dataset Selection

```bash
# Only Replica scenes
SCANNET_SCENES="" ./shells/run_vllm_batch.sh

# Only ScanNet scenes
SCENES="" ./shells/run_vllm_batch.sh

# Specific scenes from each dataset
SCENES="room0 office2" SCANNET_SCENES="scene0046_00" ./shells/run_vllm_batch.sh
```

---

## GPU Memory & Model Size

The script runs vLLM at 75% GPU memory utilization by default. Adjust for your setup:

```bash
# Larger model, more VRAM for vLLM
VLM_MODEL="google/gemma-3-4b-it" GPU_MEM_UTIL=0.5 ./shells/run_vllm_batch.sh

# Tiny model, less VRAM
VLM_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct" GPU_MEM_UTIL=0.3 PROMPT_CONFIG="prompts_compact" ./shells/run_vllm_batch.sh
```

---

## All Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | HuggingFace model ID |
| `PIPELINE_MODE` | `all-in-one` | `staged` (recommended) or `all-in-one` (legacy) |
| `SCENES` | `room1 office2` | Replica scene list (empty to skip) |
| `SCANNET_SCENES` | *(empty)* | ScanNet scene list (empty to skip) |
| `GPU_MEM_UTIL` | `0.75` | vLLM GPU memory fraction |
| `MAX_MODEL_LEN` | `3072` | Max context length |
| `PROMPT_CONFIG` | `prompts_standard` | `prompts_standard` or `prompts_compact` |
| `EXTRACT_ENCODER` | `true` | Extract VLM vision encoder embeddings |
| `STRIDE` | `10` | Frame sampling stride |
| `VLLM_PORT` | `8000` | vLLM server port |
| `HEALTH_TIMEOUT` | `300` | Seconds to wait for vLLM startup |
| `FORCE_DET` | `true` | Re-run detections even if cached |
| `MAKE_EDGES` | `true` | Generate VLM edge relations |
| `EXP_SUFFIX` | `batch_api` | Output experiment folder name |
| `ENCODER` | `openai/clip-vit-large-patch14` | Phase B evaluation encoder (staged mode) |
| `VLM` | `Qwen/Qwen3-VL-2B-Instruct` | Phase B VLM for captioning (staged mode) |

---

## Output Structure

### Staged mode

All artifacts live under `{dataset_root}/{scene_id}/stages/`:

```
~/cmu-grad/neuro-data/Replica/room0/stages/
├── raw_detections/         # RawDetRecord per frame (.npz + .json)
├── frame_data/             # FrameDataRecord per frame (.npz + .json)
├── crops/                  # 1.5x projected crops (JPEG)
├── map/                    # oracle_map.pkl.gz (intermediate)
├── oracle/                 # oracle_scene.npz + .json (immutable)
├── variants/               # embed_{enc}.npz + .json per encoder
├── captions/               # {vlm}/captions.npz + .json per VLM
├── assembled/              # {enc}_{vlm}/scene_graph.json
└── eval/                   # {enc}_{vlm}/classification.json
```

### All-in-one mode (legacy)

```
~/cmu-grad/neuro-data/Replica/room0/exps/batch_api/
├── config_params.json
├── pcd_batch_api.pkl.gz
├── objects_batch_api.json
└── semantic_snapshot_batch_api.json
```

---

## Model Reference

See [VLLM_API.md](VLLM_API.md) for the full model support matrix, VRAM budgets, and prompt config recommendations.

| Model | Env Var Override | Prompt Config |
|-------|-----------------|---------------|
| Qwen3-VL 2B | `VLM_MODEL="Qwen/Qwen3-VL-2B-Instruct"` | `prompts_standard` |
| Qwen2.5-VL 3B AWQ | *(default)* | `prompts_standard` |
| InternVL3 2B | `VLM_MODEL="OpenGVLab/InternVL3-2B"` | `prompts_standard` |
| Gemma 3 4B | `VLM_MODEL="google/gemma-3-4b-it"` | `prompts_standard` |
| SmolVLM2 2B | `VLM_MODEL="HuggingFaceTB/SmolVLM2-2.2B-Instruct"` | `prompts_compact` |
| SmolVLM 500M | `VLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"` | `prompts_compact` |
| LLaVA-OV 0.5B | `VLM_MODEL="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"` | `prompts_standard` |

---

## Troubleshooting

**vLLM health check times out** — Bump the timeout:
```bash
HEALTH_TIMEOUT=600 ./shells/run_vllm_batch.sh
```

**External vLLM server died mid-run** — The batch script prints an error. Restart vLLM in your other terminal and re-run the batch script.

**Pre-download a model** — To download without running the pipeline:
```bash
uv run huggingface-cli download "Qwen/Qwen2.5-VL-3B-Instruct"
```

**CUDA OOM** — Reduce vLLM's share or disable encoder extraction:
```bash
GPU_MEM_UTIL=0.3 EXTRACT_ENCODER=false ./shells/run_vllm_batch.sh
```

**Port conflict** — Change the port:
```bash
VLLM_PORT=8001 ./shells/run_vllm_batch.sh
```

**ScanNet scene skipped** — The `.sens` file wasn't extracted. Re-run the extraction command from Prerequisites.
