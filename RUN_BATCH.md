# Running the vLLM Batch Pipeline

## Prerequisites

1. **ScanNet `.sens` files must be extracted** before running ScanNet scenes. If you haven't done this yet:

```bash
cd ~/cmu-grad/neuro-nav/conceptgraph/scripts/scannet_process
for SCENE in scene0046_00 scene0222_00 scene0389_00 scene0435_00; do
  uv run --project ~/cmu-grad/neuro-nav python reader.py \
    --filename ~/cmu-grad/neuro-data/ScanNet/scans/${SCENE}/${SCENE}.sens \
    --output_path ~/cmu-grad/neuro-data/ScanNet/scans/${SCENE} \
    --export_depth_images --export_color_images --export_poses --export_intrinsics
done
```

Each scene takes a few minutes. Verify extraction worked:

```bash
ls ~/cmu-grad/neuro-data/ScanNet/scans/scene0046_00/{color,depth,pose,intrinsic}
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
    ├── scene0222_00/
    ├── scene0389_00/
    └── scene0435_00/
```

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
# Terminal 2 — run the batch pipeline (auto-detects the server)
cd ~/cmu-grad/neuro-nav
VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

The script checks `localhost:8000/health` before startup. If a server is already running, it prints "Found existing vLLM server" and skips its own startup. It will **not** kill or restart the external server on exit.

### Option B: Let the script manage vLLM (self-contained)

```bash
cd ~/cmu-grad/neuro-nav
VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

If no server is detected on the port, the script starts vLLM in the background, monitors its health between scenes (auto-restarting if it crashes), and cleans up on exit. The log file path is printed at startup so you can `tail -f` it from another terminal.

### Run with a different model

```bash
VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" ./shells/run_vllm_batch.sh
```

### Run with compact prompts (for smaller models)

```bash
VLM_MODEL="HuggingFaceTB/SmolVLM2-2.2B-Instruct" PROMPT_CONFIG="prompts_compact" ./shells/run_vllm_batch.sh
```

## Dataset Selection

### Only Replica scenes

```bash
SCANNET_SCENES="" ./shells/run_vllm_batch.sh
```

### Only ScanNet scenes

```bash
SCENES="" ./shells/run_vllm_batch.sh
```

### Specific scenes from each dataset

```bash
SCENES="room0 office2" SCANNET_SCENES="scene0046_00" ./shells/run_vllm_batch.sh
```

## GPU Memory & Model Size

The script runs vLLM at 40% GPU memory utilization by default. Adjust for larger models:

```bash
# Larger model, more VRAM for vLLM
VLM_MODEL="google/gemma-3-4b-it" GPU_MEM_UTIL=0.5 ./shells/run_vllm_batch.sh

# Tiny model, less VRAM
VLM_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct" GPU_MEM_UTIL=0.3 PROMPT_CONFIG="prompts_compact" ./shells/run_vllm_batch.sh
```

## Vision Encoder Extraction

To also extract VLM vision encoder embeddings alongside TinyCLIP (for research comparison):

```bash
EXTRACT_ENCODER=true ./shells/run_vllm_batch.sh
```

## All Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLM_MODEL` | `Qwen/Qwen3-VL-2B-Instruct` | HuggingFace model ID |
| `SCENES` | `room0 room1 office2 office3` | Replica scene list (empty to skip) |
| `SCANNET_SCENES` | `scene0046_00 scene0222_00 scene0389_00 scene0435_00` | ScanNet scene list (empty to skip) |
| `GPU_MEM_UTIL` | `0.4` | vLLM GPU memory fraction |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `PROMPT_CONFIG` | `prompts_standard` | `prompts_standard` or `prompts_compact` |
| `EXTRACT_ENCODER` | `false` | Extract VLM vision encoder embeddings |
| `STRIDE` | `10` | Frame sampling stride |
| `VLLM_PORT` | `8000` | vLLM server port |
| `HEALTH_TIMEOUT` | `300` | Seconds to wait for vLLM startup |
| `FORCE_DET` | `true` | Re-run detections even if cached |
| `MAKE_EDGES` | `true` | Generate VLM edge relations |
| `EXP_SUFFIX` | `batch_api` | Output experiment folder name |

## Output Structure

Results are saved under each dataset's directory:

```
# Replica
~/cmu-grad/neuro-data/Replica/room0/exps/batch_api/

# ScanNet
~/cmu-grad/neuro-data/ScanNet/scans/scene0046_00/exps/batch_api/
```

Each experiment folder contains:
- `config_params.json` -- run configuration
- `pcd_batch_api.pkl.gz` -- 3D scene graph (point clouds + objects)
- `objects_batch_api.json` -- object metadata and captions
- `semantic_snapshot_batch_api.json` -- per-frame semantic annotations

## Model Reference

See [VLLM_API.md](VLLM_API.md) for the full model support matrix (18 models), VRAM budgets, and prompt config recommendations.

| Model | Env Var Override | Prompt Config |
|-------|-----------------|---------------|
| Qwen3-VL 2B | *(default)* | `prompts_standard` |
| Qwen2.5-VL 3B | `VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"` | `prompts_standard` |
| InternVL3 2B | `VLM_MODEL="OpenGVLab/InternVL3-2B"` | `prompts_standard` |
| Gemma 3 4B | `VLM_MODEL="google/gemma-3-4b-it"` | `prompts_standard` |
| SmolVLM2 2B | `VLM_MODEL="HuggingFaceTB/SmolVLM2-2.2B-Instruct"` | `prompts_compact` |
| SmolVLM 500M | `VLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"` | `prompts_compact` |
| LLaVA-OV 0.5B | `VLM_MODEL="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"` | `prompts_standard` |

## Troubleshooting

**vLLM health check times out (self-managed mode)** -- The script pre-downloads the model before starting the server, but if the download was interrupted or the cache is corrupted, bump the timeout:
```bash
HEALTH_TIMEOUT=600 ./shells/run_vllm_batch.sh
```

**External vLLM server died mid-run** -- If you're running vLLM in a separate terminal and it crashes, the batch script will print "External vLLM server is no longer responding. Please check your vLLM terminal and restart it." Restart vLLM in your other terminal and re-run the batch script — it will pick up the new server.

**Pre-download a model manually** -- To download without running the pipeline:
```bash
uv run huggingface-cli download "Qwen/Qwen2.5-VL-3B-Instruct"
```

**CUDA OOM** -- Reduce vLLM's share or disable encoder extraction:
```bash
GPU_MEM_UTIL=0.3 EXTRACT_ENCODER=false ./shells/run_vllm_batch.sh
```

**Port conflict** -- If the script detects a server you don't want it to use, change the port:
```bash
VLLM_PORT=8001 ./shells/run_vllm_batch.sh
```

**ScanNet scene skipped** -- The `.sens` file wasn't extracted. Re-run the extraction command from Prerequisites above.
