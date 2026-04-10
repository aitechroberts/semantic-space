# Scene Graph Ablation Evaluation Suite

Complete evaluation pipeline for comparing VLM and CLIP configurations for 3D scene understanding.

**Designed for 16GB GPU (RTX 4090 Laptop)** - Models are loaded sequentially with GPU memory cleanup between runs.

## Overview

This suite evaluates **3 complete pipelines**, each with its own VLM + CLIP pairing:

| Config Folder | VLM Model | CLIP Model |
|---------------|-----------|------------|
| `oracle/` | GPT-4o-mini | MobileCLIP2-S3 |
| `qwen/` | Qwen3-VL-2B-Instruct | TinyCLIP-ViT-8M |
| `paligemma/` | PaliGemma2-3b-mix-224 | PE-Core-T-16-384 |

The embeddings are already stored in each config's `pkl.gz` files from the original mapping run.

## Evaluations Performed

1. **Scene Graph Quality** (Ablation Comparison)
   - CLIP embedding similarity (Chamfer-style semantic recall/precision)
   - VLM caption quality (CIDEr, SPICE metrics)
   - Graph triplet quality (node-edge-node evaluation)

2. **VQA Performance** (Space3D-Bench)
   - Top-1 and Top-5 accuracy for retrieval
   - Direct QA accuracy for VLM models

3. **Complex Queries** (Affordance/Negation)
   - 40 custom queries (10 per scene, 5 affordance + 5 negation each)
   - Affordance reasoning: "Something I can use for X"
   - Negation reasoning: "Something that is NOT X"
   - Breakdown by query type

4. **YOLO Baseline**
   - Detection class names only + `all-MiniLM-L6-v2`
   - Provides lower bound for comparison

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
# Required for oracle/ config (GPT-4o-mini)
export OPENAI_API_KEY="sk-your-key-here"

# AWS for S3 access
export AWS_PROFILE="your-profile"  # or run 'aws configure'
```

### 3. Run Complete Evaluation

```bash
# Run everything (syncs S3, downloads Space3D-Bench, runs all evals)
./run_complete_eval.sh

# With custom S3 bucket
./run_complete_eval.sh --s3-uri s3://your-bucket/path

# Skip SPICE metric (faster)
./run_complete_eval.sh --skip-spice

# CPU only
./run_complete_eval.sh --device cpu
```

**Note:** If `OPENAI_API_KEY` is not set, the `oracle/` config will be skipped automatically.

### 4. View Results

Results are saved to `~/ablation_eval/results/`:
- `FINAL_REPORT.md` - Complete combined report
- `ablation/ABLATION_RESULTS.md` - Scene graph comparison tables
- `vqa/*/VQA_RESULTS.md` - Space3D-Bench VQA accuracy tables
- `complex_queries/*/COMPLEX_QUERIES_RESULTS.md` - Affordance/negation results

## Individual Scripts

### Scene Graph Comparison

```bash
python run_ablation_eval.py \
    --data_root ~/ablation_eval/data \
    --output_dir ~/ablation_eval/results/ablation \
    --scenes room0 room1 office2 office3 \
    --configs qwen paligemma
```

### VLM Query (Single Scene)

```bash
python vlm_load_and_query.py \
    --model qwen \
    --obj_json /path/to/obj_json.json \
    --edge_json /path/to/edge_json.json \
    --questions /path/to/questions.json \
    --output /path/to/answers.json
```

### CLIP Retrieval Query

```bash
python clip_load_and_query.py \
    --model mobileclip \
    --pkl /path/to/pcd_*.pkl.gz \
    --questions /path/to/questions.json \
    --output /path/to/answers.json \
    --top_k 5
```

### YOLO + Sentence-Transformers Baseline

```bash
python yolo_load_and_query.py \
    --obj_json /path/to/obj_json.json \
    --questions /path/to/questions.json \
    --output /path/to/answers.json \
    --model all-MiniLM-L6-v2
```

### Complex Queries Evaluation

```bash
python run_complex_queries_eval.py \
    --scene_graphs_root ~/ablation_eval/data \
    --queries_path ./complex_queries.json \
    --output_dir ~/ablation_eval/results/complex \
    --vlm_models qwen paligemma \
    --clip_models mobileclip
```

### VQA Evaluation

```bash
python run_vqa_eval.py \
    --scene_graphs_root ~/ablation_eval/data \
    --space3d_root ~/Space3D-Bench/data \
    --output_dir ~/ablation_eval/results/vqa \
    --vlm_models qwen paligemma \
    --clip_models mobileclip
```

## Directory Structure

```
evaluations/
├── run_complete_eval.sh         # Master script
├── run_ablation_eval.py         # CLIP/VLM comparison
├── run_vqa_eval.py              # Space3D-Bench VQA
├── run_complex_queries_eval.py  # Affordance/negation eval
├── vlm_load_and_query.py        # VLM inference
├── clip_load_and_query.py       # CLIP retrieval
├── yolo_load_and_query.py       # YOLO + SentenceTransformer baseline
├── complex_queries.json         # Custom query dataset
├── requirements.txt
├── SPACE3D_BENCH_PLAYBOOK.md
└── README.md

~/ablation_eval/                 # Working directory
├── data/                        # Synced from S3
│   ├── oracle/
│   ├── qwen/
│   └── paligemma/
├── Space3D-Bench/
│   └── data/
└── results/
    ├── FINAL_REPORT.md
    ├── ablation/
    ├── vqa/
    └── complex_queries/
```

## Output Tables

### Example: Complex Queries Results

| Model | Overall ↑ | Affordance ↑ | Negation ↑ |
|-------|-----------|--------------|------------|
| Qwen3-VL-2B | **52.5%** | 60.0% | 45.0% |
| PaliGemma2-3b | 47.5% | 55.0% | 40.0% |

| Model | Top-1 ↑ | Top-5 ↑ | Aff T@1 | Aff T@5 | Neg T@1 | Neg T@5 |
|-------|---------|---------|---------|---------|---------|---------|
| MobileCLIP | 32.5% | 67.5% | 40.0% | 75.0% | 25.0% | 60.0% |
| YOLO+MiniLM | 22.5% | 52.5% | 25.0% | 55.0% | 20.0% | 50.0% |

## GPU Memory Management

For 16GB GPU systems:
- Models are loaded **ONE AT A TIME**
- GPU memory is cleared between runs using `torch.cuda.empty_cache()`
- Each model completes all scenes before the next model loads

To monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Troubleshooting

### SPICE Not Working
SPICE requires Java 1.8+:
```bash
sudo apt install default-jdk
```

### CUDA Out of Memory
Use CPU or run fewer models:
```bash
./run_complete_eval.sh --device cpu
```

### AWS Access Denied
```bash
aws configure
# or
export AWS_PROFILE=your_profile
```

## Citation

If using Space3D-Bench:
```bibtex
@inproceedings{szymanska2024space3dbench,
    title={{Space3D-Bench: Spatial 3D Question Answering Benchmark}},
    author={Szymanska, Emilia and Dusmanu, Mihai and Buurlage, Jan-Willem 
            and Rad, Mahdi and Pollefeys, Marc},
    booktitle={ECCV Workshops},
    year={2024}
}
```
