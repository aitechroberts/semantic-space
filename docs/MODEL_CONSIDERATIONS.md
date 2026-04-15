# Model Considerations

Per-model notes on context length, VRAM, prompt behavior, and recommended launch settings for the vLLM batch pipeline. Expand this file as each model is evaluated.

---

## Qwen2.5-VL-3B-Instruct

**HuggingFace ID:** `Qwen/Qwen2.5-VL-3B-Instruct`
**Prompt config:** `prompts_standard`
**Approx weights:** ~6 GB

### Image Token Budget

Qwen2.5-VL uses dynamic resolution -- it splits images into 14x14 pixel patches and applies an internal min/max pixel budget. A single annotated frame at JPEG quality 85 (as sent by `VLMAPIClient`) maps to roughly **1,200-1,600 visual tokens** after internal resizing.

| Dataset | Native resolution | Estimated visual tokens |
|---------|-----------------|------------------------|
| Replica | 1200 x 680 | ~1,200-1,400 |
| ScanNet | 1296 x 968 | ~1,400-1,600 |

### Text Token Budget (per request)

| Component | Tokens |
|-----------|--------|
| System prompt | ~15-20 |
| User prompt + `{labels}` (5-15 objects) | ~80-150 |
| Caption output (5-15 objects, ~20-30 tok each) | ~100-450 |
| Relation output (5-20 tuples) | ~100-400 |
| Consolidation input + output | ~80-200 |

### Total Per-Request Context

| Component | Tokens |
|-----------|--------|
| Visual tokens (image) | ~1,200-1,600 |
| Prompt text (input) | ~100-170 |
| Generated output | ~100-450 |
| **Total worst case** | **~1,750-2,220** |

### Context Length Recommendation

- `MAX_MODEL_LEN=2048` works for most frames (up to ~10 detections)
- `MAX_MODEL_LEN=3072` gives comfortable headroom for dense frames (15+ detections)
- `MAX_MODEL_LEN=4096` is unnecessary and eats KV cache memory with no practical benefit

### VRAM on RTX 4090 16GB Laptop

The laptop 4090 shares the same 16GB pool between vLLM and the Python pipeline (YOLO + SAM + TinyCLIP ≈ 1.5 GB).

| `GPU_MEM_UTIL` | vLLM budget | Headroom for pipeline | Notes |
|---------------|------------|----------------------|-------|
| 0.65 | ~10.4 GB | ~5.6 GB | **KV cache OOM** at `MAX_MODEL_LEN=4096` |
| 0.75 | ~12.0 GB | ~4.0 GB | **Recommended.** Works at 2048-3072 tokens |
| 0.80 | ~12.8 GB | ~3.2 GB | OK, pipeline is tight |
| 0.85 | ~13.6 GB | ~2.4 GB | Marginal -- too tight for `EXTRACT_ENCODER=true` (see below) |

### AWQ Quantized Variant for vLLM Serving

Running the full-precision 3B model via vLLM on a 16GB laptop GPU leaves very little headroom for the rest of the pipeline. The **AWQ quantized variant** (`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`) reduces the vLLM serving footprint by ~40%, freeing VRAM for YOLO/SAM/TinyCLIP and the vision encoder extractor.

vLLM handles AWQ natively -- no extra flags needed:

```bash
VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

### Vision Encoder Extraction Strategy

`EXTRACT_ENCODER=true` loads the Qwen2.5-VL vision encoder locally via `VLMEncoderExtractor` for embedding comparison research. The extraction is carefully designed to avoid VRAM spikes:

```python
# 1. Load the full BASE (non-AWQ) model entirely to CPU
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_name, torch_dtype=dtype, device_map="cpu",
)
# 2. Move ONLY model.visual (~675M params, ~1.3GB) to GPU
encoder = model.visual.to(device)
# 3. Extract the image_processor (not the full multimodal processor)
processor = AutoProcessor.from_pretrained(base_name).image_processor
# 4. Delete the full model from CPU RAM
del model
```

Key points:
- **Quantization suffix stripping:** When `VLM_MODEL` is an AWQ/GPTQ variant (e.g. `Qwen2.5-VL-3B-Instruct-AWQ`), the encoder extractor automatically strips the `-AWQ` suffix and loads the base model instead. This bypasses the deprecated `autoawq` library, which is incompatible with `transformers>=4.57`. The vision encoder weights are identical between base and quantized models.
- **CPU-first loading:** The full ~6GB base model is loaded to system RAM, then only `model.visual` (~1.3GB) is moved to GPU. The LLM decoder never touches VRAM. This avoids the previous OOM issue where loading the full model to GPU transiently required ~6GB of free VRAM.
- **Image processor only:** The extractor uses `AutoProcessor(...).image_processor` rather than the full multimodal processor, because the latter expects both text and image inputs (and raises `TypeError` when called with images only).

### Qwen2.5-VL Fused Architecture & Dual-Feature Extraction

Qwen2.5-VL's `model.visual` is a **fused** module containing both the ViT backbone (28 transformer blocks) and a spatial merger/projector. You cannot cleanly separate encoder-only from projected features by attribute access alone. The forward pass:

```
pixel_values → patch_embed → 28 ViT blocks → merger (2x2 spatial merge + linear projection) → output
                              ↑                ↑
                         dim=1280          dim=3584
                      (raw ViT feats)   (projected feats)
```

The extractor captures **both** feature levels in a single forward pass using a PyTorch forward hook:

1. A hook on `encoder.blocks[-1]` captures the raw ViT output (dim=1280) before it enters the merger
2. The normal forward return gives the post-merger projected output (dim=3584)
3. Both are mean-pooled over spatial tokens and L2-normalized per crop

This produces two feature arrays per detected object:
- `vlm_vit_ft` -- raw ViT features (dim=1280), comparable across ViT-based models
- `vlm_proj_ft` -- projected features (dim=3584), captures the model's learned spatial compression

For non-Qwen models (standalone vision towers), only `vlm_vit_ft` is populated; `vlm_proj_ft` is `None`.

Images are processed sequentially in the Qwen path because Qwen's visual model concatenates patches from all images into a single flat sequence (no batch dimension), making per-image feature splitting non-trivial.

### VRAM Budget with AWQ + Encoder Extraction

| Component | VRAM |
|-----------|------|
| vLLM serving AWQ model | ~3.5-4.5 GB |
| YOLO + SAM + TinyCLIP | ~1.5 GB |
| Qwen2.5-VL `model.visual` on GPU | ~1.3 GB |
| KV cache (MAX_MODEL_LEN=2048) | ~1-2 GB |
| **Total peak** | **~7.5-9.5 GB** |

This fits comfortably on a 16GB GPU with `GPU_MEM_UTIL=0.45-0.55`.

### Recommended Launch

Standard mapping run with AWQ (no encoder extraction):

```bash
GPU_MEM_UTIL=0.75 MAX_MODEL_LEN=2048 VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

With encoder extraction (AWQ serving + base model encoder on CPU→GPU):

```bash
GPU_MEM_UTIL=0.50 MAX_MODEL_LEN=2048 EXTRACT_ENCODER=true VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" ./shells/run_vllm_batch.sh
```

Full-precision serving (no encoder extraction, needs more VRAM headroom):

```bash
GPU_MEM_UTIL=0.75 MAX_MODEL_LEN=2048 VLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct" ./shells/run_vllm_batch.sh
```

### Notes

- The `--model` flag was removed in newer vLLM versions; the model ID must be the first positional argument to `vllm serve`. The shell script handles this correctly.
- Qwen2.5-VL's chat template is handled automatically by vLLM -- no `--trust-remote-code` quirks observed at 0.8.x.
- Consolidation requests (`consolidate_prompt`) are text-only (no image) and are far cheaper (~200 total tokens).

---

## Qwen3-VL-2B-Instruct

**HuggingFace ID:** `Qwen/Qwen3-VL-2B-Instruct`
**Prompt config:** `prompts_standard`
**Approx weights:** ~4.5 GB

### Compatibility

Requires **vLLM >= 0.11.0**. The current lockfile pins `vllm==0.8.5.post1`, which throws:

```
AttributeError: 'Qwen3VLConfig' object has no attribute 'vocab_size'
```

Do not attempt to run this model until vLLM is upgraded. The `pyproject.toml` constraint `vllm>=0.8.3,<0.9` must be updated to `>=0.11.0` and `uv sync` re-run. Verify that Open3D and pytorch3d remain compatible with whatever numpy version the new vLLM resolver pulls in before upgrading.

---
