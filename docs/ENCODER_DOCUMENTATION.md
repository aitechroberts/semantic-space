# Encoder Registry

The embedding encoder registry (`semgraph/encoding/`) provides a pluggable
strategy pattern for swapping vision and text encoders across the pipeline.
All encoders implement the `EmbeddingEncoder` ABC and return L2-normalized
`(N, D)` float32 numpy arrays — the contract downstream consumers expect.

## Architecture

```
Hydra Config (encoder_type + encoder_name)
        │
        ▼
  get_encoder()          ← if-chain factory with lazy imports
        │
        ├── hf_clip       → HFCLIPEncoder    (transformers.CLIPModel)
        ├── hf_siglip     → HFSiglipEncoder  (transformers.SiglipModel)
        ├── open_clip      → OpenCLIPEncoder   (open_clip library)
        └── vlm_vision     → VLMVisionEncoder  (VLMEncoderExtractor adapter)
        │
        ▼
  EmbeddingEncoder ABC
    .encode_images(crops) → (N, D) float32
    .encode_texts(texts)  → (N, D) float32 | None
    .feat_dim             → int
```

This mirrors the existing `get_detector()` / `get_segmenter()` factories in
`semgraph/detection/` and `get_frame_selector()` in `semgraph/sampling/`.

---

## Encoder Families

### `hf_clip` — HFCLIPEncoder

**Module**: `semgraph/encoding/hf_clip.py`

**What it covers**: Any model with `architectures: ["CLIPModel"]` on
HuggingFace — `openai/clip-vit-*`, `laion/CLIP-ViT-bigG-*`,
`wkcn/TinyCLIP-*`, and similar.

| Aspect | Detail |
|--------|--------|
| Library | `transformers.CLIPModel` + `CLIPProcessor` |
| Loading | `CLIPModel.from_pretrained(name, torch_dtype=dtype)` |
| Image encoding | `model.get_image_features()` with `BaseModelOutputWithPooling` extraction |
| Text encoding | `model.get_text_features()` with same extraction |
| `feat_dim` source | `model.config.projection_dim` |
| Default precision | `float16` (~5 GB for bigG-14, ~1.2 GB for ViT-L/14) |
| Text support | Yes |

**Example model IDs**:
- `openai/clip-vit-base-patch32` (512-d, ~350 MB)
- `openai/clip-vit-large-patch14` (768-d, ~1.2 GB)
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (1280-d, ~5 GB fp16)
- `wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M` (512-d, ~60 MB)

**Note on `BaseModelOutputWithPooling`**: Recent `transformers` versions
return this object instead of a raw tensor from `get_image_features()` /
`get_text_features()`. The encoder extracts `.pooler_output` automatically.

### `hf_siglip` — HFSiglipEncoder

**Module**: `semgraph/encoding/hf_siglip.py`

**What it covers**: `google/siglip-*`, `google/siglip2-*`.

| Aspect | Detail |
|--------|--------|
| Library | `transformers.AutoModel` + `AutoProcessor` (resolves to `SiglipModel`) |
| Loading | `AutoModel.from_pretrained(name, torch_dtype=dtype)` |
| Image encoding | `model.get_image_features()` — same API as CLIPModel |
| Text encoding | `model.get_text_features()` with **`padding="max_length"`** |
| `feat_dim` source | `model.config.projection_dim` |
| Default precision | `float16` |
| Text support | Yes |

**Critical difference from CLIP**: SigLIP was trained with
`padding="max_length"` for text tokenization. Omitting this produces
degraded text features. The encoder handles this automatically.

**Example model IDs**:
- `google/siglip-so400m-patch14-384` (1152-d)
- `google/siglip2-so400m-patch14-384` (1152-d)

### `open_clip` — OpenCLIPEncoder

**Module**: `semgraph/encoding/open_clip_enc.py`

**What it covers**: MobileCLIP, MetaCLIP, EVA-CLIP, PE-Core, ViT-H-14,
TinyCLIP (via OpenCLIP hub), and any model hosted through the `open_clip`
library.

| Aspect | Detail |
|--------|--------|
| Library | `open_clip` |
| Loading | `open_clip.create_model_and_transforms(arch, pretrained=...)` |
| Image encoding | `model.encode_image(preprocessed_tensors)` — uses `open_clip`'s own preprocessing |
| Text encoding | `model.encode_text(tokenized)` via `open_clip.get_tokenizer(arch)` |
| `feat_dim` source | Dummy forward pass (open_clip doesn't expose dim via config) |
| Default precision | Model default (typically fp32, cast to fp32 for L2-norm) |
| Text support | Yes |

**`encoder_name` format**: `"arch:pretrained"` (colon-separated), e.g.:
- `"ViT-H-14:laion2b_s32b_b79k"` (1024-d)
- `"ViT-bigG-14:laion2b_s39b_b160k"` (1280-d)
- `"hf-hub:wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"` (512-d)

If no colon is present, the string is treated as `arch` with default
pretrained weights.

### `vlm_vision` — VLMVisionEncoder

**Module**: `semgraph/encoding/vlm_vision.py`

**What it covers**: Vision tower extraction from any supported VLM. Wraps
the existing `VLMEncoderExtractor` in `semgraph/utils/vlms/vlm_encoder.py`.

| Aspect | Detail |
|--------|--------|
| Library | `transformers` (model-specific classes) via `VLMEncoderExtractor` |
| Loading | Full VLM loaded, then LLM decoder deleted to save VRAM |
| Image encoding | `extractor.encode_crops()` returning `(vit_feats, proj_feats)` |
| Text encoding | **None** — VLM vision towers have no text encoder |
| `feat_dim` source | `encoder.config.hidden_size` (primary), `vision_config.hidden_size` (nested), dummy forward (fallback) |
| Default precision | `float16` or `bfloat16` (auto-detected based on GPU support) |
| Text support | No |

**`use_proj` flag**: When `True` and the model has a fused merger (Qwen
family), returns projected features instead of raw ViT features. This
enables comparing vit-driven vs projection-driven maps in embedding drift
studies.

**Supported VLM families**: Qwen3-VL, Qwen2.5-VL, Qwen2-VL, InternVL,
CogVLM/CogVLM2, Ovis, Gemma 3, LLaVA/LLaVA-OneVision, MiniCPM-V,
SmolVLM, Idefics2/3.

**Example model IDs**:
- `Qwen/Qwen3-VL-2B-Instruct`
- `OpenGVLab/InternVL2-8B`
- `google/gemma-3-4b-it`

---

## Configuration Reference

### Hydra config fields (`batch_vlm_mapping_api.yaml`)

```yaml
embed:
  encoder_type: ${oc.env:EMBED_ENCODER_TYPE,hf_clip}
  encoder_name: ${oc.env:EMBED_ENCODER,laion/CLIP-ViT-bigG-14-laion2B-39B-b160k}
  dtype: ${oc.env:EMBED_DTYPE,float16}
  use_proj: !!bool false           # VLM-only: return proj_feats instead of vit_feats
  mode: phase_a                    # "phase_a" or "re_embed"
  use_sam_fusion: !!bool false     # +3.3% accuracy per Bare Necessities Table 9
```

| Field | Type | Description |
|-------|------|-------------|
| `encoder_type` | str | Backend key: `hf_clip`, `hf_siglip`, `open_clip`, `vlm_vision` |
| `encoder_name` | str | Model identifier (format depends on `encoder_type`) |
| `dtype` | str | Weight precision: `float16`, `bfloat16`, `float32` |
| `use_proj` | bool | VLM-only: return projected features instead of raw ViT features |
| `mode` | str | `phase_a` (per-frame) or `re_embed` (Phase B oracle re-embedding) |
| `use_sam_fusion` | bool | Re-run SAM on each crop and average with base feature |

### Environment variable overrides

| Variable | Hydra field | Example |
|----------|-------------|---------|
| `EMBED_ENCODER_TYPE` | `embed.encoder_type` | `hf_siglip` |
| `EMBED_ENCODER` | `embed.encoder_name` | `google/siglip2-so400m-patch14-384` |
| `EMBED_DTYPE` | `embed.dtype` | `bfloat16` |

### `encoder_name` format by type

| `encoder_type` | Format | Example |
|-----------------|--------|---------|
| `hf_clip` | HuggingFace model ID | `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` |
| `hf_siglip` | HuggingFace model ID | `google/siglip2-so400m-patch14-384` |
| `open_clip` | `arch:pretrained` | `ViT-H-14:laion2b_s32b_b79k` |
| `vlm_vision` | HuggingFace model ID | `Qwen/Qwen3-VL-2B-Instruct` |

---

## Embedding Flow Through the Pipeline

### Phase A — per-frame feature extraction (`embed.py`)

1. **`get_encoder(encoder_type, encoder_name, device, **kwargs)`** creates the
   encoder once at the start of the run.
2. For each frame, crop images are loaded from disk and batch-encoded with
   `encoder.encode_images(crops)` → `clip_ft` in `FrameDataRecord`.
3. If SAM fusion is enabled, SAM is loaded once and passed to each
   `_sam_fusion_feature()` call — the encoder re-encodes the masked crop and
   averages with the base feature.
4. **Text encoding guard**: Before calling `encoder.encode_texts()`, two
   conditions are checked:
   - The encoder supports text (i.e., `encode_texts()` returns non-`None`)
   - The class names are meaningful (not all matching `r"^object \d+$"`
     placeholders from SAM auto-mode)

   If either condition fails, `text_ft` remains zeroed. This keeps the ABC
   clean (the encoder doesn't know about SAM auto-mode) while `embed.py`
   handles the pipeline-level optimization.
5. `clip_ft` and `text_ft` are written back to `FrameDataRecord`.

### Visual similarity in `build_map.py`

`clip_ft` drives `compute_visual_similarities()` in the mapping stage —
cosine similarity on L2-normalized features. Since all encoders produce
L2-normalized outputs, the dot product equals cosine similarity.

### Text features

`text_ft` is stored in `FrameDataRecord` but is **currently inactive** in
the matching pipeline (the similarity aggregation in `slam/mapping.py` does
not use text features for matching). It is preserved for future semantic
matching experiments.

### Phase B — oracle re-embedding

Phase B loads an `OracleSceneRecord`, re-encodes every object's per-view
crops with a (potentially different) encoder, and saves a `VariantRecord`
with:
- `clip_ft_weighted_avg` — point-count-weighted average of per-view features
- `clip_ft_best` — feature with minimum entropy over a label set
- `pv_feats_list` — all per-view features

### SAM fusion

When `use_sam_fusion=True`, SAM is loaded once at the start of Phase A.
For each detection crop:
1. SAM predicts the dominant mask
2. Background pixels are zeroed
3. The masked crop is re-encoded
4. The base and masked features are averaged and re-normalized

This was measured at +3.3% accuracy per the Bare Necessities Table 9.

---

## Evaluation Considerations

### Dimension mismatch across encoders

You **cannot** compare `build_map` results across encoders with different
`feat_dim` values. The map's `clip_ft` arrays are encoder-specific — a
map built with bigG-14 (1280-d) is incompatible with one built with
ViT-L/14 (768-d). Each encoder sweep must re-run `embed` + `build_map` +
`oracle_finalize` end-to-end.

### Encoder sweep workflow

```bash
# Sweep CLIP variants:
for ENC in "openai/clip-vit-base-patch32" \
           "openai/clip-vit-large-patch14" \
           "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" \
           "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"; do
  EMBED_ENCODER=$ENC python -m semgraph.stages.embed ...
done

# SigLIP:
EMBED_ENCODER_TYPE=hf_siglip \
EMBED_ENCODER=google/siglip2-so400m-patch14-384 \
  python -m semgraph.stages.embed ...

# OpenCLIP (arch:pretrained format):
EMBED_ENCODER_TYPE=open_clip \
EMBED_ENCODER="ViT-H-14:laion2b_s32b_b79k" \
  python -m semgraph.stages.embed ...

# VLM vision tower:
EMBED_ENCODER_TYPE=vlm_vision \
EMBED_ENCODER=Qwen/Qwen3-VL-2B-Instruct \
  python -m semgraph.stages.embed ...
```

### `vlm_vit_feats` / `vlm_proj_feats` in `RawGobs`/`SerializedDetection`

These fields are populated by the monolithic VLM detection path and are
**out of scope** for the encoder registry. The encoder registry covers the
standalone embedding stage (`embed.py`), not the inline VLM feature
extraction during detection. Documenting this to prevent confusion.

### `use_proj` flag

Use this when comparing VLM projection-head embeddings vs raw ViT features
for the same VLM. Only Qwen-family models produce both — for all other VLM
families, `proj_feats` is `None` regardless of this flag.

---

## Adding a New Encoder Family

1. **Create** `semgraph/encoding/my_encoder.py` implementing
   `EmbeddingEncoder`:

   ```python
   from semgraph.encoding.base import EmbeddingEncoder

   class MyEncoder(EmbeddingEncoder):
       def __init__(self, encoder_name, device="cuda", **kwargs):
           # Load model...
           self._feat_dim = ...

       @property
       def feat_dim(self) -> int:
           return self._feat_dim

       def encode_images(self, crops):
           # Return (N, D) float32 L2-normalized numpy
           ...

       def encode_texts(self, texts):
           # Return (N, D) float32 L2-normalized numpy, or None
           ...
   ```

2. **Add an `elif` branch** in `semgraph/encoding/__init__.py`:

   ```python
   if encoder_type == "my_encoder":
       from semgraph.encoding.my_encoder import MyEncoder
       return MyEncoder(encoder_name, device=device, **kwargs)
   ```

3. **Update the `ValueError` message** to include the new type.

4. **Test** with:

   ```bash
   EMBED_ENCODER_TYPE=my_encoder \
   EMBED_ENCODER=org/model-id \
     python -m semgraph.stages.embed ...
   ```

5. **Document** the new family in this file.

---

## Memory Footprint Reference

| Model | `encoder_type` | `feat_dim` | fp16 VRAM |
|-------|---------------|-----------|-----------|
| ViT-B/32 (OpenAI) | `hf_clip` | 512 | ~350 MB |
| ViT-L/14 (OpenAI) | `hf_clip` | 768 | ~1.2 GB |
| ViT-bigG/14 (LAION) | `hf_clip` | 1280 | ~5 GB |
| TinyCLIP-8M | `hf_clip` | 512 | ~60 MB |
| SigLIP-SO400M | `hf_siglip` | 1152 | ~1.6 GB |
| ViT-H/14 (OpenCLIP) | `open_clip` | 1024 | ~2.5 GB |
| Qwen3-VL-2B vision tower | `vlm_vision` | 1536 | ~1.2 GB |

All values approximate. VLM vision towers strip the LLM decoder and
projector at load time, keeping only the ViT — actual VRAM is much less
than the full VLM.
