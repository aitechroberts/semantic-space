"""
Stage A2 / B1 — Encoder feature extraction.

**Phase A mode** (default): For each frame's detections, load saved 1.5x
crop images, batch-encode with the oracle encoder (e.g. ViT-H/14), and
write clip_ft / text_ft back into frame_data.

**Phase B re-embed mode** (``embed.mode=re_embed``): Load the oracle
scene, re-extract features for every object's per_view_records using the
evaluation encoder, compute weighted average + per-view + entropy-selected
best feature, save as variant.

Optional ``--use_sam_fusion``: re-run SAM2 on each crop, black out
background, extract second feature, average with full-crop feature
(+3.3% per Bare Necessities Table 9).

Standalone usage::

    python -m semgraph.stages.embed <hydra overrides>
    python -m semgraph.stages.embed embed.mode=re_embed embed.encoder_name=...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder loading (one model at a time)
# ---------------------------------------------------------------------------

def _load_encoder(encoder_name: str, device: str) -> tuple:
    """Load a HuggingFace CLIP-family model + processor. Returns (model, processor)."""
    from transformers import CLIPModel, CLIPProcessor
    import os

    cache_dir = os.environ.get("HF_HOME")
    ckpt_dir = os.environ.get("CKPT_DIR", "")
    if ckpt_dir and os.path.exists(ckpt_dir):
        hf_cache_dir = os.path.join(ckpt_dir, "huggingface")
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        cache_dir = hf_cache_dir

    model = CLIPModel.from_pretrained(encoder_name, cache_dir=cache_dir).to(device)
    processor = CLIPProcessor.from_pretrained(encoder_name, cache_dir=cache_dir)
    model.eval()
    return model, processor


def _encode_crops(
    model: Any,
    processor: Any,
    crops: list[Image.Image],
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Batch-encode PIL crops and return L2-normalized features as (N, D) float32."""
    all_feats = []
    for i in range(0, len(crops), batch_size):
        batch = crops[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu().numpy())
    if not all_feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def _encode_text(
    model: Any,
    processor: Any,
    texts: list[str],
    device: str,
) -> np.ndarray:
    """Encode text labels and return L2-normalized features as (N, D) float32."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return feats.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# SAM fusion helper
# ---------------------------------------------------------------------------

def _sam_fusion_feature(
    model: Any,
    processor: Any,
    crop: Image.Image,
    base_feat: np.ndarray,
    device: str,
) -> np.ndarray:
    """Re-run SAM2 on the crop, black out background, encode, average with base."""
    try:
        from ultralytics import SAM
    except ImportError:
        return base_feat

    sam = SAM("sam2.1_b.pt")
    crop_np = np.array(crop)
    results = sam.predict(crop_np, verbose=False)
    if results and results[0].masks is not None and results[0].masks.data.numel() > 0:
        mask = results[0].masks.data[0].cpu().numpy() > 0.5
        masked = crop_np.copy()
        masked[~mask] = 0
        masked_pil = Image.fromarray(masked)
        masked_feat = _encode_crops(model, processor, [masked_pil], device)[0]
        fused = (base_feat + masked_feat) / 2.0
        fused = fused / (np.linalg.norm(fused) + 1e-10)
        return fused
    return base_feat


# ---------------------------------------------------------------------------
# Phase A: per-frame feature extraction
# ---------------------------------------------------------------------------

def _run_phase_a(cfg: Any) -> None:
    """Encode all frame_data crops with the oracle encoder."""
    from tqdm import tqdm
    from semgraph.stages.paths import stage_paths
    from semgraph.io import list_frame_indices, load_frame_data, save_frame_data

    paths = stage_paths(cfg)
    encoder_name = cfg.get("embed", {}).get("encoder_name", "openai/clip-vit-large-patch14") \
        if hasattr(cfg, "get") else "openai/clip-vit-large-patch14"
    use_sam_fusion = cfg.get("embed", {}).get("use_sam_fusion", False) \
        if hasattr(cfg, "get") else False
    device = cfg.get("device", "cuda")

    print(f"[embed] Phase A: encoder={encoder_name}, sam_fusion={use_sam_fusion}")
    model, processor = _load_encoder(encoder_name, device)

    frame_indices = list_frame_indices(paths["frame_data"])
    print(f"[embed] Processing {len(frame_indices)} frames")

    for frame_idx in tqdm(frame_indices, desc="embed"):
        record = load_frame_data(paths["frame_data"], frame_idx)
        if record is None:
            continue

        crops = []
        valid_indices = []
        for det_idx, dm in enumerate(record.det_meta):
            crop_path = dm.crop_path
            if crop_path and Path(crop_path).is_file():
                crops.append(Image.open(crop_path).convert("RGB"))
                valid_indices.append(det_idx)
            else:
                logger.debug("Missing crop for frame %d det %d: %s", frame_idx, det_idx, crop_path)

        if not crops:
            continue

        feats = _encode_crops(model, processor, crops, device)

        if use_sam_fusion:
            for i, crop in enumerate(crops):
                feats[i] = _sam_fusion_feature(model, processor, crop, feats[i], device)

        class_names = [record.det_meta[vi].class_name for vi in valid_indices]
        text_feats = _encode_text(model, processor, class_names, device)

        n_det = record.n_detections
        feat_dim = feats.shape[1] if len(feats) > 0 else 512
        clip_ft = record.clip_ft if record.clip_ft is not None else np.zeros((n_det, feat_dim), dtype=np.float32)
        text_ft = record.text_ft if record.text_ft is not None else np.zeros((n_det, feat_dim), dtype=np.float32)

        for i, det_idx in enumerate(valid_indices):
            clip_ft[det_idx] = feats[i]
            if i < len(text_feats):
                text_ft[det_idx] = text_feats[i]

        record.clip_ft = clip_ft
        record.text_ft = text_ft
        save_frame_data(paths["frame_data"], frame_idx, record)

    print("[embed] Phase A done.")


# ---------------------------------------------------------------------------
# Phase B: re-embed oracle scene objects
# ---------------------------------------------------------------------------

def _compute_entropy(feat: np.ndarray, label_feats: np.ndarray) -> float:
    """Softmax similarity entropy over a label set."""
    sims = feat @ label_feats.T
    probs = np.exp(sims - sims.max())
    probs = probs / (probs.sum() + 1e-10)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def _run_phase_b(cfg: Any) -> None:
    """Re-embed oracle scene objects with an evaluation encoder."""
    from semgraph.stages.paths import stage_paths
    from semgraph.io import load_oracle_scene, save_variant, VariantRecord

    paths = stage_paths(cfg)
    embed_cfg = cfg.get("embed", {}) if hasattr(cfg, "get") else {}
    encoder_name = embed_cfg.get("encoder_name", "openai/clip-vit-large-patch14")
    device = cfg.get("device", "cuda")

    print(f"[embed] Phase B re-embed: encoder={encoder_name}")
    model, processor = _load_encoder(encoder_name, device)

    oracle = load_oracle_scene(paths["oracle"])
    if oracle is None:
        raise FileNotFoundError(f"oracle_scene not found at {paths['oracle']}")

    n_objects = len(oracle.class_names)

    # Load entropy label set for best-feature selection
    label_feats = None
    entropy_labels_path = Path("config/replica_50_labels.txt")
    if entropy_labels_path.is_file():
        labels = [line.strip() for line in entropy_labels_path.read_text().splitlines() if line.strip()]
        if labels:
            label_feats = _encode_text(model, processor, labels, device)
    if label_feats is None or len(label_feats) == 0:
        logger.warning("No entropy labels found; best_entropy will default to 0.0")

    all_weighted_avg = []
    all_best = []
    all_entropy = []
    all_pv_feats: list[np.ndarray] = []

    for obj_idx in range(n_objects):
        pv_meta = oracle.per_view_meta[obj_idx] if obj_idx < len(oracle.per_view_meta) else []
        if not pv_meta:
            all_weighted_avg.append(np.zeros(0, dtype=np.float32))
            all_best.append(np.zeros(0, dtype=np.float32))
            all_entropy.append(0.0)
            all_pv_feats.append(np.empty((0, 0), dtype=np.float32))
            continue

        crops = []
        valid_indices = []
        for pi, pm in enumerate(pv_meta):
            cp = pm.crop_path
            if cp and Path(cp).is_file():
                crops.append(Image.open(cp).convert("RGB"))
                valid_indices.append(pi)

        if not crops:
            all_weighted_avg.append(np.zeros(0, dtype=np.float32))
            all_best.append(np.zeros(0, dtype=np.float32))
            all_entropy.append(0.0)
            all_pv_feats.append(np.empty((0, 0), dtype=np.float32))
            continue

        feats = _encode_crops(model, processor, crops, device)

        weights = np.array([pv_meta[pi].n_points for pi in valid_indices], dtype=np.float32)
        total_w = weights.sum() + 1e-10
        weighted_avg = (feats * weights[:, None]).sum(axis=0) / total_w
        weighted_avg = weighted_avg / (np.linalg.norm(weighted_avg) + 1e-10)

        best_entropy_val = 0.0
        best_feat = weighted_avg
        if label_feats is not None and len(label_feats) > 0:
            min_ent = float("inf")
            for feat in feats:
                ent = _compute_entropy(feat, label_feats)
                if ent < min_ent:
                    min_ent = ent
                    best_feat = feat
                    best_entropy_val = ent

        all_weighted_avg.append(weighted_avg)
        all_best.append(best_feat)
        all_entropy.append(best_entropy_val)
        all_pv_feats.append(feats)

    safe_name = encoder_name.replace("/", "_")
    variant = VariantRecord(
        encoder_name=encoder_name,
        clip_ft_weighted_avg=np.stack(all_weighted_avg) if all_weighted_avg and all_weighted_avg[0].ndim > 0 and all_weighted_avg[0].size > 0 else np.empty((0, 0), dtype=np.float32),
        clip_ft_best=np.stack(all_best) if all_best and all_best[0].ndim > 0 and all_best[0].size > 0 else np.empty((0, 0), dtype=np.float32),
        best_entropy=all_entropy,
        pv_feats_list=all_pv_feats,
    )
    save_variant(paths["variants"], variant, f"embed_{safe_name}")
    print(f"[embed] Phase B done. Saved variant for encoder={encoder_name}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main_standalone(cfg):
    """Dispatch to Phase A or Phase B based on embed.mode config."""
    from semgraph.slam.utils import process_cfg
    cfg = process_cfg(cfg)

    mode = "phase_a"
    embed_cfg = cfg.get("embed", {}) if hasattr(cfg, "get") else {}
    if isinstance(embed_cfg, dict):
        mode = embed_cfg.get("mode", "phase_a")
    elif hasattr(embed_cfg, "mode"):
        mode = embed_cfg.mode

    if mode == "re_embed":
        _run_phase_b(cfg)
    else:
        _run_phase_a(cfg)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../hydra_configs", config_name="batch_vlm_mapping_api")
    def main(cfg: DictConfig):
        main_standalone(cfg)

    main()
