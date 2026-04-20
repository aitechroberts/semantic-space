"""
HFSiglipEncoder — HuggingFace ``SiglipModel`` backend.

Covers ``google/siglip-*``, ``google/siglip2-*``.

SiglipModel shares the ``get_image_features()`` / ``get_text_features()``
API with CLIPModel but was trained with ``padding="max_length"`` for text
tokenization — omitting this produces degraded text features.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from semgraph.encoding.base import EmbeddingEncoder

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class HFSiglipEncoder(EmbeddingEncoder):
    """SigLIP encoder using ``transformers.SiglipModel``.

    Parameters
    ----------
    encoder_name : str
        HuggingFace model ID (e.g. ``google/siglip2-so400m-patch14-384``).
    device : str
        Target device.
    dtype : str
        Weight precision — ``"float16"`` (default) or ``"bfloat16"``.
    batch_size : int
        Max crops per forward pass (default 32).
    """

    def __init__(
        self,
        encoder_name: str,
        device: str = "cuda",
        dtype: str = "float16",
        batch_size: int = 32,
        **kwargs,
    ):
        from transformers import AutoModel, AutoProcessor

        self._device = device
        self._batch_size = batch_size

        cache_dir = os.environ.get("HF_HOME")
        torch_dtype = _DTYPE_MAP.get(dtype, torch.float16)
        logger.info("[HFSiglipEncoder] Loading %s (dtype=%s)", encoder_name, dtype)

        self._model = AutoModel.from_pretrained(
            encoder_name, torch_dtype=torch_dtype, cache_dir=cache_dir,
        ).to(device)
        self._processor = AutoProcessor.from_pretrained(
            encoder_name, cache_dir=cache_dir,
        )
        self._model.eval()
        # SigLIP vision tower uses SiglipMultiheadAttentionPoolingHead, which
        # keeps features at ``vision_config.hidden_size`` (no nn.Linear
        # projection on the image side).  ``SiglipConfig`` does NOT expose
        # ``projection_dim`` (that's the CLIP name), and ``projection_size``
        # exists only on ``text_config``, not ``vision_config``.  Since
        # ``encode_images`` calls ``get_image_features`` the correct dim is
        # ``vision_config.hidden_size``; we fall back to a top-level
        # ``hidden_size`` for edge cases (e.g. a ``SiglipVisionModel`` loaded
        # directly without the composite config).
        cfg = self._model.config
        vcfg = getattr(cfg, "vision_config", None) or cfg
        self._feat_dim: int = int(
            getattr(vcfg, "hidden_size", None)
            or getattr(cfg, "hidden_size", 0)
        )
        if self._feat_dim <= 0:
            raise RuntimeError(
                f"Could not determine feat_dim for {encoder_name} from config"
            )

    # -- ABC implementation --------------------------------------------------

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @torch.no_grad()
    def encode_images(self, crops: list[Image.Image]) -> np.ndarray:
        if not crops:
            return np.empty((0, self._feat_dim), dtype=np.float32)

        all_feats: list[np.ndarray] = []
        for i in range(0, len(crops), self._batch_size):
            batch = crops[i : i + self._batch_size]
            inputs = self._processor(
                images=batch, return_tensors="pt", padding=True,
            ).to(self._device)
            feats = self._model.get_image_features(**inputs)
            if not isinstance(feats, torch.Tensor):
                feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[1]
            feats = F.normalize(feats, dim=-1)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return np.empty((0, self._feat_dim), dtype=np.float32)

        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(self._device)
        feats = self._model.get_text_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats[1]
        feats = F.normalize(feats, dim=-1)
        return feats.cpu().numpy().astype(np.float32)
