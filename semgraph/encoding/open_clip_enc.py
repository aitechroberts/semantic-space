"""
OpenCLIPEncoder — ``open_clip`` library backend.

Covers MobileCLIP, MetaCLIP, EVA-CLIP, PE-Core, ViT-H-14, TinyCLIP
(via OpenCLIP hub), and any model hosted through ``open_clip``.

``encoder_name`` uses ``"arch:pretrained"`` format, e.g.
``"ViT-H-14:laion2b_s32b_b79k"`` or
``"hf-hub:wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"``.

File named ``open_clip_enc.py`` (not ``open_clip.py``) to avoid
shadowing the ``open_clip`` package.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from semgraph.encoding.base import EmbeddingEncoder

logger = logging.getLogger(__name__)


class OpenCLIPEncoder(EmbeddingEncoder):
    """OpenCLIP encoder using ``open_clip.create_model_and_transforms``.

    Parameters
    ----------
    encoder_name : str
        Format ``"arch:pretrained"`` (e.g. ``"ViT-H-14:laion2b_s32b_b79k"``)
        or ``"hf-hub:org/model"`` for HuggingFace-hosted OpenCLIP models.
        If no colon is present the string is treated as ``arch`` with
        ``pretrained=""`` (downloads default weights).
    device : str
        Target device.
    batch_size : int
        Max crops per forward pass (default 32).
    """

    def __init__(
        self,
        encoder_name: str,
        device: str = "cuda",
        batch_size: int = 32,
        **kwargs,
    ):
        import open_clip

        self._device = device
        self._batch_size = batch_size

        if encoder_name.startswith("hf-hub:"):
            arch = encoder_name
            pretrained = ""
        elif ":" in encoder_name:
            arch, pretrained = encoder_name.split(":", 1)
        else:
            arch = encoder_name
            pretrained = ""

        logger.info(
            "[OpenCLIPEncoder] Loading arch=%s pretrained=%s", arch, pretrained or "(default)",
        )

        create_kwargs = {"model_name": arch}
        if pretrained:
            create_kwargs["pretrained"] = pretrained

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            **create_kwargs,
        )
        self._model = self._model.to(device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(arch)

        # Resolve feat_dim from a dummy forward pass on a 1-pixel image.
        # open_clip models don't expose projection_dim via a config attribute.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=device)
            dummy_out = self._model.encode_image(dummy)
            self._feat_dim: int = dummy_out.shape[-1]

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
            tensors = torch.stack([self._preprocess(img) for img in batch]).to(self._device)
            feats = self._model.encode_image(tensors)
            feats = F.normalize(feats.float(), dim=-1)
            all_feats.append(feats.cpu().numpy())
        return np.concatenate(all_feats, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray | None:
        if not texts:
            return np.empty((0, self._feat_dim), dtype=np.float32)

        tokens = self._tokenizer(texts).to(self._device)
        feats = self._model.encode_text(tokens)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().astype(np.float32)
