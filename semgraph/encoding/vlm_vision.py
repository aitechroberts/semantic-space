"""
VLMVisionEncoder — adapter for :class:`VLMEncoderExtractor`.

Wraps the existing VLM vision tower extraction machinery into the
:class:`EmbeddingEncoder` ABC, enabling VLM-derived embeddings to
participate in encoder sweeps.

Text encoding is available when ``use_proj=True``: the LLM's embedding
table (kept during extraction) maps tokens into the same space as the
post-merger projected image features.  When ``use_proj=False`` the image
features are in the ViT's native space, which is not text-aligned.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from semgraph.encoding.base import EmbeddingEncoder

logger = logging.getLogger(__name__)


class VLMVisionEncoder(EmbeddingEncoder):
    """Vision-tower encoder extracted from a VLM.

    Parameters
    ----------
    encoder_name : str
        HuggingFace model ID (e.g. ``Qwen/Qwen3-VL-2B-Instruct``).
    device : str
        Target device.
    use_proj : bool
        If ``True`` and the model has a fused merger (Qwen family),
        return projected features instead of raw ViT features. Enables
        comparing vit-driven vs proj-driven maps in embedding drift
        studies without a second refactor.
    """

    def __init__(
        self,
        encoder_name: str,
        device: str = "cuda",
        use_proj: bool = False,
        **kwargs,
    ):
        from semgraph.utils.vlms.vlm_encoder import VLMEncoderExtractor

        self._use_proj = use_proj
        self._extractor = VLMEncoderExtractor(encoder_name, device=device)
        self._feat_dim = self._resolve_feat_dim()

        if self._use_proj and self._extractor._embed_table is not None:
            embed_dim = self._extractor._embed_table.embedding_dim
            if embed_dim != self._feat_dim:
                raise ValueError(
                    f"Dimension mismatch: embed_table.embedding_dim={embed_dim} "
                    f"!= feat_dim={self._feat_dim}. Text retrieval (Path B) "
                    f"requires proj_feats and text embeddings to share the "
                    f"same dimensionality."
                )

    # -- feat_dim resolution -------------------------------------------------

    def _resolve_feat_dim(self) -> int:
        """Resolve embedding dimensionality from model config, falling
        back to a dummy forward pass only for unknown architectures."""
        enc = self._extractor.encoder
        config = getattr(enc, "config", None)

        # 1. Direct hidden_size (most VLM vision towers)
        if config is not None and hasattr(config, "hidden_size"):
            dim = config.hidden_size
            if self._use_proj and self._extractor._has_merger:
                # Projected features have a different dim; try embed_dim
                proj_dim = getattr(config, "embed_dim", None) or getattr(config, "projection_dim", None)
                if proj_dim is not None:
                    return int(proj_dim)
            return int(dim)

        # 2. Nested vision_config
        if config is not None and hasattr(config, "vision_config"):
            vc = config.vision_config
            if hasattr(vc, "hidden_size"):
                return int(vc.hidden_size)

        # 3. Dummy forward pass (unknown architectures only)
        logger.warning(
            "[VLMVisionEncoder] Could not resolve feat_dim from config, "
            "running dummy forward pass."
        )
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        vit_feats, proj_feats = self._extractor.encode_crops([dummy])
        if self._use_proj and proj_feats is not None:
            return int(proj_feats.shape[-1])
        return int(vit_feats.shape[-1])

    # -- ABC implementation --------------------------------------------------

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def has_aligned_text_space(self) -> bool:
        return self._use_proj

    def encode_images(self, crops: list[Image.Image]) -> np.ndarray:
        if not crops:
            return np.empty((0, self._feat_dim), dtype=np.float32)

        vit_feats, proj_feats = self._extractor.encode_crops(crops)
        if self._use_proj and proj_feats is not None:
            return proj_feats.astype(np.float32)
        return vit_feats.astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray | None:
        if not self._use_proj:
            return None
        return self._extractor.encode_text(texts)

    def cleanup(self):
        """Free GPU memory held by the underlying VLM vision encoder."""
        self._extractor.cleanup()
