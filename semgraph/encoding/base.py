"""
EmbeddingEncoder ABC.

Every embedding encoder implements:

- ``encode_images(crops) -> (N, D) float32 L2-normalized``
- ``encode_texts(texts) -> (N, D) float32 L2-normalized | None``
- ``feat_dim -> int``

The output contract is identical across all encoder families: downstream
consumers (``build_map.py`` similarity computation, ``FrameDataRecord``
serialization) always receive L2-normalized float32 numpy arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class EmbeddingEncoder(ABC):
    """Strategy interface for embedding encoders.

    Implementations: ``HFCLIPEncoder``, ``HFSiglipEncoder``,
    ``OpenCLIPEncoder``, ``VLMVisionEncoder``.
    """

    @abstractmethod
    def encode_images(self, crops: list[Image.Image]) -> np.ndarray:
        """Encode PIL image crops.

        Returns
        -------
        np.ndarray
            ``(N, D)`` float32, L2-normalized along dim=-1.
            Returns ``(0, D)`` when *crops* is empty (D = :attr:`feat_dim`).
        """

    def encode_texts(self, texts: list[str]) -> np.ndarray | None:
        """Encode text labels.

        Returns
        -------
        np.ndarray | None
            ``(N, D)`` float32, L2-normalized, or ``None`` if the encoder
            does not support text (e.g. vision-only VLM towers).
        """
        return None

    @property
    def has_aligned_text_space(self) -> bool:
        """Whether encode_texts() produces vectors in the same space as encode_images().

        Contrastive encoders (CLIP, SigLIP, OpenCLIP) return ``True``.
        VLM-extracted encoders return ``True`` only when ``use_proj=True``
        (post-merger features share the LLM input space with text embeddings).
        """
        return True

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        """Dimensionality of the output embedding space."""
