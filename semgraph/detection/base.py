"""
Detection and segmentation ABCs.

Two independent jobs compose into a detection pipeline:

1. **Detector** — produces bounding boxes (+ optional class labels) from an image.
2. **Segmenter** — produces pixel-precise masks, optionally prompted by boxes.

Concrete implementations live in sibling modules (``yolo_world.py``,
``sam.py``).  The factory in ``__init__.py`` maps config strings to
implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    """Output of a Detector: boxes + class info, no masks."""

    xyxy: np.ndarray  # (N, 4) float32
    confidence: np.ndarray  # (N,) float32
    class_ids: np.ndarray  # (N,) int32
    class_labels: list[str]  # per-detection, e.g. ["chair 0", "table 1"]
    classes: list[str]  # vocabulary, e.g. ["chair", "table", ...]


@dataclass
class SegmentationResult:
    """Output of a Segmenter: masks + derived boxes/confidence."""

    masks: np.ndarray  # (N, H, W) bool
    xyxy: np.ndarray  # (N, 4) float32 — from auto-mode boxes or pass-through
    confidence: np.ndarray  # (N,) float32 — from auto-mode conf or pass-through


class Detector(ABC):
    """Strategy interface for object detectors (box producers).

    Implementations: ``YOLOWorldDetector``, future ``Florence2Detector``, etc.
    """

    @abstractmethod
    def load(self, weights: str, device: str = "cuda", **kwargs: Any) -> None:
        """Load model weights.

        Parameters
        ----------
        weights : str
            Path to weights file or HuggingFace model ID.
        device : str
            Target device (``"cuda"``, ``"cpu"``).
        **kwargs
            Model-specific options (e.g. ``classes`` for YOLO-World).
        """

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        color_path: Path | None = None,
    ) -> DetectionResult:
        """Run detection on a single image.

        Parameters
        ----------
        image_rgb : np.ndarray
            (H, W, 3) uint8 RGB image — the primary input.
        color_path : Path | None
            Optional file path.  Some backends (ultralytics) prefer loading
            from disk; others ignore this and work from the array.
        """


class Segmenter(ABC):
    """Strategy interface for instance segmenters (mask producers).

    Implementations: ``SAMSegmenter``, future ``MobileSAMSegmenter``, etc.
    """

    @abstractmethod
    def load(self, weights: str, device: str = "cuda", **kwargs: Any) -> None:
        """Load model weights.

        Parameters
        ----------
        weights : str
            Path to weights file or HuggingFace model ID.
        device : str
            Target device.
        **kwargs
            Model-specific options.
        """

    @abstractmethod
    def segment(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray | None = None,
        *,
        color_path: Path | None = None,
    ) -> SegmentationResult:
        """Run segmentation on a single image.

        Parameters
        ----------
        image_rgb : np.ndarray
            (H, W, 3) uint8 RGB image — the primary input.
        boxes : np.ndarray | None
            If ``None``, run in automatic (segment-everything) mode.
            If ``(N, 4) float32``, run box-prompted segmentation.
        color_path : Path | None
            Optional file path for backends that prefer disk loading.
        """
