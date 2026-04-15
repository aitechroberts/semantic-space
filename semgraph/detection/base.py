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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Weight resolution (shared by the factory in __init__.py and detect.py)
# ---------------------------------------------------------------------------

_HF_WEIGHT_REPOS: dict[str, str] = {
    "sam3.pt": "facebook/sam3",
}

_DETECTOR_DEFAULTS: dict[str, str] = {
    "yoloe": "yoloe-v8l-seg.pt",
    "yolo_world": "yolov8l-worldv2.pt",
    "florence2": "microsoft/Florence-2-large",
    "gdino": "IDEA-Research/grounding-dino-base",
}


def resolve_weights(filename: str) -> str:
    """Resolve model weights path.

    Search order:
    1. ``$CKPT_DIR/<filename>``
    2. Current working directory (bare *filename*)
    3. HuggingFace hub cache (if *filename* is mapped in ``_HF_WEIGHT_REPOS``)
    4. Fall back to bare *filename* (lets ultralytics try its own download).
    """
    ckpt_dir = os.environ.get("CKPT_DIR", "")
    if ckpt_dir and (Path(ckpt_dir) / filename).exists():
        return str(Path(ckpt_dir) / filename)
    if Path(filename).exists():
        return filename
    if filename in _HF_WEIGHT_REPOS:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=_HF_WEIGHT_REPOS[filename], filename=filename)
            return str(path)
        except Exception:
            pass
    return filename


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

    Implementations: ``YOLOWorldDetector``, ``YOLOEDetector``,
    ``Florence2Detector``, ``GroundingDINODetector``.
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

    @property
    def vocab_driven(self) -> bool:
        """Whether this detector accepts a class vocabulary at load time.

        Vocab-driven detectors (YOLOE, YOLO-World, GroundingDINO) receive
        a class list via ``classes`` kwarg in ``load()`` and produce
        ``class_id`` values that index into that vocabulary.

        Non-vocab-driven detectors (Florence-2) produce their own ad-hoc
        labels and class_ids per frame.
        """
        return False

    @property
    def classes(self) -> list[str] | None:
        """The class vocabulary this detector was loaded with, if any."""
        return None


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
