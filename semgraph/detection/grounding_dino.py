"""
GroundingDINODetector — open-vocabulary detection via HuggingFace transformers.

Uses ``AutoModelForZeroShotObjectDetection`` with a text prompt built from
the class vocabulary.  Vocab-driven: ``class_id`` values index into the
vocabulary passed at load time.

Reference: https://huggingface.co/IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from semgraph.detection.base import Detector, DetectionResult


class GroundingDINODetector(Detector):
    """Open-vocabulary detector using GroundingDINO (HuggingFace).

    Pass ``classes=list[str]`` to :meth:`load` to set the detection
    vocabulary.  The text prompt is constructed by joining class names
    with ``". "`` separators as required by the GroundingDINO API.
    """

    _model: Any = None
    _processor: Any = None
    _device: str = "cuda"
    _classes_list: list[str] = []
    _text_prompt: str = ""
    _box_threshold: float = 0.3
    _text_threshold: float = 0.25

    def load(self, weights: str, device: str = "cuda", **kwargs: Any) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self._device = device
        self._torch = torch

        self._processor = AutoProcessor.from_pretrained(weights)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            weights,
        ).to(device)
        self._model.eval()

        self._classes_list = list(kwargs.get("classes", []))
        self._text_prompt = (
            ". ".join(self._classes_list) + "." if self._classes_list else ""
        )
        self._box_threshold = kwargs.get("box_threshold", 0.3)
        self._text_threshold = kwargs.get("text_threshold", 0.25)

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        color_path: Path | None = None,
    ) -> DetectionResult:
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image_rgb)
        inputs = self._processor(
            images=pil_img, text=self._text_prompt, return_tensors="pt",
        ).to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[pil_img.size[::-1]],
        )[0]

        xyxy = results["boxes"].cpu().numpy().astype(np.float32)
        confidence = results["scores"].cpu().numpy().astype(np.float32)
        raw_labels: list[str] = results["labels"]

        n = len(xyxy)
        if n == 0:
            return DetectionResult(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
                classes=list(self._classes_list),
            )

        class_ids = []
        class_labels = []
        for i, label in enumerate(raw_labels):
            label_clean = label.strip().lower()
            matched_idx = 0
            for j, cls in enumerate(self._classes_list):
                if cls.lower() in label_clean or label_clean in cls.lower():
                    matched_idx = j
                    break
            class_ids.append(matched_idx)
            class_labels.append(f"{self._classes_list[matched_idx]} {i}")

        return DetectionResult(
            xyxy=xyxy,
            confidence=confidence,
            class_ids=np.array(class_ids, dtype=np.int32),
            class_labels=class_labels,
            classes=list(self._classes_list),
        )

    @property
    def vocab_driven(self) -> bool:
        return True

    @property
    def classes(self) -> list[str] | None:
        return list(self._classes_list) if self._classes_list else None
