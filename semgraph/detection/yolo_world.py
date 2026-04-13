"""
YOLOWorldDetector — YOLO-World v2 closed-vocabulary object detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from semgraph.detection.base import Detector, DetectionResult


class YOLOWorldDetector(Detector):
    """YOLO-World v2 via ultralytics.

    Produces bounding boxes with class labels from a fixed vocabulary.
    Pass ``classes=list[str]`` to :meth:`load` to set the vocabulary
    (calls ``model.set_classes()`` internally).
    """

    _model: Any = None
    _classes: list[str] | None = None

    def load(self, weights: str, device: str = "cuda", **kwargs: Any) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        classes = kwargs.get("classes")
        if classes is not None:
            self._classes = list(classes)
            self._model.set_classes(self._classes)

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        color_path: Path | None = None,
    ) -> DetectionResult:
        source: Any = str(color_path) if color_path is not None else image_rgb
        results = self._model.predict(source, conf=0.1, verbose=False)

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confidence = boxes.conf.cpu().numpy().astype(np.float32)
        class_ids = boxes.cls.cpu().numpy().astype(np.int32)

        classes = self._classes if self._classes is not None else ["object"]
        class_labels = [
            f"{classes[cid]} {ci}" for ci, cid in enumerate(class_ids)
        ]

        return DetectionResult(
            xyxy=xyxy,
            confidence=confidence,
            class_ids=class_ids,
            class_labels=class_labels,
            classes=classes,
        )
