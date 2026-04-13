"""
Florence2Detector — Florence-2 open-vocabulary object detection via HuggingFace.

Uses the ``<OD>`` (object detection) task prompt to produce bounding boxes
with class labels.  Florence-2 is a vision foundation model that accepts
PIL images and returns structured bounding-box + label predictions.

Reference: https://huggingface.co/microsoft/Florence-2-large
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from semgraph.detection.base import Detector, DetectionResult


class Florence2Detector(Detector):
    """Florence-2 via HuggingFace ``transformers``.

    Produces bounding boxes with class labels using the ``<OD>`` task prompt.
    Works directly from ``image_rgb`` (converted to PIL internally);
    ``color_path`` is ignored.

    Pass ``task=<str>`` to :meth:`load` via ``**kwargs`` to override the
    default ``<OD>`` prompt (e.g. ``<DENSE_REGION_CAPTION>``).
    """

    _model: Any = None
    _processor: Any = None
    _device: str = "cuda"
    _torch_dtype: Any = None
    _task: str = "<OD>"

    def load(self, weights: str, device: str = "cuda", **kwargs: Any) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = device
        self._torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self._task = kwargs.get("task", "<OD>")

        self._model = AutoModelForCausalLM.from_pretrained(
            weights,
            torch_dtype=self._torch_dtype,
            trust_remote_code=True,
        ).to(self._device)

        self._processor = AutoProcessor.from_pretrained(
            weights,
            trust_remote_code=True,
        )

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        color_path: Path | None = None,
    ) -> DetectionResult:
        import torch
        from PIL import Image

        pil_image = Image.fromarray(image_rgb)

        inputs = self._processor(
            text=self._task,
            images=pil_image,
            return_tensors="pt",
        ).to(self._device, self._torch_dtype)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False,
            )

        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=self._task,
            image_size=(pil_image.width, pil_image.height),
        )

        od_result = parsed.get(self._task, {})
        bboxes = od_result.get("bboxes", [])
        labels = od_result.get("labels", [])

        n = len(bboxes)
        if n == 0:
            return DetectionResult(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
                classes=[],
            )

        xyxy = np.array(bboxes, dtype=np.float32)

        # Florence-2 OD does not produce confidence scores; use 1.0
        confidence = np.ones(n, dtype=np.float32)

        # Build vocabulary from unique labels, preserving order
        seen: dict[str, int] = {}
        classes: list[str] = []
        class_ids = np.empty(n, dtype=np.int32)
        class_labels: list[str] = []

        for i, label in enumerate(labels):
            if label not in seen:
                seen[label] = len(classes)
                classes.append(label)
            class_ids[i] = seen[label]
            class_labels.append(f"{label} {i}")

        return DetectionResult(
            xyxy=xyxy,
            confidence=confidence,
            class_ids=class_ids,
            class_labels=class_labels,
            classes=classes,
        )
