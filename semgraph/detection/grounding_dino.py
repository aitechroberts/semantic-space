"""
GroundingDINODetector — open-vocabulary detection via HuggingFace transformers.

Uses ``AutoModelForZeroShotObjectDetection`` with a text prompt built from
the class vocabulary.  Vocab-driven: ``class_id`` values index into the
vocabulary passed at load time.

The GroundingDINO text backbone (BERT) has a hard 256-token limit (the
model was trained with this cap; see the paper §Implementation Details).
Large vocabularies (e.g. ScanNet-200 at ~200 classes) are automatically
chunked into the minimum number of equal parts that each fit within the
token budget.  Inference runs once per chunk with cross-chunk NMS to
deduplicate overlapping detections.

API follows transformers v5.x GroundingDinoProcessor:
  - ``text`` input as ``list[list[str]]`` (processor handles period-joining)
  - ``post_process_grounded_object_detection`` with ``threshold`` kwarg
  - ``target_sizes`` as ``[(height, width)]``

Reference: https://huggingface.co/IDEA-Research/grounding-dino-base
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from semgraph.detection.base import Detector, DetectionResult

logger = logging.getLogger(__name__)

_MAX_TEXT_TOKENS = 250  # headroom below the model's hard 256 limit
_CROSS_CHUNK_NMS_IOU = 0.7  # IoU threshold for deduplicating across chunks


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of xyxy boxes."""
    x1 = np.maximum(a[:, 0:1], b[:, 0])
    y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2])
    y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _nms(
    xyxy: np.ndarray,
    confidence: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy NMS, returns indices of kept detections."""
    order = confidence.argsort()[::-1]
    keep: list[int] = []
    suppressed: set[int] = set()

    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for other in order:
            if other in suppressed or other == idx:
                continue
            iou = _box_iou_matrix(
                xyxy[idx : idx + 1], xyxy[other : other + 1],
            )[0, 0]
            if iou > iou_threshold:
                suppressed.add(other)

    return np.array(keep, dtype=np.intp)


class GroundingDINODetector(Detector):
    """Open-vocabulary detector using GroundingDINO (HuggingFace).

    Pass ``classes=list[str]`` to :meth:`load` to set the detection
    vocabulary.  The processor receives classes as ``list[list[str]]``
    and handles period-separated prompt construction internally.

    When the vocabulary is too large for a single forward pass, it is
    split into the minimum number of equal-sized chunks that each fit
    within the 256-token text limit.  Cross-chunk NMS deduplicates
    overlapping detections after merging.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._device: str = "cuda"
        self._torch: Any = None
        self._classes_list: list[str] = []
        self._chunks: list[tuple[list[str], int]] = []  # (chunk_classes, id_offset)
        self._box_threshold: float = 0.3
        self._text_threshold: float = 0.25

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
        self._box_threshold = kwargs.get("box_threshold", 0.3)
        self._text_threshold = kwargs.get("text_threshold", 0.25)

        self._chunks = self._build_chunks(
            self._classes_list, self._processor.tokenizer,
        )
        logger.info(
            "GroundingDINO: %d classes → %d chunk(s) of sizes %s",
            len(self._classes_list),
            len(self._chunks),
            [len(c) for c, _ in self._chunks],
        )

    @staticmethod
    def _build_chunks(
        classes: list[str], tokenizer: Any,
    ) -> list[tuple[list[str], int]]:
        """Split vocabulary into the minimum number of equal chunks that fit.

        Returns a list of ``(chunk_classes, id_offset)`` tuples.
        """
        if not classes:
            return []

        for n_chunks in range(1, len(classes) + 1):
            chunk_size = math.ceil(len(classes) / n_chunks)
            chunks: list[tuple[list[str], int]] = []
            fits = True

            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(classes))
                part = classes[start:end]
                if not part:
                    continue
                prompt = ". ".join(part) + "."
                n_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
                if n_tokens > _MAX_TEXT_TOKENS:
                    fits = False
                    break
                chunks.append((part, start))

            if fits:
                return chunks

        return [([cls], i) for i, cls in enumerate(classes)]

    def _detect_chunk(
        self,
        pil_img: Any,
        img_h: int,
        img_w: int,
        chunk_classes: list[str],
        id_offset: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Run one forward pass for a chunk, return boxes + globally-mapped IDs."""
        inputs = self._processor(
            images=pil_img,
            text=[chunk_classes],
            return_tensors="pt",
        ).to(self._device)

        with self._torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(img_h, img_w)],
        )[0]

        xyxy = results["boxes"].cpu().numpy().astype(np.float32)
        confidence = results["scores"].cpu().numpy().astype(np.float32)
        text_labels: list[str] = results.get("text_labels", results.get("labels", []))

        if len(xyxy) == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                [],
            )

        class_ids: list[int] = []
        class_labels: list[str] = []

        for label in text_labels:
            label_clean = label.strip().lower()
            matched_local = 0
            for j, cls in enumerate(chunk_classes):
                if cls.lower() in label_clean or label_clean in cls.lower():
                    matched_local = j
                    break
            global_id = id_offset + matched_local
            class_ids.append(global_id)
            class_labels.append(self._classes_list[global_id])

        return xyxy, confidence, np.array(class_ids, dtype=np.int32), class_labels

    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        color_path: Path | None = None,
    ) -> DetectionResult:
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image_rgb)
        img_h, img_w = image_rgb.shape[:2]

        all_xyxy: list[np.ndarray] = []
        all_conf: list[np.ndarray] = []
        all_class_ids: list[np.ndarray] = []
        all_class_labels: list[str] = []

        for chunk_classes, offset in self._chunks:
            xyxy, conf, cids, clabels = self._detect_chunk(
                pil_img, img_h, img_w, chunk_classes, offset,
            )
            if len(xyxy) == 0:
                continue
            all_xyxy.append(xyxy)
            all_conf.append(conf)
            all_class_ids.append(cids)
            all_class_labels.extend(clabels)

        if not all_xyxy:
            return DetectionResult(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_ids=np.empty((0,), dtype=np.int32),
                class_labels=[],
                classes=list(self._classes_list),
            )

        merged_xyxy = np.concatenate(all_xyxy, axis=0)
        merged_conf = np.concatenate(all_conf, axis=0)
        merged_cids = np.concatenate(all_class_ids, axis=0)

        if len(self._chunks) > 1:
            keep = _nms(merged_xyxy, merged_conf, _CROSS_CHUNK_NMS_IOU)
            merged_xyxy = merged_xyxy[keep]
            merged_conf = merged_conf[keep]
            merged_cids = merged_cids[keep]
            all_class_labels = [all_class_labels[i] for i in keep]

        final_labels = [
            f"{name} {i}" for i, name in enumerate(all_class_labels)
        ]

        return DetectionResult(
            xyxy=merged_xyxy,
            confidence=merged_conf,
            class_ids=merged_cids,
            class_labels=final_labels,
            classes=list(self._classes_list),
        )

    @property
    def vocab_driven(self) -> bool:
        return True

    @property
    def classes(self) -> list[str] | None:
        return list(self._classes_list) if self._classes_list else None
