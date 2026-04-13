"""
SAM3Segmenter — SAM 3 (Segment Anything with Concepts) via ultralytics.

SAM 3 is backward-compatible with SAM 2's box-prompted and auto-mode
segmentation API.  This implementation uses the standard ``SAM("sam3.pt")``
interface for mask generation, matching the same ``Segmenter`` contract as
:class:`SAMSegmenter`.

SAM 3 additionally supports text-prompted concept segmentation via
``SAM3SemanticPredictor``, but that capability is not exposed here — it
would be a different kind of ``Detector`` (one that returns masks directly).

Reference: https://docs.ultralytics.com/models/sam-3/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from semgraph.detection.base import Segmenter, SegmentationResult


class SAM3Segmenter(Segmenter):
    """SAM 3 via ultralytics (SAM 2–compatible interface).

    Supports two modes controlled by the ``boxes`` argument to
    :meth:`segment`:

    * **Auto mode** (``boxes=None``): SAM 3 runs in "segment everything" mode
      and returns all discovered masks with their bounding boxes and
      confidence scores.
    * **Box-prompted mode** (``boxes=(N,4)``): SAM 3 produces one mask per
      input bounding box.

    SAM 3 weights (``sam3.pt``) must be downloaded manually from
    `HuggingFace <https://huggingface.co/facebook/sam3>`_ — they are not
    auto-downloaded by ultralytics.
    """

    _model: Any = None

    def load(self, weights: str, device: str = "cuda", **kwargs: Any) -> None:
        from ultralytics import SAM

        self._model = SAM(weights)

    def segment(
        self,
        image_rgb: np.ndarray,
        boxes: np.ndarray | None = None,
        *,
        color_path: Path | None = None,
    ) -> SegmentationResult:
        source: Any = str(color_path) if color_path is not None else image_rgb
        H, W = image_rgb.shape[:2]

        if boxes is not None and len(boxes) > 0:
            return self._segment_box_prompted(source, boxes, H, W)
        return self._segment_auto(source, H, W)

    # ------------------------------------------------------------------

    def _segment_box_prompted(
        self, source: Any, boxes: np.ndarray, H: int, W: int,
    ) -> SegmentationResult:
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        sam_out = self._model.predict(source, bboxes=boxes_tensor, verbose=False)
        masks_tensor = sam_out[0].masks.data
        masks_np = masks_tensor.detach().cpu().numpy()
        if masks_np.dtype != np.bool_:
            masks_np = masks_np > 0.5

        n = min(boxes.shape[0], masks_np.shape[0])
        if n == 0:
            return self._empty(H, W)

        masks_np = masks_np[:n]
        xyxy = boxes[:n].astype(np.float32)
        confidence = np.ones(n, dtype=np.float32)
        return SegmentationResult(masks=masks_np, xyxy=xyxy, confidence=confidence)

    def _segment_auto(
        self, source: Any, H: int, W: int,
    ) -> SegmentationResult:
        sam_results = self._model.predict(source, verbose=False)
        r = sam_results[0]

        if r.masks is not None and r.masks.data.numel() > 0:
            masks_np = r.masks.data.detach().cpu().numpy()
            if masks_np.dtype != np.bool_:
                masks_np = masks_np > 0.5
            xyxy_np = r.boxes.xyxy.cpu().numpy()
            confidence = (
                r.boxes.conf.cpu().numpy()
                if r.boxes.conf is not None
                else np.ones(len(xyxy_np), dtype=np.float32)
            )
            return SegmentationResult(masks=masks_np, xyxy=xyxy_np, confidence=confidence)

        return self._empty(H, W)

    @staticmethod
    def _empty(H: int, W: int) -> SegmentationResult:
        return SegmentationResult(
            masks=np.empty((0, H, W), dtype=np.bool_),
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
        )
