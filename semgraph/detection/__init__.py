"""
semgraph.detection — pluggable detection and segmentation backends.

Factories
---------
``get_detector(name)``  — returns a :class:`Detector` instance (or ``None``).
``get_segmenter(name)`` — returns a :class:`Segmenter` instance.

Usage::

    from semgraph.detection import get_detector, get_segmenter

    detector = get_detector("yolo_world")
    detector.load(weights="yolov8l-worldv2.pt", classes=[...])

    segmenter = get_segmenter("sam")
    segmenter.load(weights="sam2.1_b.pt")
"""

from semgraph.detection.base import (
    Detector,
    DetectionResult,
    Segmenter,
    SegmentationResult,
)


def get_detector(name: str | None) -> Detector | None:
    """Factory that returns the appropriate :class:`Detector` for *name*.

    Returns ``None`` if *name* is ``None`` (used for auto-segmentation
    modes where no detector is needed).
    """
    if name is None:
        return None
    if name == "yolo_world":
        from semgraph.detection.yolo_world import YOLOWorldDetector

        return YOLOWorldDetector()
    if name == "yoloe":
        from semgraph.detection.yoloe import YOLOEDetector

        return YOLOEDetector()
    if name == "florence2":
        from semgraph.detection.florence2 import Florence2Detector

        return Florence2Detector()
    raise ValueError(
        f"Unknown detector '{name}'. Valid options: 'yolo_world', 'yoloe', 'florence2'"
    )


def get_segmenter(name: str) -> Segmenter:
    """Factory that returns the appropriate :class:`Segmenter` for *name*."""
    if name == "sam":
        from semgraph.detection.sam import SAMSegmenter

        return SAMSegmenter()
    if name == "sam3":
        from semgraph.detection.sam3 import SAM3Segmenter

        return SAM3Segmenter()
    raise ValueError(
        f"Unknown segmenter '{name}'. Valid options: 'sam', 'sam3'"
    )


__all__ = [
    "Detector",
    "DetectionResult",
    "Segmenter",
    "SegmentationResult",
    "get_detector",
    "get_segmenter",
]
