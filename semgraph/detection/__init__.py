"""
semgraph.detection — pluggable detection and segmentation backends.

Factories
---------
``get_detector(detector_type, detector_name, ...)`` — returns a **loaded**
:class:`Detector` instance.  Mirrors ``get_encoder()`` in
``semgraph/encoding/``.

``get_segmenter(name)`` — returns a :class:`Segmenter` instance.

Usage::

    from semgraph.detection import get_detector, get_segmenter

    detector = get_detector("yoloe", "yoloe-v8l-seg.pt", device="cuda",
                            classes=["chair", "table"])

    segmenter = get_segmenter("sam")
    segmenter.load(weights="sam2.1_b.pt")
"""

from semgraph.detection.base import (
    Detector,
    DetectionResult,
    Segmenter,
    SegmentationResult,
    _DETECTOR_DEFAULTS,
    resolve_weights,
)


def get_detector(
    detector_type: str,
    detector_name: str | None = None,
    device: str = "cuda",
    **kwargs,
) -> Detector:
    """Factory that returns a **loaded** :class:`Detector`.

    Mirrors ``get_encoder(encoder_type, encoder_name, device, **kwargs)``
    in ``semgraph/encoding/__init__.py``.

    Parameters
    ----------
    detector_type : str
        Backend key: ``"yoloe"``, ``"yolo_world"``, ``"florence2"``,
        ``"gdino"``.
    detector_name : str, optional
        Model ID or weight path.  If ``None``, uses the default from
        ``_DETECTOR_DEFAULTS[detector_type]``.
    device : str
        Target device for model loading.
    **kwargs
        Forwarded to ``detector.load()``.  For vocab-driven detectors,
        pass ``classes=list[str]``.
    """
    if detector_name is None:
        detector_name = _DETECTOR_DEFAULTS.get(detector_type)
        if detector_name is None:
            raise ValueError(
                f"No default weights for detector_type={detector_type!r}. "
                f"Pass detector_name explicitly."
            )

    weights = resolve_weights(detector_name)

    if detector_type == "yoloe":
        from semgraph.detection.yoloe import YOLOEDetector

        det = YOLOEDetector()

    elif detector_type == "yolo_world":
        from semgraph.detection.yolo_world import YOLOWorldDetector

        det = YOLOWorldDetector()

    elif detector_type == "florence2":
        from semgraph.detection.florence2 import Florence2Detector

        det = Florence2Detector()

    elif detector_type == "gdino":
        from semgraph.detection.grounding_dino import GroundingDINODetector

        det = GroundingDINODetector()

    else:
        raise ValueError(
            f"Unknown detector_type={detector_type!r}. "
            f"Valid: yoloe, yolo_world, florence2, gdino"
        )

    det.load(weights=weights, device=device, **kwargs)
    return det


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
