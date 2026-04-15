"""
semgraph.sampling — pluggable frame selection backends.

Factory
-------
``get_frame_selector(name)`` — returns a :class:`FrameSelector` instance.

Usage::

    from semgraph.sampling import get_frame_selector

    selector = get_frame_selector("fps_pose")
    result = selector.select(poses, n_frames=30, position_weight=1.0)
"""

from semgraph.sampling.base import FrameSelector, SelectionResult


def get_frame_selector(name: str) -> FrameSelector:
    """Factory that returns the appropriate :class:`FrameSelector` for *name*."""
    if name == "stride":
        from semgraph.sampling.stride import StrideSelector

        return StrideSelector()
    if name == "fps_pose":
        from semgraph.sampling.fps_pose import FPSPoseSelector

        return FPSPoseSelector()
    raise ValueError(
        f"Unknown frame selector '{name}'. Valid options: 'stride', 'fps_pose'"
    )


__all__ = [
    "FrameSelector",
    "SelectionResult",
    "get_frame_selector",
]
