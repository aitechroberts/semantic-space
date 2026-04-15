"""
FrameSelector ABC and SelectionResult dataclass.

Every frame selector implements a single method:
  select(poses, n_frames, **kwargs) -> SelectionResult

The ``**kwargs`` contract is critical: concrete selectors **must** accept
``**kwargs`` so that config dicts containing keys for *other* selectors
(e.g. ``stride`` passed to ``FPSPoseSelector``) do not raise errors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SelectionResult:
    """Output of frame selection."""

    frame_indices: np.ndarray  # (K,) int32, sorted ascending
    method: str                # e.g. "fps_pose", "stride"
    metadata: dict[str, Any] = field(default_factory=dict)


class FrameSelector(ABC):
    """Strategy interface for frame selection algorithms."""

    @abstractmethod
    def select(
        self,
        poses: dict[int, np.ndarray],
        n_frames: int | None = None,
        **kwargs: Any,
    ) -> SelectionResult:
        """Select frames from available poses.

        Parameters
        ----------
        poses : dict mapping frame_idx -> (4, 4) c2w matrix
        n_frames : target frame count (interpretation varies by method)
        **kwargs : method-specific parameters; implementations must accept
            ``**kwargs`` to absorb keys intended for other selectors.
        """
        ...
