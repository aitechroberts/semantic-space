"""StrideSelector — uniform stride frame selection (backward-compatible default)."""

from __future__ import annotations

from typing import Any

import numpy as np

from semgraph.sampling.base import FrameSelector, SelectionResult


class StrideSelector(FrameSelector):
    """Select every Nth frame from sorted pose indices."""

    def select(
        self,
        poses: dict[int, np.ndarray],
        n_frames: int | None = None,
        *,
        stride: int = 10,
        **kwargs: Any,
    ) -> SelectionResult:
        all_indices = sorted(poses.keys())
        selected = all_indices[::stride]
        return SelectionResult(
            frame_indices=np.array(selected, dtype=np.int32),
            method="stride",
            metadata={"stride": stride, "total_available": len(all_indices)},
        )
