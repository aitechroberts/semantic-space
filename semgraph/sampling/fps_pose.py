"""FPSPoseSelector — farthest point sampling on SE(3) camera poses."""

from __future__ import annotations

from typing import Any

import numpy as np

from semgraph.sampling.base import FrameSelector, SelectionResult


class FPSPoseSelector(FrameSelector):
    """Select frames that maximise viewpoint diversity via greedy FPS.

    Each camera pose is represented as a 6D vector:
    ``[position * position_weight, forward_direction * direction_weight]``
    and standard greedy farthest-point sampling picks the most spatially
    diverse subset.
    """

    def select(
        self,
        poses: dict[int, np.ndarray],
        n_frames: int | None = None,
        *,
        position_weight: float = 1.0,
        direction_weight: float = 0.5,
        **kwargs: Any,
    ) -> SelectionResult:
        all_indices = sorted(poses.keys())
        n_available = len(all_indices)

        if n_frames is None:
            n_frames = n_available

        points = np.zeros((n_available, 6), dtype=np.float64)
        for i, idx in enumerate(all_indices):
            c2w = poses[idx]
            points[i, :3] = c2w[:3, 3] * position_weight
            points[i, 3:] = -c2w[:3, 2] * direction_weight

        selected_local = _greedy_fps(points, n_frames)
        result_indices = sorted(all_indices[i] for i in selected_local)

        return SelectionResult(
            frame_indices=np.array(result_indices, dtype=np.int32),
            method="fps_pose",
            metadata={
                "n_requested": n_frames,
                "position_weight": position_weight,
                "direction_weight": direction_weight,
                "total_available": n_available,
            },
        )


def _greedy_fps(points: np.ndarray, k: int) -> list[int]:
    """Greedy farthest point sampling in arbitrary-dimensional space."""
    n = len(points)
    if k >= n:
        return list(range(n))

    selected = [0]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        last = points[selected[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))

    return selected
