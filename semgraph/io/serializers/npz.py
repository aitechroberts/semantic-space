"""NpzSerializer — np.savez_compressed + JSON sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from semgraph.io.serializers.base import BaseSerializer


class NpzSerializer(BaseSerializer):
    """Write arrays as compressed ``.npz`` and metadata as ``.json``."""

    def save(
        self,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
        path: Path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, default=str)

    def load(self, path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        path = Path(path)
        arrays = dict(np.load(path.with_suffix(".npz"), allow_pickle=False))
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)
        return arrays, metadata
