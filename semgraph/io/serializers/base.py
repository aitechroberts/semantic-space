"""Abstract base for all serializers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseSerializer(ABC):
    """Read/write contract: arrays go to a binary store, metadata to a sidecar."""

    @abstractmethod
    def save(
        self,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
        path: Path,
    ) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...
