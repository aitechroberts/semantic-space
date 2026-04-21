"""Rich caption prompt bundle — longer prompts for capable VLMs.

Shares the merge-override contract of :class:`StandardPromptBundle`; only
the packaged content differs.  Suggests ``top_k=5`` per-object views via
:attr:`PromptBundle.suggested_top_k` (non-binding hint).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

from semgraph.prompting._cfg import _STRING_FIELDS, _select
from semgraph.prompting.base import PromptBundle, _packaged_data_path


class RichPromptBundle(PromptBundle):
    bundle_id: ClassVar[str] = "rich"

    @classmethod
    def from_cfg(cls, cfg: Any = None) -> "RichPromptBundle":
        base = cls.from_yaml(_packaged_data_path(cls.bundle_id))
        if cfg is None:
            return base

        overrides: dict[str, Any] = {}
        for k in _STRING_FIELDS:
            val = _select(cfg, k)
            if isinstance(val, str) and val.strip():
                overrides[k] = val

        hint = _select(cfg, "suggested_top_k")
        if hint is not None:
            if isinstance(hint, bool):
                overrides["suggested_top_k"] = hint
            else:
                try:
                    overrides["suggested_top_k"] = int(hint)
                except (TypeError, ValueError):
                    overrides["suggested_top_k"] = hint

        return replace(base, **overrides) if overrides else base


__all__ = ["RichPromptBundle"]
