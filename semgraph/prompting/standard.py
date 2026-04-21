"""Standard caption prompt bundle — baseline prompts for capable VLMs.

The class is a thin wrapper: content lives in
``semgraph/prompting/data/standard.yaml`` (source of truth) and is loaded
via ``importlib.resources`` on demand.  :meth:`from_cfg` applies the C1
merge-override contract on top of the packaged defaults.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

from semgraph.prompting._cfg import _STRING_FIELDS, _select
from semgraph.prompting.base import PromptBundle, _packaged_data_path


class StandardPromptBundle(PromptBundle):
    """Baseline prompt bundle.  Used when a user specifies neither an
    override nor a custom file.  Content lives in
    :file:`semgraph/prompting/data/standard.yaml`.
    """

    bundle_id: ClassVar[str] = "standard"

    @classmethod
    def from_cfg(cls, cfg: Any = None) -> "StandardPromptBundle":
        """Build a bundle honoring the C1 override-precedence contract.

        Packaged defaults are always schema-validated via
        :meth:`PromptBundle.from_yaml`.  When *cfg* is ``None`` the defaults
        are returned untouched; otherwise the following fields may be
        merged from *cfg*:

        * ``caption`` / ``color`` / ``material`` / ``consolidation`` —
          a non-empty, non-whitespace string replaces the packaged default.
          Empty and whitespace-only strings are silently ignored (cannot
          accidentally wipe a prompt).
        * ``suggested_top_k`` — any non-``None`` value (including ``0``)
          overrides the packaged hint; ``None``/missing preserves it.

        All reads go through :func:`_select` so a struct-mode ``DictConfig``
        upstream does not crash on missing keys.
        """
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
                overrides["suggested_top_k"] = hint  # validator will reject
            else:
                try:
                    overrides["suggested_top_k"] = int(hint)
                except (TypeError, ValueError):
                    overrides["suggested_top_k"] = hint  # validator rejects

        return replace(base, **overrides) if overrides else base


__all__ = ["StandardPromptBundle"]
