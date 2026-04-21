"""Shared cfg helpers for prompt bundles.

Three small utilities the concrete bundle implementations consume in common:

* :data:`_STRING_FIELDS` — the ordered tuple of prompt-string fields. Both
  :mod:`semgraph.prompting.standard` / :mod:`~.rich` / :mod:`~.compact`
  iterate over this when merging Hydra cfg overrides in ``from_cfg``.
* :func:`_select` (C10) — struct-mode-safe config reader. If a user (or a
  future upstream compose step) enables ``OmegaConf.set_struct(cfg, True)``,
  a plain ``cfg.get("prompts_path")`` raises ``ConfigAttributeError`` on a
  missing key. ``_select`` uses ``OmegaConf.select`` for ``DictConfig`` and
  falls back to ``.get`` for plain ``Mapping`` inputs, returning the default
  in both cases.
* :func:`_truthy` (C9) — case-folded whitelist for trust-gate env vars and
  cfg values. Default-deny for anything outside ``{1, true, yes, on}``. The
  closes the classic silent-bypass footgun where ``bool("0") == True``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_STRING_FIELDS: tuple[str, ...] = ("caption", "color", "material", "consolidation")


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    """Struct-mode-safe ``cfg[key]`` with a default.

    Works with ``OmegaConf.DictConfig`` (including struct mode), plain
    ``dict``, or any other ``Mapping`` with ``.get``.  Returns ``default``
    when ``cfg`` is ``None`` or the key is missing.
    """
    if cfg is None:
        return default

    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        DictConfig = None  # type: ignore[assignment]
        OmegaConf = None  # type: ignore[assignment]

    if DictConfig is not None and isinstance(cfg, DictConfig):
        return OmegaConf.select(cfg, key, default=default)

    if isinstance(cfg, Mapping):
        return cfg.get(key, default)

    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            return default

    return default


def _truthy(val: Any) -> bool:
    """Case-folded whitelist truthiness (C9 — default-deny).

    Returns ``True`` only when *val* is ``True`` (``bool``) or, when coerced
    to ``str`` and case-folded, is one of ``{"1", "true", "yes", "on"}``.

    Every other value — including ``"0"``, ``"false"``, ``"no"``, ``"off"``,
    ``""``, ``None``, random strings like ``"maybe"``/``"enabled"``, and
    non-bool ints/floats — returns ``False``. This is the **only** coercion
    permitted for trust-like env/cfg values; callers must never use bare
    ``bool()`` or ``int(val)`` truthiness on these inputs.
    """
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    try:
        text = str(val).strip().lower()
    except Exception:
        return False
    return text in {"1", "true", "yes", "on"}


__all__ = ["_STRING_FIELDS", "_select", "_truthy"]
