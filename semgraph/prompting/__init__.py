"""semgraph.prompting — pluggable caption-stage prompt bundles.

Factory
-------

``get_prompt_bundle(bundle_id, cfg=None)`` returns a
:class:`~semgraph.prompting.base.PromptBundle` instance configured for the
caption stage.  Internal bundles (``standard``, ``rich``, ``compact``) are
registered via ``[project.entry-points."semgraph.prompt_bundles"]`` in
:file:`pyproject.toml` and flow through the same entry-point discovery path
as third-party plugins.  ``custom`` is routed directly because it has no
registerable class; it is parameterized by a user YAML.

Design invariants
-----------------

**Unified code path.** Every bundle — internal or third-party — is resolved
by :func:`_discover_plugins` and then constructed via ``target.from_cfg(cfg)``.
There is no production-time if-chain for internal bundles; registration is
dogfooded from day one.

**Structural shadow-check-in-discovery (C-style preemption).** Internal
names are owned by the first-party ``semantic-space`` distribution
regardless of ``importlib.metadata`` iteration order — which is **not**
guaranteed stable across ``pip``/``uv``/``conda`` installs.  When a third
party registers one of the :data:`_INTERNAL` names from a non-first-party
dist, :func:`_discover_plugins` logs a ``WARNING`` naming both dists and
overrides the cache slot with a direct import of the first-party class.
The defense-in-depth post-loop pass guarantees every non-``custom``
internal name resolves to a first-party class even if the first-party
``pyproject.toml`` entry-point block is somehow missing.

**Broken plugins do not kill the factory.** Per-entry-point ``except
Exception`` stores a :class:`_FailedPlugin` sentinel; the factory raises a
focused :class:`RuntimeError` only when the user specifically asks for the
broken ``bundle_id``.

**Cache is test-clearable.** The discovery loop is memoized via
:func:`functools.lru_cache`.  Tests that install/uninstall plugins call
``_discover_plugins.cache_clear()`` in setup/teardown.  Editable install
(``pip install -e .``) is required before ``pytest`` for entry-point tests
to see the first-party registrations.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Union

from semgraph.prompting.base import PromptBundle, ValidationReport

logger = logging.getLogger(__name__)


_INTERNAL: frozenset[str] = frozenset(
    {"standard", "default", "rich", "compact", "custom"}
)
"""Reserved bundle IDs owned by the first-party distribution.

A third-party plugin that registers any of these via
``semgraph.prompt_bundles`` gets a WARNING logged and the cache slot
overridden with the first-party class.  Exported so tests and the CLI can
share the single source of truth."""

_FIRST_PARTY_DIST: str = "semantic-space"
"""Distribution name of the first-party package (matches
``[project].name`` in :file:`pyproject.toml`).  Used by the structural
shadow-check inside :func:`_discover_plugins`."""


@dataclass(frozen=True)
class _FailedPlugin:
    """Sentinel placed in the discovery cache when a plugin import fails.

    :func:`get_prompt_bundle` raises a focused :class:`RuntimeError`
    (chained via ``from exc``) only when the failed bundle is specifically
    requested; every other bundle keeps working.
    """

    ep_name: str
    dist_name: str
    exc: BaseException


def _first_party_class_for(bundle_id: str) -> type[PromptBundle]:
    """Direct-import fallback when overriding a third-party shadow.

    Kept separate from :func:`_discover_plugins` so tests can monkeypatch
    the discovery cache without accidentally short-circuiting this helper.

    Raises :class:`KeyError` for ``custom`` (not registerable) and any
    unknown name.
    """
    if bundle_id in ("standard", "default"):
        from semgraph.prompting.standard import StandardPromptBundle

        return StandardPromptBundle
    if bundle_id == "rich":
        from semgraph.prompting.rich import RichPromptBundle

        return RichPromptBundle
    if bundle_id == "compact":
        from semgraph.prompting.compact import CompactPromptBundle

        return CompactPromptBundle
    raise KeyError(bundle_id)


@functools.lru_cache(maxsize=1)
def _discover_plugins() -> dict[str, Union[type[PromptBundle], _FailedPlugin]]:
    """Load every ``semgraph.prompt_bundles`` entry point into a dict.

    Internal-name preemption is **structural**: each entry point's
    ``ep.dist.name`` is compared to :data:`_FIRST_PARTY_DIST`.  A non-first
    party registration on an :data:`_INTERNAL` name is logged and
    overridden with :func:`_first_party_class_for` regardless of iteration
    order.

    Broken plugins do not crash the factory; a :class:`_FailedPlugin`
    sentinel is stored in the cache and the loop continues.

    The post-loop defense-in-depth pass guarantees every non-``custom``
    internal name resolves to a first-party class even if the first-party
    entry-point block is missing or broken.
    """
    out: dict[str, Union[type[PromptBundle], _FailedPlugin]] = {}

    for ep in entry_points(group="semgraph.prompt_bundles"):
        try:
            dist = ep.dist
            dist_name = dist.name if dist is not None else "<unknown dist>"
        except Exception:
            dist_name = "<unknown dist>"

        if ep.name in _INTERNAL and dist_name != _FIRST_PARTY_DIST:
            logger.warning(
                "[prompt_bundle] third-party plugin %r from dist %r attempted to "
                "shadow internal bundle_id %r; using first-party class from %r instead",
                ep.name,
                dist_name,
                ep.name,
                _FIRST_PARTY_DIST,
            )
            try:
                out[ep.name] = _first_party_class_for(ep.name)
            except KeyError:
                pass
            continue

        try:
            out[ep.name] = ep.load()
        except Exception as exc:
            logger.warning(
                "[prompt_bundle] failed to load plugin %r from dist %r: %s",
                ep.name,
                dist_name,
                exc,
            )
            out[ep.name] = _FailedPlugin(ep.name, dist_name, exc)

    for name in _INTERNAL - {"custom"}:
        target = out.get(name)
        if target is None or isinstance(target, _FailedPlugin):
            try:
                out[name] = _first_party_class_for(name)
            except KeyError:
                continue

    return out


def get_prompt_bundle(bundle_id: str, cfg: Any = None) -> PromptBundle:
    """Return the :class:`PromptBundle` instance for *bundle_id*.

    ``custom`` is routed directly because it is parameterized by a user
    YAML rather than by a registered class.  Every other ``bundle_id``
    flows through the unified entry-point discovery path.  The string
    ``"default"`` is a synonym for ``"standard"`` to match the Hydra
    pointer filename.

    Raises :class:`ValueError` when *bundle_id* is unknown (the message
    lists every bundle visible to this installation).  Raises
    :class:`RuntimeError` when the requested plugin exists in the entry-
    point table but its import failed; other bundles continue to work.
    """
    if bundle_id == "custom":
        from semgraph.prompting.custom import CustomPromptBundle

        return CustomPromptBundle.from_cfg(cfg)

    plugins = _discover_plugins()
    lookup_id = "standard" if bundle_id == "default" else bundle_id
    target = plugins.get(lookup_id)

    if isinstance(target, _FailedPlugin):
        raise RuntimeError(
            f"Plugin {target.ep_name!r} from dist {target.dist_name!r} "
            f"failed to load: {target.exc!r}"
        ) from target.exc

    if target is None:
        raise ValueError(
            f"Unknown bundle_id {bundle_id!r}. "
            f"Valid: {sorted(set(plugins) | _INTERNAL)}"
        )

    return target.from_cfg(cfg)


__all__ = [
    "PromptBundle",
    "ValidationReport",
    "get_prompt_bundle",
    "_discover_plugins",
    "_FailedPlugin",
    "_first_party_class_for",
    "_INTERNAL",
    "_FIRST_PARTY_DIST",
]
