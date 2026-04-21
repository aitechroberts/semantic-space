"""Custom caption prompt bundle — user-supplied YAML with guardrails.

Resolution flow (C3):

1. CLI override: ``caption_prompts.prompts_path=<value>`` (or the unsugared
   ``caption_prompts.prompts_package`` + ``caption_prompts.prompts_resource``
   pair).
2. Env fallback: ``CAPTION_PROMPTS_FILE=/abs/path/to/my.yaml``.
3. Neither set: raise a :class:`ValueError` that lists all three options
   (CLI / env / ``pkg://``).

Path classification after resolution:

* ``pkg://<package>/<resource_path>`` — resolved via
  :mod:`importlib.resources`.  Skips repo-confinement because any
  ``importlib.resources`` path is necessarily inside an installed package
  (the allowlist is structural).
* Anything else — treated as a filesystem path.  Must be inside the
  detected repo root, or the ``trust_path`` cfg / ``CAPTION_PROMPTS_TRUST``
  env var must be truthy per :func:`semgraph.prompting._cfg._truthy`
  (case-folded whitelist, default-deny).

Six guardrails on the final file (in order):

1. ``yaml.safe_load`` (no tag execution).
2. Path confinement or ``_truthy`` trust flag (filesystem only).
3. Whitelisted top-level keys (via :meth:`PromptBundle.validate_raw`).
4. Placeholder audit (via :meth:`PromptBundle.validate_raw`).
5. Size caps: 8 KiB per template (via ``validate_raw``) and 64 KiB per
   file (enforced here before load).
6. SHA-256 hash logged at INFO level so runs are reproducible.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

from semgraph.prompting._cfg import _STRING_FIELDS, _select, _truthy
from semgraph.prompting.base import PromptBundle

logger = logging.getLogger(__name__)

_PKG_SCHEME = "pkg://"
_MAX_FILE_BYTES = 64 * 1024


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_repo_root() -> Path:
    """Best-effort repo root detection.

    Looks for a ``pyproject.toml`` walking upward from this module. Falls
    back to the parent of the ``semgraph`` package directory (the typical
    editable-install layout). Returns an absolute, resolved path.
    """
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent.resolve()
    return here.parents[2].resolve()


def _resolve_pkg_scheme(raw_path: str) -> Path:
    """Resolve a ``pkg://<package>/<resource>`` path via importlib.resources.

    Raises :class:`ValueError` if the package or resource is missing, so
    the user sees a clear error instead of a FileNotFoundError later.
    """
    spec = raw_path[len(_PKG_SCHEME):]
    if "/" not in spec:
        raise ValueError(
            f"pkg:// path must be 'pkg://<package>/<resource_path>', got {raw_path!r}"
        )
    package, resource_rel = spec.split("/", 1)
    if not package or not resource_rel:
        raise ValueError(
            f"pkg:// path must have non-empty package and resource, got {raw_path!r}"
        )
    try:
        root = resources.files(package)
    except (ModuleNotFoundError, TypeError) as exc:
        raise ValueError(
            f"pkg:// package {package!r} is not installed or not importable: {exc}"
        ) from exc

    target = root
    for part in resource_rel.split("/"):
        target = target / part

    path = Path(str(target)).resolve()
    if not path.is_file():
        raise ValueError(
            f"pkg:// resource not found: {raw_path!r} "
            f"(resolved to {path!r})"
        )
    return path


def _resolve_filesystem_path(raw_path: str, *, trust: bool) -> Path:
    """Resolve a filesystem path with repo-confinement or trust gate.

    When *trust* is ``False`` the path must be inside the repo root
    detected by :func:`_detect_repo_root`.  When *trust* is ``True`` any
    existing absolute path is accepted (the operator has explicitly opted
    into loading prompts from outside the repo).
    """
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Custom prompts file does not exist: {path}")

    if trust:
        return path

    repo_root = _detect_repo_root()
    try:
        path.relative_to(repo_root)
    except ValueError:
        raise ValueError(
            f"Custom prompts path {path} is outside the repo root "
            f"{repo_root}. Set CAPTION_PROMPTS_TRUST=1 (or "
            f"caption_prompts.trust_path=1) to load from an external "
            f"location, or pass a pkg:// URI for packaged prompts."
        )
    return path


class CustomPromptBundle(PromptBundle):
    """User-supplied prompt bundle loaded from a YAML file or package
    resource.  See module docstring for the resolution flow and guardrails.
    """

    bundle_id: ClassVar[str] = "custom"

    @classmethod
    def from_cfg(cls, cfg: Any = None) -> "CustomPromptBundle":
        """Resolve, validate, and load a custom bundle.

        The three-tier resolution + path classification + guardrails live
        here (and only here) so the ``custom.yaml`` Hydra pointer can stay
        minimal and Python owns the authoritative logic.  String / hint
        overrides from *cfg* (the same C1 contract the internal bundles
        use) are applied on top of the YAML-loaded bundle when present.
        """
        cli_val = _select(cfg, "prompts_path")
        if cli_val and str(cli_val) not in ("???",):
            raw_path = str(cli_val)
        elif os.environ.get("CAPTION_PROMPTS_FILE"):
            raw_path = os.environ["CAPTION_PROMPTS_FILE"]
        else:
            # Check two-field unsugared form.
            pkg = _select(cfg, "prompts_package")
            res = _select(cfg, "prompts_resource")
            if pkg and res:
                raw_path = f"{_PKG_SCHEME}{pkg}/{res}"
            else:
                raise ValueError(
                    "CustomPromptBundle requires a prompts file. Provide one of:\n"
                    "  CLI:  caption_prompts.prompts_path=/abs/path/to/my.yaml\n"
                    "  env:  export CAPTION_PROMPTS_FILE=/abs/path/to/my.yaml\n"
                    "  pkg:  caption_prompts.prompts_path=pkg://your_package/prompts/file.yaml"
                )

        if raw_path.startswith(_PKG_SCHEME):
            path = _resolve_pkg_scheme(raw_path)
        else:
            trust = _truthy(_select(cfg, "trust_path")) or _truthy(
                os.environ.get("CAPTION_PROMPTS_TRUST")
            )
            path = _resolve_filesystem_path(raw_path, trust=trust)

        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ValueError(
                f"Custom prompts file too large: {size} bytes "
                f"(cap {_MAX_FILE_BYTES} bytes)"
            )

        bundle = cls.from_yaml(path)

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
        if overrides:
            bundle = replace(bundle, **overrides)

        digest = _sha256_file(path)
        logger.info(
            "[prompt_bundle] loaded custom bundle from %s (sha256=%s)",
            path,
            digest[:16],
        )
        return bundle


__all__ = ["CustomPromptBundle"]
