"""PromptBundle abstract base class + schema-version dispatch.

The ABC is the single source of truth for:

* Validation rules (:meth:`PromptBundle.validate_raw` — C2).  Every concrete
  or third-party bundle goes through the same ruleset because
  :meth:`__post_init__` delegates to ``validate_raw(asdict(self))``.  A new
  rule is added here once; it cannot drift between the programmatic and the
  CLI ``validate`` paths.

* Schema-version dispatch (:meth:`from_yaml`).  Concrete subclasses declare
  fields and default content; they do **not** carry per-version loaders.  A
  future schema bump adds ``_load_v2`` to this ABC, leaving every already-
  shipped internal and third-party bundle working indefinitely.

The module-level :data:`SCHEMA_VERSION` pins the newest schema this build
understands; a user YAML declaring a version higher than this fails with an
"upgrade semgraph" error rather than a confusing ``KeyError`` inside a
loader.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import ClassVar, Optional

import yaml


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _find_foreign_placeholders(text: str, allowed: set[str]) -> set[str]:
    """Return the set of ``{name}``-style placeholders in *text* that are
    NOT in the *allowed* whitelist.

    Unescaped-literal curlies like ``{{"key": ...}}`` produce no matches on
    the inner token because ``str.format`` treats ``{{`` / ``}}`` as escapes;
    the regex here mirrors that by only matching bare ``{name}`` tokens. The
    caller is responsible for keeping JSON literals in the consolidation
    prompt escaped as ``{{...}}``.
    """
    found = {m.group(1) for m in _PLACEHOLDER_RE.finditer(text)}
    return found - allowed


def _packaged_data_path(bundle_id: str) -> Path:
    """Resolve ``semgraph/prompting/data/<bundle_id>.yaml`` via
    :mod:`importlib.resources`.

    Works in editable installs, wheel installs, and zipped packages. A
    :class:`FileNotFoundError` here means the ``[tool.setuptools.package-data]``
    block in :file:`pyproject.toml` is missing or misconfigured.
    """
    resource = resources.files("semgraph.prompting") / "data" / f"{bundle_id}.yaml"
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(
            f"Packaged YAML not found for bundle '{bundle_id}' at {path}. "
            f"If this is a wheel/docker install, verify pyproject.toml has "
            f"[tool.setuptools.package-data] 'semgraph.prompting' = ['data/*.yaml']"
        )
    return path


# --------------------------------------------------------------------------- #
# Validation report                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationReport:
    """Result of :meth:`PromptBundle.validate_raw`.

    *warnings* is a future-extension slot populated by forthcoming soft
    rules (e.g. "consolidation longer than X characters is an anti-pattern
    for small VLMs"). Kept in the schema now so CLI consumers that render
    ``report.warnings`` don't need to adapt on the next minor bump.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# PromptBundle ABC                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PromptBundle(ABC):
    """Abstract base class for caption-stage prompt bundles.

    Concrete subclasses declare a ``bundle_id`` class variable and
    implement :meth:`from_cfg`.  Everything else — YAML loading, schema
    dispatch, validation, content hashing — is inherited from this class.

    Instances are ``frozen=True``; to mutate use :func:`dataclasses.replace`.
    """

    bundle_id: ClassVar[str] = "base"
    """Subclasses override this; used by the factory and for hash logging."""

    SCHEMA_VERSION: ClassVar[int] = 1
    """Newest schema version this build of semgraph can read.  A user YAML
    declaring ``schema_version > SCHEMA_VERSION`` fails with an
    "upgrade semgraph" error."""

    caption: str = ""
    color: str = ""
    material: str = ""
    consolidation: str = ""
    suggested_top_k: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        """Run :meth:`validate_raw` on our own fields and raise if not ok.

        This is the C2 "single SoT" guarantee: programmatic construction
        and YAML-loaded construction hit the exact same ruleset, because
        both paths end here.
        """
        report = type(self).validate_raw(asdict(self))
        if not report.ok:
            raise ValueError(
                f"[{self.bundle_id}] {'; '.join(report.errors)}"
            )

    @classmethod
    def validate_raw(cls, raw: Mapping) -> ValidationReport:
        """Single source of truth for every bundle rule.

        Rules enforced:

        1. No unknown top-level keys (beyond the documented whitelist).
        2. Each of ``caption`` / ``color`` / ``material`` / ``consolidation``
           is a non-empty string no larger than 8 KiB (UTF-8).
        3. ``consolidation`` contains the literal ``{captions}`` placeholder
           and no other bare placeholders.
        4. ``caption`` / ``color`` / ``material`` contain no bare placeholders
           (prevents accidental injection like ``{object_id}``).
        5. Placeholder checks short-circuit on empty fields so the error
           output names the real problem ("empty") not a redundant
           ("missing placeholder") follow-on.
        """
        errors: list[str] = []
        warnings: list[str] = []

        allowed_top_keys = {
            "caption",
            "color",
            "material",
            "consolidation",
            "suggested_top_k",
            "schema_version",
            "bundle_id",
        }
        unknown = set(raw) - allowed_top_keys
        if unknown:
            errors.append(f"unknown top-level keys: {sorted(unknown)}")

        field_is_nonempty: dict[str, bool] = {}
        for name in ("caption", "color", "material", "consolidation"):
            val = raw.get(name)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"'{name}' must be a non-empty string")
                field_is_nonempty[name] = False
            elif len(val.encode("utf-8")) > 8 * 1024:
                errors.append(f"'{name}' exceeds 8 KiB size cap")
                field_is_nonempty[name] = True
            else:
                field_is_nonempty[name] = True

        consolidation = raw.get("consolidation", "")
        if isinstance(consolidation, str) and field_is_nonempty.get("consolidation", False):
            if "{captions}" not in consolidation:
                errors.append(
                    "consolidation must contain '{captions}' placeholder"
                )
            foreign = _find_foreign_placeholders(
                consolidation, allowed={"captions"}
            )
            if foreign:
                errors.append(
                    f"consolidation has foreign placeholders: {sorted(foreign)}"
                )

        for name in ("caption", "color", "material"):
            val = raw.get(name, "")
            if isinstance(val, str) and field_is_nonempty.get(name, False):
                foreign = _find_foreign_placeholders(val, allowed=set())
                if foreign:
                    errors.append(
                        f"'{name}' has disallowed placeholders: {sorted(foreign)}"
                    )

        top_k = raw.get("suggested_top_k")
        if top_k is not None and not isinstance(top_k, bool):
            try:
                k = int(top_k)
            except (TypeError, ValueError):
                errors.append(
                    f"'suggested_top_k' must be int or null, got {top_k!r}"
                )
            else:
                if k < 0:
                    errors.append(
                        f"'suggested_top_k' must be >= 0, got {k}"
                    )
        elif isinstance(top_k, bool):
            errors.append("'suggested_top_k' must be int or null, not bool")

        return ValidationReport(ok=not errors, errors=errors, warnings=warnings)

    @classmethod
    def validate_file(cls, path: Path) -> ValidationReport:
        """Thin wrapper: parse the YAML (utf-8) and forward to
        :meth:`validate_raw`."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            return ValidationReport(
                ok=False,
                errors=[f"YAML root must be a mapping, got {type(raw).__name__}"],
            )
        return cls.validate_raw(raw)

    # ------------------------------------------------------------------ #
    # Schema-version dispatch                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, path: Path) -> "PromptBundle":
        """Load a bundle from a YAML file with schema-version dispatch.

        The file's ``schema_version`` (defaulting to ``1`` when omitted) is
        coerced to ``int`` per C4.  The ABC looks up ``_load_vN`` on
        ``cls`` and invokes it; concrete subclasses therefore do not carry
        per-version loaders.  A version higher than :attr:`SCHEMA_VERSION`
        yields a clear upgrade-semgraph error rather than an ``AttributeError``.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Bundle YAML root must be a mapping, got {type(raw).__name__}"
            )

        version_raw = raw.get("schema_version", 1)
        if isinstance(version_raw, bool):
            raise ValueError(
                f"schema_version must be int, got {version_raw!r}"
            )
        try:
            v = int(version_raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"schema_version must be int, got {version_raw!r}"
            )

        if v > cls.SCHEMA_VERSION:
            raise ValueError(
                f"Bundle schema_version={v}, this semgraph supports up to "
                f"v{cls.SCHEMA_VERSION}. Upgrade semgraph."
            )

        loader = getattr(cls, f"_load_v{v}", None)
        if loader is None:
            raise ValueError(
                f"Unsupported schema_version={v} (no _load_v{v} on {cls.__name__})"
            )

        # Run validate_raw on the *original* YAML dict so rules that apply
        # to keys *not* in our field list (e.g. unknown-top-level-key) are
        # enforced. __post_init__ also calls validate_raw on asdict(self),
        # but by then the unknown keys have been dropped by _load_vN.
        report = cls.validate_raw(raw)
        if not report.ok:
            raise ValueError(
                f"[{getattr(cls, 'bundle_id', cls.__name__)}] "
                f"{'; '.join(report.errors)}"
            )

        return cls(**loader(raw))

    @classmethod
    def _load_v1(cls, raw: Mapping) -> dict:
        """v1 schema loader: flat ``{caption, color, material, consolidation,
        suggested_top_k}``."""
        return {
            "caption": raw.get("caption", ""),
            "color": raw.get("color", ""),
            "material": raw.get("material", ""),
            "consolidation": raw.get("consolidation", ""),
            "suggested_top_k": raw.get("suggested_top_k"),
        }

    # ------------------------------------------------------------------ #
    # Content hashing                                                     #
    # ------------------------------------------------------------------ #

    @property
    def content_sha256(self) -> str:
        """Stable hex digest of the bundle's content, including
        ``suggested_top_k``.

        Two bundles with identical prompt strings but different
        ``suggested_top_k`` values hash distinctly.  This is intentional:
        the hash log at the run header ("exact bundle that ran") treats a
        changed hint as a bundle change.  Documented alongside the
        hash-log line in :doc:`docs/VLLM_API.md </VLLM_API>`.
        """
        data = asdict(self)
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


__all__ = [
    "PromptBundle",
    "ValidationReport",
    "_find_foreign_placeholders",
    "_packaged_data_path",
]
