"""Prompt-bundle CLI: ``python -m semgraph.prompting {list,show,validate}``.

Subcommands
-----------

``list``
    Enumerate every registered ``bundle_id`` (internal + third-party).
    Broken plugins are flagged but do not fail the command.

``show <bundle_id>``
    Print the resolved bundle content (after packaged defaults apply).

``validate <path>``
    Run the full :meth:`PromptBundle.validate_raw` ruleset against a
    user-supplied YAML and exit ``0`` if the report is ok, ``1`` otherwise.
    Error lines and warning lines are printed one per line to stderr / stdout
    so the command is drop-in usable as a pre-commit hook.

The CLI intentionally does not accept arbitrary ``cfg`` overrides; every
interaction stays at the bundle layer.  Stage-level cfg sweeps belong in
Hydra.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from semgraph.prompting import (
    _FailedPlugin,
    _discover_plugins,
    _INTERNAL,
    get_prompt_bundle,
)
from semgraph.prompting.base import PromptBundle


def _cmd_list(_args: argparse.Namespace) -> int:
    plugins = _discover_plugins()
    names = sorted(set(plugins) | _INTERNAL)
    for name in names:
        target = plugins.get(name)
        if isinstance(target, _FailedPlugin):
            print(f"{name}\t[BROKEN: {target.dist_name}: {target.exc!r}]")
        else:
            print(name)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    bundle: PromptBundle = get_prompt_bundle(args.bundle_id)
    print(f"bundle_id:       {bundle.bundle_id}")
    print(f"schema_version:  {bundle.SCHEMA_VERSION}")
    print(f"suggested_top_k: {bundle.suggested_top_k}")
    print(f"content_sha256:  {bundle.content_sha256[:16]}")
    print("")
    data = asdict(bundle)
    for key in ("caption", "color", "material", "consolidation"):
        print(f"--- {key} ---")
        print(data[key])
        print("")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        print(f"[validate] file not found: {path}", file=sys.stderr)
        return 1
    report = PromptBundle.validate_file(path)
    for warn in report.warnings:
        print(f"[warn] {warn}")
    if report.ok:
        print(f"[validate] OK {path}")
        return 0
    for err in report.errors:
        print(f"[error] {err}", file=sys.stderr)
    print(f"[validate] FAIL {path}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m semgraph.prompting",
        description="Inspect and validate semgraph prompt bundles.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List every registered bundle_id.")

    show = sub.add_parser("show", help="Print a resolved bundle.")
    show.add_argument("bundle_id", help="Bundle id (e.g. 'standard', 'rich').")

    validate = sub.add_parser(
        "validate",
        help="Validate a YAML file against the prompt-bundle schema.",
    )
    validate.add_argument("path", help="Path to a bundle YAML to validate.")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "list":
        return _cmd_list(args)
    if args.cmd == "show":
        return _cmd_show(args)
    if args.cmd == "validate":
        return _cmd_validate(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
