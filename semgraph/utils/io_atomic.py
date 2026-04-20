"""Atomic file-write helpers for Phase B artifacts.

``json.dump`` is not atomic: a SIGKILL mid-write leaves a truncated file
that ``json.load`` explodes on the next run.  Use ``atomic_write_json``
for every new JSON artifact written during long sweeps (captions,
scene_graph, vetted_questions, predictions, summary, attribute_vocab).

The write-then-rename pattern relies on POSIX ``os.replace`` being atomic
within the same filesystem.  Callers are responsible for placing the tmp
file on the same filesystem as the target — passing a ``Path`` under the
same directory tree is sufficient.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    """Serialize ``obj`` to ``path`` via tmp file + atomic rename.

    Any pre-existing ``.tmp`` leftover from a crashed prior write is
    overwritten.  The parent directory is created if missing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # fsync may fail on some virtual filesystems; rename is still
            # atomic, we just lose the durability guarantee.
            pass
    os.replace(tmp, p)


def touch_done(path: str | Path) -> None:
    """Write a ``.done`` marker file (idempotent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def is_done(path: str | Path) -> bool:
    """True if a ``.done`` marker file exists."""
    return Path(path).is_file()
