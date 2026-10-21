from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any


@contextmanager
def _atomic_file(path: Path) -> Generator[IO[str]]:
    """Open a sibling temp file for writing and `os.replace` it onto `path` once it is complete,
    so readers never observe a half-written file (important on shared filesystems and when jobs
    get pre-empted). The temp file is removed if anything goes wrong."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            yield f
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write `data` as JSON, atomically (see `_atomic_file`)."""
    with _atomic_file(path) as f:
        json.dump(data, f, indent=indent)


def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` (UTF-8), atomically (see `_atomic_file`)."""
    with _atomic_file(path) as f:
        f.write(text)
