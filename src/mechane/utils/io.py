from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON so readers never observe a half-written file (important on shared filesystems
    and when jobs get pre-empted): write to a sibling temp file, then `os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
