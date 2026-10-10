"""Deterministic JSON conversion and hashing.

`stable_hash` is byte-for-byte the same function the original `experiments`
package used (SHA-256 of canonical JSON, first 24 hex chars), so result trees
created before the refactor keep resolving to the same directories.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively convert `obj` into plain JSON types (enums -> values, dataclasses -> dicts,
    numpy scalars/arrays -> Python numbers/lists, paths -> str)."""
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {
            (k if isinstance(k, str) else str(to_jsonable(k))): to_jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy scalars and arrays
        return to_jsonable(obj.tolist())
    raise TypeError(f"Cannot serialize {type(obj).__name__!r} to JSON: {obj!r}")


def stable_hash(data: Any, length: int = 24) -> str:
    blob = json.dumps(to_jsonable(data), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:length]
