"""Find a Laboratory class from a string, with no import side effects and no closed enum.

A reference is either a dotted path ("package.module:ClassName") or the name of an entry point
in the group "mechane.laboratories". Manifests store the dotted path, so any machine that can
import your package can resume a sweep.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mechane.laboratory import Laboratory

ENTRY_POINT_GROUP = "mechane.laboratories"


def laboratory_ref(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def resolve_laboratory(ref: str) -> type[Laboratory]:
    if ":" in ref:
        module_name, _, attr = ref.partition(":")
        obj = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj  # type: ignore[return-value]
    entry_points = {ep.name: ep for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}
    if ref not in entry_points:
        raise KeyError(
            f"Unknown laboratory {ref!r}. Use 'module:Class' or register an entry point in "
            f"group {ENTRY_POINT_GROUP!r}. Registered: {sorted(entry_points)}"
        )
    return entry_points[ref].load()
