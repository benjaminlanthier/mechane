"""Finding and reading manifests.

A manifest is the JSON file `setup` writes once; every later job rebuilds exactly the work it
needs from it. This module reads the file and turns it back into the right `Laboratory`
subclass. The manifest's *content* is produced and consumed by `Laboratory.manifest()` and
`Laboratory.from_manifest()`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechane.core.registry import resolve_laboratory

if TYPE_CHECKING:
    from mechane.core.laboratory import Laboratory


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with manifest_path.open() as f:
        return json.load(f)


def load_laboratory(manifest_path: Path) -> Laboratory:
    """Rebuild the right Laboratory subclass from a manifest alone."""
    manifest = read_manifest(manifest_path)
    ref = manifest.get("laboratory") or manifest["experiment_type"]  # older manifests: type name
    return resolve_laboratory(ref).from_manifest(manifest_path)
