from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from importlib.metadata import EntryPoint

import pytest

from mechane.core.registry import ENTRY_POINT_GROUP


@pytest.fixture
def register_laboratory(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Make `mechane.laboratories` entry points visible without installing a package.

    Usage: `register_laboratory(toy="toy:ToyLab")`.
    """

    def register(**refs: str) -> None:
        entry_points = [
            EntryPoint(name=name, value=value, group=ENTRY_POINT_GROUP)
            for name, value in refs.items()
        ]

        def fake_entry_points(**selection: str) -> list[EntryPoint]:
            return entry_points if selection.get("group") == ENTRY_POINT_GROUP else []

        monkeypatch.setattr(metadata, "entry_points", fake_entry_points)

    return register