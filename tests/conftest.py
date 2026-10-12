from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import metadata
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from mechane import grid
from mechane.core.registry import ENTRY_POINT_GROUP
from mechane.core.runner import run_batch
from toy import ToyLab

SWEEP_PARAMS = {
    "seed": 7,
    "num_instances": 6,
    "model": grid(L=[2, 3]),
    "solver": {"sigma": 1.0, "n_samples": 20},
}


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


@pytest.fixture
def lab(tmp_path: Path) -> ToyLab:
    """Two experiments (L = 2 and 3) of six instances each, set up on disk, nothing run yet."""
    laboratory = ToyLab.from_configs(tmp_path / "results", SWEEP_PARAMS)
    assert isinstance(laboratory, ToyLab)
    laboratory.setup()
    return laboratory


@pytest.fixture
def run_instances() -> Callable[[ToyLab, int, Sequence[int]], None]:
    """`run_instances(lab, experiment_id, [0, 1, 2])` runs those instances for real."""

    def run(laboratory: ToyLab, experiment_id: int, instance_ids: Sequence[int]) -> None:
        assert not run_batch(laboratory, experiment_id, instance_ids).failed

    return run


@pytest.fixture
def fail_instance() -> Callable[..., Path]:
    """`fail_instance(lab, experiment_id, instance_id, text)` leaves a job.err, like a failure."""

    def fail(laboratory: ToyLab, experiment_id: int, instance_id: int, text: str = "boom") -> Path:
        path = laboratory.experiment(experiment_id).error_path(instance_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    return fail
