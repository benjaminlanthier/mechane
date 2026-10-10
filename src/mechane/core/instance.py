from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from mechane.core.experiment import Experiment
from mechane.utils.io import atomic_write_json
from mechane.utils.serialization import to_jsonable


@dataclass
class InstanceOutput:
    inputs: dict[str, Any] = field(default_factory=dict)  # what was sampled / generated
    outputs: dict[str, Any] = field(default_factory=dict)  # what was measured
    meta: dict[str, Any] = field(default_factory=dict)  # extra bookkeeping (backend, ...)


class Instance(ABC):
    """One stochastic repetition of an `Experiment`. Subclasses implement `run()`."""

    def __init__(self, experiment: Experiment, instance_id: int) -> None:
        self.experiment = experiment
        self.instance_id = instance_id

    @abstractmethod
    def run(self) -> InstanceOutput | Mapping[str, Any]: ...

    @property
    def seed(self) -> int:
        return self.experiment.instance_seed(self.instance_id)

    @cached_property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    @property
    def dir(self) -> Path:
        return self.experiment.instance_dir(self.instance_id)

    @property
    def results_file(self) -> Path:
        return self.experiment.results_path(self.instance_id)

    @property
    def error_file(self) -> Path:
        return self.experiment.error_path(self.instance_id)

    def execute(self, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run and wrap the output into the on-disk record (does not write it)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        out = self.run()
        wall = time.perf_counter() - t0
        if not isinstance(out, InstanceOutput):
            out = InstanceOutput(outputs=dict(out))
        return {
            "instance_id": self.instance_id,
            "experiment_id": self.experiment.config.experiment_id,
            "seed": self.seed,
            "inputs": to_jsonable(out.inputs),
            "outputs": to_jsonable(out.outputs),
            "meta": {
                **(provenance or {}),
                "started_at": started.isoformat(),
                "wall_time_s": wall,
                **to_jsonable(out.meta),
            },
        }

    def save(self, record: dict[str, Any]) -> None:
        atomic_write_json(self.results_file, record)
