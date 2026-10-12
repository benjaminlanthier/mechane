"""Describing experiments by their parameters, so that an experiment id means something."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechane.core.selection import parameters_of

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment


@dataclass(frozen=True)
class ExperimentRow:
    experiment_id: int
    parameters: dict[str, Any]  # flat: {"model.L": 2, "solver.sigma": 1.0}
    content_hash: str
    path: Path  # the experiment's `instances` directory
    status: dict[str, int] | None = None  # {"done": n, "failed": n, "missing": n}


def describe(experiments: Iterable[Experiment], with_status: bool = False) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for exp in experiments:
        status = None
        if with_status:
            status = {name: len(ids) for name, ids in exp.status().items()}
        rows.append(
            ExperimentRow(
                experiment_id=exp.config.experiment_id,
                parameters=parameters_of(exp),
                content_hash=exp.config.content_hash(),
                path=exp.instances_dir,
                status=status,
            )
        )
    return rows


def varying_keys(rows: Sequence[ExperimentRow]) -> list[str]:
    """The parameters that are not the same in every row: the ones that tell experiments apart."""
    if not rows:
        return []
    return [
        key
        for key in rows[0].parameters
        if len({json.dumps(row.parameters[key], sort_keys=True) for row in rows}) > 1
    ]
