"""Results as a flat table: one row per finished instance, one column per scalar.

Columns are `experiment_id`, `instance_id`, `seed`, then the experiment's parameters (`model.L`),
then the outputs (`outputs.msd`), and on request the inputs (`inputs.*`) and run metadata
(`meta.*`). Nested outputs are flattened with dots; lists stay lists, for the writer to encode.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mechane.core.selection import parameters_of
from mechane.utils.serialization import flatten

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment


@dataclass(frozen=True)
class Table:
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


def _prefixed(prefix: str, value: Any) -> dict[str, Any]:
    return flatten({prefix: value if isinstance(value, dict) else {}})


def build_table(
    experiments: Iterable[Experiment], include_inputs: bool = False, include_meta: bool = False
) -> Table:
    rows: list[dict[str, Any]] = []
    for exp in experiments:
        parameters = parameters_of(exp)
        for instance_id, record in exp.iter_results():
            row: dict[str, Any] = {
                "experiment_id": exp.config.experiment_id,
                "instance_id": instance_id,
                "seed": record.get("seed"),
                **parameters,
                **_prefixed("outputs", record.get("outputs")),
            }
            if include_inputs:
                row.update(_prefixed("inputs", record.get("inputs")))
            if include_meta:
                row.update(_prefixed("meta", record.get("meta")))
            rows.append(row)
    columns = tuple(dict.fromkeys(key for row in rows for key in row))
    return Table(columns, tuple(rows))
