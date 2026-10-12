"""Removing results, tracebacks and debris, so that instances can run again.

Nothing is deleted here by accident: `find_targets` only lists what a clean would remove, and
`remove_targets` removes exactly that list.

  * `errors`:  `job.err` files. A failed instance becomes a missing one, ready to rerun.
  * `results`: `results.json` (and any `job.err`), which makes finished instances run again.
  * `stray`:   whole instance directories past `num_instances` (see `diagnose`).
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment

CleanKind = Literal["errors", "results", "stray"]


@dataclass(frozen=True)
class CleanTarget:
    kind: CleanKind
    experiment_id: int
    instance_id: int
    path: Path  # a file, or for "stray" a directory


def find_targets(
    experiments: Iterable[Experiment], kind: CleanKind, instance_ids: Sequence[int] | None = None
) -> list[CleanTarget]:
    """What cleaning `kind` would remove. `instance_ids` narrows "errors" and "results" (ids
    outside an experiment's range are ignored); it does not apply to "stray"."""
    targets: list[CleanTarget] = []
    for exp in experiments:
        eid, n = exp.config.experiment_id, exp.config.num_instances
        if kind == "stray":
            if exp.instances_dir.is_dir():
                stray = sorted(
                    (int(p.name), p)
                    for p in exp.instances_dir.iterdir()
                    if p.is_dir() and p.name.isdigit() and int(p.name) >= n
                )
                targets += [CleanTarget(kind, eid, i, path) for i, path in stray]
            continue
        wanted = range(n) if instance_ids is None else [i for i in instance_ids if 0 <= i < n]
        for i in wanted:
            if kind == "results" and exp.results_path(i).exists():
                targets.append(CleanTarget(kind, eid, i, exp.results_path(i)))
            if exp.error_path(i).exists():
                targets.append(CleanTarget(kind, eid, i, exp.error_path(i)))
    return targets


def remove_targets(targets: Iterable[CleanTarget]) -> int:
    """Delete the targets and return how many were removed."""
    removed = 0
    for target in targets:
        if target.path.is_dir():
            shutil.rmtree(target.path)
        else:
            target.path.unlink(missing_ok=True)
        removed += 1
    return removed
