"""Running work: execute instances and persist what they produce.

Nothing here knows about terminals. Progress is reported as events through an optional listener
(`mechane.cli` turns them into text), so the same functions serve the command line, a notebook,
or any scheduler script that drives `run_batch` directly.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mechane.core.laboratory import Laboratory


@dataclass(frozen=True)
class InstanceStarted:
    laboratory: str
    experiment_id: int
    instance_id: int


@dataclass(frozen=True)
class InstanceSkipped:
    """The instance already had a results file and `overwrite` was off."""

    results_file: Path


@dataclass(frozen=True)
class InstanceSaved:
    results_file: Path


@dataclass(frozen=True)
class InstanceFailed:
    """Reported by `run_batch` for an instance that raised. Its traceback is in `job.err`."""

    instance_id: int
    error: Exception


type RunEvent = InstanceStarted | InstanceSkipped | InstanceSaved | InstanceFailed
type Listener = Callable[[RunEvent], None]


def _ignore(_event: RunEvent) -> None:
    """The default listener: say nothing."""


@dataclass(frozen=True)
class BatchResult:
    instance_ids: tuple[int, ...]
    failed: tuple[int, ...]


def run_instance(
    lab: Laboratory,
    experiment_id: int,
    instance_id: int,
    overwrite: bool = False,
    listener: Listener = _ignore,
) -> Path | None:
    """Run one instance and persist it. Returns the results path, or None if skipped.

    On failure the traceback is written to the instance's `job.err` and the exception is
    re-raised. A later success removes the stale `job.err`.
    """
    instance = lab.build_instance(experiment_id, instance_id)
    if not overwrite and instance.results_file.exists():
        listener(InstanceSkipped(instance.results_file))
        return None
    listener(InstanceStarted(lab.name, experiment_id, instance_id))
    try:
        record = instance.execute(provenance=lab.provenance())
        instance.save(record)
        instance.error_file.unlink(missing_ok=True)  # clear a stale traceback from an earlier try
    except Exception:
        instance.dir.mkdir(parents=True, exist_ok=True)
        instance.error_file.write_text(traceback.format_exc())
        raise
    listener(InstanceSaved(instance.results_file))
    return instance.results_file


def run_batch(
    lab: Laboratory,
    experiment_id: int,
    instance_ids: Iterable[int],
    overwrite: bool = False,
    continue_on_error: bool = True,
    listener: Listener = _ignore,
) -> BatchResult:
    """Run several instances of one experiment, sequentially, in this process.

    With `continue_on_error` (the default) a failing instance is reported and the rest still
    run; the failures are listed in the result. Otherwise the first error is re-raised.
    """
    ids = tuple(instance_ids)
    failed: list[int] = []
    for instance_id in ids:
        try:
            run_instance(lab, experiment_id, instance_id, overwrite=overwrite, listener=listener)
        except Exception as exc:
            failed.append(instance_id)
            listener(InstanceFailed(instance_id, exc))
            if not continue_on_error:
                raise
    return BatchResult(instance_ids=ids, failed=tuple(failed))
