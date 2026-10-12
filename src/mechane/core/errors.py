"""Failed instances, grouped by what went wrong.

A failed instance is one without results but with a `job.err` (exactly what `Experiment.status`
calls failed). Its traceback's last line, the exception, is the signature; instances sharing a
signature are one problem, however many there are.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment

EMPTY_SIGNATURE = "<empty job.err>"


@dataclass(frozen=True)
class FailedInstance:
    experiment_id: int
    instance_id: int
    error_file: Path

    def read_traceback(self) -> str:
        return self.error_file.read_text(errors="replace")


@dataclass(frozen=True)
class FailureGroup:
    signature: str
    instances: tuple[FailedInstance, ...]

    @property
    def count(self) -> int:
        return len(self.instances)


def error_signature(traceback_text: str) -> str:
    """The last non-empty line of a traceback (the exception and its message)."""
    lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    return lines[-1] if lines else EMPTY_SIGNATURE


def collect_failures(experiments: Iterable[Experiment]) -> list[FailureGroup]:
    """Failure groups, the most frequent first (ties broken by signature)."""
    groups: dict[str, list[FailedInstance]] = defaultdict(list)
    for exp in experiments:
        for instance_id in exp.status()["failed"]:
            failed = FailedInstance(
                exp.config.experiment_id, instance_id, exp.error_path(instance_id)
            )
            groups[error_signature(failed.read_traceback())].append(failed)
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    return [FailureGroup(signature, tuple(items)) for signature, items in ordered]
