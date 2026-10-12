"""Re-running one instance exactly, without touching its stored results.

Seeds follow from the configuration alone, so a replay sees the same random numbers as the
original run. The instance runs against a scratch copy of its experiment: anything it writes into
its own directory lands under the scratch root, never next to the real results.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mechane.core.laboratory import Laboratory


class ReplayError(ValueError):
    """The requested experiment or instance does not exist."""


#: The parts of a record that a replay is expected to reproduce (`meta` is about the machine).
COMPARED_FIELDS = ("seed", "inputs", "outputs")


@dataclass(frozen=True)
class ReplayResult:
    record: dict[str, Any]  # what would have been written to results.json
    saved: dict[str, Any] | None  # the stored record, if the instance had finished
    differences: tuple[str, ...]  # empty when they agree, or when there is nothing to compare to

    @property
    def reproduced(self) -> bool | None:
        """True/False once there is a stored result to compare with, else None."""
        return None if self.saved is None else not self.differences


def replay_instance(
    lab: Laboratory,
    experiment_id: int,
    instance_id: int,
    scratch_root: Path,
    rtol: float = 0.0,
) -> ReplayResult:
    """Run `instance_id` of `experiment_id` under `scratch_root` and compare with its stored result
    (numbers within relative tolerance `rtol`, exact by default)."""
    try:
        exp = lab.experiment(experiment_id)
    except KeyError as exc:
        raise ReplayError(str(exc.args[0])) from exc
    n = exp.config.num_instances
    if not 0 <= instance_id < n:
        raise ReplayError(
            f"instance_id {instance_id} is out of range for experiment {experiment_id} "
            f"(valid: 0..{n - 1})"
        )
    scratch = type(exp)(scratch_root, exp.config)
    record = lab.instance_class(scratch, instance_id).execute(provenance=lab.provenance())

    stored = exp.results_path(instance_id)
    if not stored.exists():
        return ReplayResult(record, None, ())
    saved = json.loads(stored.read_text())
    return ReplayResult(record, saved, tuple(compare_records(record, saved, rtol)))


def compare_records(replay: dict[str, Any], saved: dict[str, Any], rtol: float = 0.0) -> list[str]:
    """Human-readable differences between a replayed and a stored record."""
    differences: list[str] = []
    for field in COMPARED_FIELDS:
        _diff(replay.get(field), saved.get(field), field, rtol, differences)
    return differences


def _is_number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _close(a: float, b: float, rtol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=0.0) if rtol > 0 else a == b


def _diff(a: Any, b: Any, path: str, rtol: float, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys(), key=str):
            if key not in a:
                out.append(f"{path}.{key}: only in the stored result")
            elif key not in b:
                out.append(f"{path}.{key}: only in the replay")
            else:
                _diff(a[key], b[key], f"{path}.{key}", rtol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: replay has {len(a)} items, stored result has {len(b)}")
        else:
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                _diff(x, y, f"{path}[{index}]", rtol, out)
    elif _is_number(a) and _is_number(b):
        if not _close(a, b, rtol):
            out.append(f"{path}: replay {a!r} != stored {b!r}")
    elif a != b:
        out.append(f"{path}: replay {a!r} != stored {b!r}")
