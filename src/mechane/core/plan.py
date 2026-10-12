"""Planning SLURM array jobs: which instances each array task runs.

A *plan* splits the work into tasks of at most `batch_size` instances of one experiment. Saved as
JSON it is the contract between submission time and run time: array task `N` always runs the same
instances, even when other tasks finish in the meantime (which would shift an "unfinished only"
plan that was recomputed inside each job).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechane.utils.io import atomic_write_json

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment
    from mechane.core.laboratory import Laboratory

PLAN_VERSION = 1


class PlanError(ValueError):
    """A plan that cannot be built, read, or indexed."""


@dataclass(frozen=True)
class Task:
    experiment_id: int
    instance_ids: tuple[int, ...]


@dataclass(frozen=True)
class Plan:
    manifest_path: Path
    batch_size: int
    tasks: tuple[Task, ...]

    @property
    def num_instances(self) -> int:
        return sum(len(task.instance_ids) for task in self.tasks)

    @property
    def array_spec(self) -> str:
        """The value for `sbatch --array=`: `"0-4"`, `"0"` for a single task, `""` for none."""
        if not self.tasks:
            return ""
        return "0" if len(self.tasks) == 1 else f"0-{len(self.tasks) - 1}"

    def task(self, task_id: int) -> Task:
        if not 0 <= task_id < len(self.tasks):
            raise PlanError(
                f"task id {task_id} is out of range: the plan has {len(self.tasks)} task(s)"
            )
        return self.tasks[task_id]


def build_plan(
    lab: Laboratory,
    batch_size: int,
    experiments: Sequence[Experiment] | None = None,
    unfinished_only: bool = False,
) -> Plan:
    """Chunk the instances of `experiments` (default: all) into tasks of `batch_size`.

    With `unfinished_only`, only instances that are missing or failed are planned, so a plan
    resubmits exactly the work that is still to do.
    """
    if batch_size < 1:
        raise PlanError(f"batch_size must be at least 1, got {batch_size}")
    chosen = lab.experiments if experiments is None else experiments
    tasks: list[Task] = []
    for exp in sorted(chosen, key=lambda e: e.config.experiment_id):
        if unfinished_only:
            status = exp.status()
            ids = sorted(status["failed"] + status["missing"])
        else:
            ids = list(range(exp.config.num_instances))
        for start in range(0, len(ids), batch_size):
            tasks.append(Task(exp.config.experiment_id, tuple(ids[start : start + batch_size])))
    return Plan(lab.manifest_path.resolve(), batch_size, tuple(tasks))


def plan_to_dict(plan: Plan) -> dict[str, Any]:
    return {
        "version": PLAN_VERSION,
        "manifest_path": str(plan.manifest_path),
        "batch_size": plan.batch_size,
        "tasks": [
            {"experiment_id": t.experiment_id, "instance_ids": list(t.instance_ids)}
            for t in plan.tasks
        ],
    }


def write_plan(plan: Plan, path: Path) -> None:
    atomic_write_json(Path(path), plan_to_dict(plan))


def read_plan(path: Path) -> Plan:
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise PlanError(f"No plan at {path}") from None
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != PLAN_VERSION:
        raise PlanError(f"{path} is not a mechane plan (expected version {PLAN_VERSION})")
    try:
        tasks = tuple(
            Task(int(t["experiment_id"]), tuple(int(i) for i in t["instance_ids"]))
            for t in data["tasks"]
        )
        return Plan(Path(data["manifest_path"]), int(data["batch_size"]), tasks)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"{path} is a malformed plan: {exc!r}") from exc
