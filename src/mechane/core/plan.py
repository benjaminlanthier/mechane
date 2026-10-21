"""Planning tasks: which instances each unit of scheduled work runs.

A *plan* splits the work into tasks of at most `batch_size` instances of one experiment. Saved as
JSON it is the contract between planning time and run time: task `N` always runs the same
instances, even when other tasks finish in the meantime (which would shift an "unfinished only"
plan that was recomputed inside each job).

The file groups tasks by experiment, and writes each task's instances as a range string
(`"0-999"`, `"3,7-9"`), one batch per line, so a plan stays small however many instances there
are. `num_tasks` states how many tasks there are in all, for workflow tools that only need to
count them; task `N` is the Nth batch in file order. Earlier versions are still read: version 1
listed every instance id of every task, version 2 one range per task.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechane.utils.io import atomic_write_text
from mechane.utils.ranges import compact_ranges, parse_ranges

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment
    from mechane.core.laboratory import Laboratory

PLAN_VERSION = 3  # 1: every instance id of every task. 2: one range per task. Both still read.
READABLE_VERSIONS = (1, 2, PLAN_VERSION)


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

    def task(self, task_id: int) -> Task:
        if not 0 <= task_id < len(self.tasks):
            raise _out_of_range(task_id, len(self.tasks))
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
    groups = [
        {"experiment_id": experiment_id, "batches": [compact_ranges(t.instance_ids) for t in tasks]}
        for experiment_id, tasks in groupby(plan.tasks, key=lambda t: t.experiment_id)
    ]
    return {
        "version": PLAN_VERSION,
        "manifest_path": str(plan.manifest_path),
        "batch_size": plan.batch_size,
        "num_tasks": len(plan.tasks),
        "tasks": groups,
    }


def plan_to_json(plan: Plan) -> str:
    """The plan as JSON text, one batch per line: the file format, and what `plan --format json`
    prints. (`json.dumps(indent=2)` would put every list element on its own line.)"""
    data = plan_to_dict(plan)
    groups = []
    for group in data.pop("tasks"):
        batches = ",\n".join(f"      {json.dumps(batch)}" for batch in group["batches"])
        opening = f'    {{"experiment_id": {json.dumps(group["experiment_id"])}, "batches": ['
        groups.append(f"{opening}\n{batches}\n    ]}}")
    head = json.dumps(data, indent=2)[:-2]  # without the closing "\n}"
    body = '"tasks": [\n' + ",\n".join(groups) + "\n  ]" if groups else '"tasks": []'
    return f"{head},\n  {body}\n}}\n"


def write_plan(plan: Plan, path: Path) -> None:
    atomic_write_text(Path(path), plan_to_json(plan))


def _load(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PlanError(f"No plan at {path}") from None
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") not in READABLE_VERSIONS:
        raise PlanError(f"{path} is not a mechane plan (expected version {PLAN_VERSION})")
    return data


def _count_tasks(data: dict[str, Any]) -> int:
    """How many tasks a loaded plan has. Version 3 states it (`num_tasks`): it must agree with
    the batches it summarises, so that a hand-edited file cannot say one thing and mean another."""
    if data["version"] < 3:
        return len(data["tasks"])
    count = sum(len(group["batches"]) for group in data["tasks"])
    if data["num_tasks"] != count:
        raise ValueError(f"num_tasks is {data['num_tasks']!r} but the file lists {count} task(s)")
    return count


def _task_from_entry(entry: dict[str, Any]) -> Task:
    """One task of a version 1 or 2 file: a list of ids, or a range string."""
    if "instances" in entry:
        ids = parse_ranges(entry["instances"])
    else:
        ids = [int(i) for i in entry["instance_ids"]]
    return Task(int(entry["experiment_id"]), tuple(ids))


def _all_tasks(data: dict[str, Any]) -> Iterator[Task]:
    if data["version"] < 3:
        for entry in data["tasks"]:
            yield _task_from_entry(entry)
        return
    for group in data["tasks"]:
        for batch in group["batches"]:
            yield Task(int(group["experiment_id"]), tuple(parse_ranges(batch)))


def _nth_task(data: dict[str, Any], task_id: int) -> Task:
    """Task `task_id` of a loaded plan. The other tasks are counted, never parsed."""
    count = _count_tasks(data)
    if not 0 <= task_id < count:
        raise _out_of_range(task_id, count)
    if data["version"] < 3:
        return _task_from_entry(data["tasks"][task_id])
    groups = data["tasks"]
    ends = list(accumulate(len(group["batches"]) for group in groups))  # tasks up to each group
    index = bisect_right(ends, task_id)
    offset = task_id - (ends[index - 1] if index else 0)
    group = groups[index]
    return Task(int(group["experiment_id"]), tuple(parse_ranges(group["batches"][offset])))


def _out_of_range(task_id: int, count: int) -> PlanError:
    return PlanError(f"task id {task_id} is out of range: the plan has {count} task(s)")


def read_plan(path: Path) -> Plan:
    """The whole plan, every task expanded to its instance ids."""
    data = _load(path)
    try:
        _count_tasks(data)
        tasks = tuple(_all_tasks(data))
        return Plan(Path(data["manifest_path"]), int(data["batch_size"]), tasks)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"{path} is a malformed plan: {exc!r}") from exc


def read_task(path: Path, task_id: int) -> tuple[Path, Task]:
    """One task of a plan, and the manifest it belongs to, without expanding the other tasks.

    This is what an array task needs: the cost of starting task N does not grow with the number
    of tasks, or of instances, in the plan.
    """
    data = _load(path)
    try:
        return Path(data["manifest_path"]), _nth_task(data, task_id)
    except PlanError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"{path} is a malformed plan: {exc!r}") from exc
