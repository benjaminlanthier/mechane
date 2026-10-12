"""`mechane.core.plan`: splitting work into SLURM array tasks."""

import json
from pathlib import Path

import pytest

from mechane.core.plan import (
    PLAN_VERSION,
    Plan,
    PlanError,
    Task,
    build_plan,
    plan_to_dict,
    read_plan,
    write_plan,
)


def test_every_instance_is_planned_once_in_batches(lab):
    plan = build_plan(lab, batch_size=4)

    assert plan.manifest_path == lab.manifest_path.resolve()
    assert plan.batch_size == 4
    # 6 instances per experiment, in batches of 4: the last batch of each is partial.
    assert plan.tasks == (
        Task(0, (0, 1, 2, 3)),
        Task(0, (4, 5)),
        Task(1, (0, 1, 2, 3)),
        Task(1, (4, 5)),
    )
    assert plan.num_instances == 12


@pytest.mark.parametrize(("n_tasks_wanted", "spec"), [(0, ""), (1, "0"), (2, "0-1"), (4, "0-3")])
def test_array_spec(n_tasks_wanted, spec):
    tasks = tuple(Task(0, (i,)) for i in range(n_tasks_wanted))
    assert Plan(Path("m.json"), 1, tasks).array_spec == spec


def test_a_selection_of_experiments_is_planned_in_id_order(lab):
    plan = build_plan(lab, 6, experiments=[lab.experiment(1), lab.experiment(0)])
    assert [t.experiment_id for t in plan.tasks] == [0, 1]
    assert [t.experiment_id for t in build_plan(lab, 6, experiments=[lab.experiment(1)]).tasks] == [
        1
    ]


def test_unfinished_only_plans_missing_and_failed_instances(lab, run_instances, fail_instance):
    run_instances(lab, 0, [0, 1, 3])  # done: 0, 1, 3; missing: 2, 4, 5
    fail_instance(lab, 0, 4)  # failed, so it must be retried too

    plan = build_plan(lab, batch_size=2, unfinished_only=True)

    assert plan.tasks[:2] == (Task(0, (2, 4)), Task(0, (5,)))  # arbitrary ids, not a range
    assert [t for t in plan.tasks if t.experiment_id == 1] == [
        Task(1, (0, 1)),
        Task(1, (2, 3)),
        Task(1, (4, 5)),
    ]


def test_a_finished_experiment_contributes_no_tasks(lab, run_instances):
    run_instances(lab, 0, range(6))
    plan = build_plan(lab, 3, unfinished_only=True)
    assert {t.experiment_id for t in plan.tasks} == {1}


def test_a_batch_size_below_one_is_rejected(lab):
    with pytest.raises(PlanError, match="batch_size must be at least 1"):
        build_plan(lab, 0)


def test_task_lookup_is_bounds_checked(lab):
    plan = build_plan(lab, 6)
    assert plan.task(1) == Task(1, (0, 1, 2, 3, 4, 5))
    for bad in (-1, 2):
        with pytest.raises(PlanError, match=r"task id .* out of range: the plan has 2 task\(s\)"):
            plan.task(bad)


def test_a_plan_survives_a_round_trip_through_a_file(lab, tmp_path):
    plan = build_plan(lab, 4)
    path = tmp_path / "plan.json"

    write_plan(plan, path)

    assert read_plan(path) == plan
    assert json.loads(path.read_text()) == plan_to_dict(plan)
    assert plan_to_dict(plan)["version"] == PLAN_VERSION


def test_reading_a_missing_plan_names_the_path(tmp_path):
    with pytest.raises(PlanError, match=r"No plan at .*nope\.json"):
        read_plan(tmp_path / "nope.json")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "is not valid JSON"),
        ("[1, 2]", "is not a mechane plan"),
        (json.dumps({"version": 99, "tasks": []}), "is not a mechane plan"),
        (json.dumps({"version": PLAN_VERSION}), "is a malformed plan"),
        (
            json.dumps(
                {
                    "version": PLAN_VERSION,
                    "manifest_path": "m.json",
                    "batch_size": 1,
                    "tasks": [{"experiment_id": "x", "instance_ids": [0]}],
                }
            ),
            "is a malformed plan",
        ),
    ],
)
def test_malformed_plans_are_plan_errors(tmp_path, content, message):
    path = tmp_path / "plan.json"
    path.write_text(content)
    with pytest.raises(PlanError, match=message):
        read_plan(path)
