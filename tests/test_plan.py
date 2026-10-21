"""`mechane.core.plan`: splitting work into tasks."""

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
    plan_to_json,
    read_plan,
    read_task,
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
        with pytest.raises(
            PlanError, match=r"^task id -?\d+ is out of range: the plan has 2 task\(s\)"
        ):
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


# --------------------------------------------------------------------------- the range format
def test_a_plan_file_groups_tasks_by_experiment_one_batch_per_line(lab, tmp_path):
    path = tmp_path / "plan.json"
    write_plan(build_plan(lab, 4), path)

    text = path.read_text()

    assert PLAN_VERSION == 3
    assert '"num_tasks": 4' in text
    assert '    {"experiment_id": 0, "batches": [\n      "0-3",\n      "4-5"\n    ]},\n' in text
    assert '    {"experiment_id": 1, "batches": [\n      "0-3",\n      "4-5"\n    ]}\n' in text
    assert "instance_ids" not in text and '"instances"' not in text
    assert json.loads(text) == plan_to_dict(build_plan(lab, 4))


def test_a_plan_with_no_tasks_is_still_valid_json(lab):
    empty = build_plan(lab, 4, experiments=[])
    assert empty.tasks == ()
    assert json.loads(plan_to_json(empty)) == plan_to_dict(empty)
    assert '"tasks": []' in plan_to_json(empty) and '"num_tasks": 0' in plan_to_json(empty)


def test_scattered_instances_round_trip_as_ranges(lab, run_instances, tmp_path):
    run_instances(
        lab, 0, [0, 1, 4]
    )  # leaves 2-3 and 5 for experiment 0, so the ids are not contiguous
    plan = build_plan(lab, 3, unfinished_only=True)
    path = tmp_path / "plan.json"

    write_plan(plan, path)

    assert plan.tasks[0] == Task(0, (2, 3, 5))
    assert '"2-3,5"' in path.read_text()
    assert read_plan(path) == plan


def test_a_plan_stays_small_however_many_instances_there_are(tmp_path):
    from toy import ToyLab

    lab = ToyLab.from_configs(
        tmp_path, {"num_instances": 1_000_000, "model": {"L": 2}, "solver": {"sigma": 1.0}}
    )
    plan = build_plan(lab, 1000)

    text = plan_to_json(plan)

    assert len(plan.tasks) == 1000
    assert len(text) < 30_000  # one id per line took 16 MB; one range per task, 56 KB
    assert '      "999000-999999"\n' in text


# --------------------------------------------------------------------------- version 1 plans
def write_v1(path, tasks, manifest="m.json", batch_size=2):
    entries = [{"experiment_id": e, "instance_ids": ids} for e, ids in tasks]
    path.write_text(
        json.dumps(
            {"version": 1, "manifest_path": manifest, "batch_size": batch_size, "tasks": entries}
        )
    )


def test_a_version_1_plan_is_still_read(tmp_path):
    path = tmp_path / "plan.json"
    write_v1(path, [(0, [0, 1]), (1, [4, 5, 9])])

    plan = read_plan(path)

    assert plan.tasks == (Task(0, (0, 1)), Task(1, (4, 5, 9)))
    assert plan.manifest_path == Path("m.json")
    assert read_task(path, 1) == (Path("m.json"), Task(1, (4, 5, 9)))


# --------------------------------------------------------------------------- read_task
def test_read_task_returns_one_task_and_its_manifest(lab, tmp_path):
    plan = build_plan(lab, 4)
    path = tmp_path / "plan.json"
    write_plan(plan, path)

    manifest, task = read_task(path, 2)

    assert manifest == plan.manifest_path
    assert task == plan.tasks[2] == Task(1, (0, 1, 2, 3))


def test_read_task_does_not_expand_the_other_tasks(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "manifest_path": "m.json",
                "batch_size": 2,
                "tasks": [
                    {"experiment_id": 0, "instances": "0-1"},
                    {"experiment_id": 0, "instances": "not a range"},  # never looked at
                ],
            }
        )
    )

    assert read_task(path, 0) == (Path("m.json"), Task(0, (0, 1)))
    with pytest.raises(PlanError, match="is a malformed plan"):
        read_task(path, 1)
    with pytest.raises(PlanError, match="is a malformed plan"):
        read_plan(path)  # reading the whole plan does expand every task


def test_read_task_checks_the_task_id_and_the_file(lab, tmp_path):
    path = tmp_path / "plan.json"
    write_plan(build_plan(lab, 6), path)
    for bad in (-1, 2):
        with pytest.raises(
            PlanError, match=r"^task id -?\d+ is out of range: the plan has 2 task\(s\)"
        ):
            read_task(path, bad)
    with pytest.raises(PlanError, match=r"No plan at .*missing\.json"):
        read_task(tmp_path / "missing.json", 0)


@pytest.mark.parametrize(
    "content",
    [
        {"version": 2},  # no tasks, no manifest
        {"version": 2, "manifest_path": "m.json", "tasks": 5},  # tasks is not a list
        {"version": 2, "manifest_path": "m.json", "tasks": [{"experiment_id": 0}]},  # no instances
        {
            "version": 2,
            "manifest_path": "m.json",
            "tasks": [{"experiment_id": 0, "instances": "3-1"}],
        },
    ],
)
def test_read_task_reports_a_malformed_plan(tmp_path, content):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(content))
    with pytest.raises(PlanError, match="is a malformed plan"):
        read_task(path, 0)


# --------------------------------------------------------------------------- grouping
def tasks_e0_e1_e0():
    # An experiment may appear in several places: the order of the tasks is what matters.
    return (
        Task(0, (0, 1)),
        Task(0, (2, 3)),
        Task(1, (0, 1)),
        Task(2, (5,)),
        Task(2, (6,)),
        Task(0, (4,)),
    )


def test_consecutive_tasks_of_an_experiment_share_a_group_and_the_order_is_kept(tmp_path):
    plan = Plan(Path("/m.json"), 2, tasks_e0_e1_e0())

    groups = plan_to_dict(plan)["tasks"]

    assert [(g["experiment_id"], g["batches"]) for g in groups] == [
        (0, ["0-1", "2-3"]),
        (1, ["0-1"]),
        (2, ["5", "6"]),
        (0, ["4"]),  # experiment 0 again: a new group, so that the task order is preserved
    ]
    assert plan_to_dict(plan)["num_tasks"] == 6


def test_every_task_is_found_across_group_boundaries(tmp_path):
    plan = Plan(Path("/m.json"), 2, tasks_e0_e1_e0())
    path = tmp_path / "plan.json"
    write_plan(plan, path)

    assert read_plan(path) == plan
    for task_id, expected in enumerate(plan.tasks):  # first and last of each group, and between
        assert read_task(path, task_id) == (Path("/m.json"), expected)


def test_read_task_of_a_grouped_plan_does_not_expand_the_other_batches(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "manifest_path": "m.json",
                "batch_size": 2,
                "num_tasks": 3,
                "tasks": [
                    {"experiment_id": 0, "batches": ["0-1", "not a range"]},
                    {"experiment_id": 1, "batches": ["0-1"]},
                ],
            }
        )
    )

    assert read_task(path, 0) == (Path("m.json"), Task(0, (0, 1)))
    assert read_task(path, 2) == (Path("m.json"), Task(1, (0, 1)))
    with pytest.raises(PlanError, match="is a malformed plan"):
        read_task(path, 1)
    with pytest.raises(PlanError, match="is a malformed plan"):
        read_plan(path)


# --------------------------------------------------------------------------- version 2 plans
def test_a_version_2_plan_is_still_read(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "manifest_path": "m.json",
                "batch_size": 3,
                "tasks": [
                    {"experiment_id": 0, "instances": "0-2"},
                    {"experiment_id": 1, "instances": "3,7-8"},
                ],
            }
        )
    )

    assert read_plan(path).tasks == (Task(0, (0, 1, 2)), Task(1, (3, 7, 8)))
    assert read_task(path, 1) == (Path("m.json"), Task(1, (3, 7, 8)))
    with pytest.raises(PlanError, match=r"^task id 2 is out of range: the plan has 2 task\(s\)"):
        read_task(path, 2)


# --------------------------------------------------------------------------- num_tasks
def v3(**overrides):
    plan = {
        "version": 3,
        "manifest_path": "m.json",
        "batch_size": 2,
        "num_tasks": 2,
        "tasks": [{"experiment_id": 0, "batches": ["0-1", "2-3"]}],
    }
    return {**plan, **overrides}


def test_num_tasks_must_agree_with_the_batches(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(v3(num_tasks=5)))
    for read in (lambda: read_plan(path), lambda: read_task(path, 0)):
        with pytest.raises(PlanError, match="num_tasks is 5 but the file lists 2 task"):
            read()


@pytest.mark.parametrize(
    "content",
    [
        {k: v for k, v in v3().items() if k != "num_tasks"},  # num_tasks is part of the format
        v3(tasks=5),  # not a list of groups
        v3(tasks=[5]),  # a group that is not a mapping
        v3(tasks=[{"experiment_id": 0}], num_tasks=0),  # a group without batches
        v3(tasks=[{"experiment_id": 0, "batches": 5}], num_tasks=0),  # batches is not a list
        v3(tasks=[{"experiment_id": 0, "batches": ["3-1"]}], num_tasks=1),  # an empty range
        v3(tasks=[{"batches": ["0-1"]}], num_tasks=1),  # no experiment
    ],
)
def test_malformed_version_3_plans_are_plan_errors(tmp_path, content):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(content))
    with pytest.raises(PlanError, match="is a malformed plan"):
        read_task(path, 0)
