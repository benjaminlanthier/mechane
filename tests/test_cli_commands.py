"""The new commands through the command line: options, output, streams and exit codes. Their
logic is tested in the `core` test modules; here it is the wiring that is under test."""

import code
import csv
import io
import json
import re

import pytest
from click.testing import CliRunner

from mechane.cli import (
    clean,
    doctor,
    errors,
    export,
    init,
    list_experiments,
    main,
    plan,
    replay,
    run_plan,
    shell,
    stats,
    status,
)
from mechane.core.plan import build_plan, plan_to_dict, write_plan
from toy import ToyLab


def invoke(command, *args):
    return CliRunner().invoke(command, [str(a) for a in args])


def with_manifest(lab, *args):
    return ["--manifest-path", lab.manifest_path, *args]


@pytest.fixture
def failing_lab(tmp_path):
    """One experiment of four instances that all raise ValueError("negative sigma")."""
    lab = ToyLab.from_configs(
        tmp_path / "failing",
        {"num_instances": 4, "model": {"L": 2}, "solver": {"sigma": -1.0, "n_samples": 5}},
    )
    lab.setup()
    return lab


def set_wall_time(lab, experiment_id, instance_id, seconds):
    path = lab.experiment(experiment_id).results_path(instance_id)
    record = json.loads(path.read_text())
    record["meta"]["wall_time_s"] = seconds
    path.write_text(json.dumps(record))


# --------------------------------------------------------------------------- the command group
def test_every_command_is_registered():
    commands = "clean doctor errors export init list plan replay run-plan shell stats"
    commands += " aggregate run run-batch setup status"
    result = invoke(main, "--help")
    for name in commands.split():
        assert f"  {name} " in result.output


# --------------------------------------------------------------------------- plan
def test_plan_prints_a_summary_and_a_table(lab):
    result = invoke(plan, *with_manifest(lab, "--batch-size", 4))

    assert result.exit_code == 0, result.output
    assert "12 instance(s) in 4 task(s) of up to 4; run task N (0 to 3)" in result.stdout
    assert "task  experiment  count  instances" in result.stdout
    assert "0-3" in result.stdout and "4-5" in result.stdout


def test_plan_has_no_scheduler_specific_format(lab):
    result = invoke(plan, *with_manifest(lab, "--format", "array"))
    assert result.exit_code == 2  # the plan's JSON has everything a workflow tool needs


def test_plan_json_format_is_the_plan_file_format(lab):
    result = invoke(plan, *with_manifest(lab, "--batch-size", 4, "--format", "json"))
    assert json.loads(result.stdout) == plan_to_dict(build_plan(lab, 4))
    assert '"num_tasks": 4' in result.stdout
    assert '{"experiment_id": 0, "batches": [\n      "0-3",\n      "4-5"\n    ]}' in result.stdout


def test_a_saved_plan_is_what_run_plan_runs(lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    result = invoke(plan, *with_manifest(lab, "--batch-size", 4, "--output", plan_file))
    assert result.exit_code == 0 and plan_file.exists()

    result = invoke(run_plan, "--plan-file", plan_file, "--task-id", 1)

    assert result.exit_code == 0, result.output
    assert (
        "[BatchJob] Completed 2/2 instance(s)." in result.stdout
    )  # task 1: instances 4-5 of exp 0
    assert lab.experiment(0).status()["done"] == [4, 5]
    assert lab.experiment(1).status()["done"] == []


def test_plan_unfinished_covers_only_what_is_left(lab, run_instances):
    def plan_json():
        result = invoke(
            plan, *with_manifest(lab, "--batch-size", 4, "--unfinished", "--format", "json")
        )
        return json.loads(result.stdout)

    run_instances(lab, 0, range(6))
    only_experiment_1 = plan_json()
    assert only_experiment_1["num_tasks"] == 2  # two tasks, in one group
    assert [g["experiment_id"] for g in only_experiment_1["tasks"]] == [1]
    assert only_experiment_1["tasks"][0]["batches"] == ["0-3", "4-5"]

    run_instances(lab, 1, range(6))
    assert plan_json()["num_tasks"] == 0
    result = invoke(plan, *with_manifest(lab, "--unfinished"))
    assert "Nothing to run: the plan has no tasks." in result.stdout


@pytest.mark.parametrize(
    ("selection", "experiments"),
    [(["--experiment-id", 1], {1}), (["--where", "model.L=2"], {0})],
)
def test_plan_can_be_narrowed_to_some_experiments(lab, selection, experiments):
    result = invoke(plan, *with_manifest(lab, "--batch-size", 6, "--format", "json", *selection))
    assert {t["experiment_id"] for t in json.loads(result.stdout)["tasks"]} == experiments


def test_plan_reports_a_bad_filter_as_a_usage_error(lab):
    result = invoke(plan, *with_manifest(lab, "--where", "model.X=1"))
    assert result.exit_code == 2
    assert "unknown parameter 'model.X'" in result.stderr


# --------------------------------------------------------------------------- run-plan
def test_run_plan_reports_failures_and_exits_with_status_1(failing_lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    write_plan(build_plan(failing_lab, 2), plan_file)

    result = invoke(run_plan, "--plan-file", plan_file, "--task-id", 0)

    assert result.exit_code == 1
    assert "[BatchJob] 2/2 failed: [0, 1]" in result.stderr
    assert failing_lab.experiment(0).status()["failed"] == [0, 1]


def test_run_plan_can_fail_fast(failing_lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    write_plan(build_plan(failing_lab, 4), plan_file)

    result = invoke(run_plan, "--plan-file", plan_file, "--task-id", 0, "--fail-fast")

    assert isinstance(result.exception, ValueError)
    assert failing_lab.experiment(0).status() == {"done": [], "failed": [0], "missing": [1, 2, 3]}


def test_run_plan_skips_finished_work_unless_told_to_overwrite(lab, run_instances, tmp_path):
    plan_file = tmp_path / "plan.json"
    write_plan(build_plan(lab, 6), plan_file)
    run_instances(lab, 0, [0])

    skipped = invoke(run_plan, "--plan-file", plan_file, "--task-id", 0)
    assert "already exists, skipping" in skipped.stdout

    overwritten = invoke(run_plan, "--plan-file", plan_file, "--task-id", 0, "--overwrite")
    assert "already exists" not in overwritten.stdout
    assert overwritten.stdout.count("[Save]") == 6


def test_run_plan_still_runs_a_version_1_plan(lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "version": 1,
                "manifest_path": str(lab.manifest_path),
                "batch_size": 3,
                "tasks": [{"experiment_id": 1, "instance_ids": [0, 2, 4]}],
            }
        )
    )

    result = invoke(run_plan, "--plan-file", plan_file, "--task-id", 0)

    assert result.exit_code == 0, result.output
    assert lab.experiment(1).status()["done"] == [0, 2, 4]


def test_run_plan_runs_a_task_of_a_version_2_plan(lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "version": 2,
                "manifest_path": str(lab.manifest_path),
                "batch_size": 2,
                "tasks": [{"experiment_id": 1, "instances": "3-5"}],
            }
        )
    )

    assert invoke(run_plan, "--plan-file", plan_file, "--task-id", 0).exit_code == 0
    assert lab.experiment(1).status()["done"] == [3, 4, 5]


def test_run_plan_reads_only_its_own_task(lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "version": 3,
                "manifest_path": str(lab.manifest_path),
                "batch_size": 2,
                "num_tasks": 3,
                "tasks": [
                    {"experiment_id": 0, "batches": ["0-1"]},
                    {"experiment_id": 1, "batches": ["garbage", "0-1"]},
                ],
            }
        )
    )
    assert invoke(run_plan, "--plan-file", plan_file, "--task-id", 0).exit_code == 0
    assert invoke(run_plan, "--plan-file", plan_file, "--task-id", 2).exit_code == 0
    broken = invoke(run_plan, "--plan-file", plan_file, "--task-id", 1)
    assert broken.exit_code == 2 and "is a malformed plan" in broken.stderr


def test_run_plan_reads_only_its_own_task_in_a_flat_plan(lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        json.dumps(
            {
                "version": 2,
                "manifest_path": str(lab.manifest_path),
                "batch_size": 2,
                "tasks": [
                    {"experiment_id": 0, "instances": "0-1"},
                    {"experiment_id": 0, "instances": "garbage"},
                ],
            }
        )
    )

    assert invoke(run_plan, "--plan-file", plan_file, "--task-id", 0).exit_code == 0
    broken = invoke(run_plan, "--plan-file", plan_file, "--task-id", 1)
    assert broken.exit_code == 2 and "is a malformed plan" in broken.stderr


def test_run_plan_rejects_a_task_that_is_not_in_the_plan(lab, tmp_path):
    plan_file = tmp_path / "plan.json"
    write_plan(build_plan(lab, 6), plan_file)

    result = invoke(run_plan, "--plan-file", plan_file, "--task-id", 2)

    assert result.exit_code == 2
    assert "task id 2 is out of range: the plan has 2 task(s)" in result.stderr
    assert "malformed" not in result.stderr  # a bad task id is not a bad plan


def test_run_plan_rejects_a_file_that_is_not_a_plan(tmp_path):
    not_a_plan = tmp_path / "plan.json"
    not_a_plan.write_text("{}")
    result = invoke(run_plan, "--plan-file", not_a_plan, "--task-id", 0)
    assert result.exit_code == 2
    assert "is not a mechane plan" in result.stderr


# --------------------------------------------------------------------------- status --check
def test_status_check_succeeds_once_everything_is_done(lab, run_instances):
    for experiment_id in (0, 1):
        run_instances(lab, experiment_id, range(6))

    result = invoke(status, *with_manifest(lab, "--check"))

    assert result.exit_code == 0
    assert result.stderr == ""
    assert "experiment    1: 6 done, 0 failed, 0 missing" in result.stdout


def test_status_check_fails_while_anything_is_missing_or_failed(lab, run_instances, fail_instance):
    run_instances(lab, 0, range(6))
    run_instances(lab, 1, range(4))
    fail_instance(lab, 1, 4)  # instance 4 failed, instance 5 never ran

    result = invoke(status, *with_manifest(lab, "--check"))

    assert result.exit_code == 1
    assert "2 instance(s) are not done." in result.stderr
    assert "experiment    1: 4 done, 1 failed, 1 missing" in result.stdout  # still printed


def test_status_without_check_exits_0_whatever_the_state(lab):
    result = invoke(status, *with_manifest(lab))
    assert result.exit_code == 0
    assert "experiment    0: 0 done, 0 failed, 6 missing" in result.stdout


# --------------------------------------------------------------------------- errors
def test_errors_with_no_failures(lab):
    assert "No failed instances." in invoke(errors, *with_manifest(lab)).stdout


def test_errors_groups_failures_and_lists_examples(lab, fail_instance):
    for i in range(3):
        fail_instance(lab, 0, i, "Traceback\nMemoryError")
    fail_instance(lab, 1, 0, "Traceback\nValueError: negative sigma")

    result = invoke(errors, *with_manifest(lab, "--examples", 2))

    assert result.exit_code == 0
    assert "4 failed instance(s), 2 distinct error(s)" in result.stdout
    assert "    3x  MemoryError" in result.stdout
    assert "    1x  ValueError: negative sigma" in result.stdout
    assert "experiment 0 instance 1:" in result.stdout
    assert "experiment 0 instance 2:" not in result.stdout  # only two examples per error
    assert "... and 1 more" in result.stdout


def test_errors_can_show_the_first_traceback_of_each_error(lab, fail_instance):
    fail_instance(lab, 0, 0, 'Traceback (most recent call last):\n  File "x.py"\nMemoryError\n')

    result = invoke(errors, *with_manifest(lab, "--traceback"))

    assert "        | Traceback (most recent call last):" in result.stdout
    assert '        |   File "x.py"' in result.stdout


def test_errors_without_examples_still_counts_them(lab, fail_instance):
    fail_instance(lab, 0, 0, "Boom")
    result = invoke(errors, *with_manifest(lab, "--examples", 0))
    assert "    1x  Boom" in result.stdout and "... and 1 more" in result.stdout
    assert "instance 0:" not in result.stdout


def test_errors_can_be_narrowed_to_an_experiment(lab, fail_instance):
    fail_instance(lab, 0, 0, "A")
    fail_instance(lab, 1, 0, "B")

    result = invoke(errors, *with_manifest(lab, "--experiment-id", 1))

    assert "1 failed instance(s), 1 distinct error(s)" in result.stdout
    assert "    1x  B" in result.stdout
    assert "experiment 0" not in result.stdout


# --------------------------------------------------------------------------- doctor
def test_doctor_on_a_healthy_laboratory(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    result = invoke(doctor, *with_manifest(lab))
    assert result.exit_code == 0
    assert "No problems found." in result.stdout


def test_doctor_exits_1_on_errors(lab, run_instances):
    run_instances(lab, 0, [0])
    path = lab.experiment(0).results_path(0)
    record = json.loads(path.read_text())
    record["seed"] += 1
    path.write_text(json.dumps(record))

    result = invoke(doctor, *with_manifest(lab))

    assert result.exit_code == 1
    assert "ERROR   experiment 0: instance 0: the recorded seed or instance id" in result.stdout
    assert "1 error(s), 0 warning(s)" in result.stdout


def test_doctor_warnings_only_fail_with_strict(lab):
    (lab.experiment(1).instances_dir / "7").mkdir()

    relaxed = invoke(doctor, *with_manifest(lab))
    assert relaxed.exit_code == 0
    assert "WARNING experiment 1: instance directories 7 are outside 0..5" in relaxed.stdout
    assert "0 error(s), 1 warning(s)" in relaxed.stdout

    assert invoke(doctor, *with_manifest(lab, "--strict")).exit_code == 1


def test_doctor_labels_laboratory_wide_findings_and_skips_them_when_narrowed(lab):
    (lab.root / "orphan").mkdir()

    wide = invoke(doctor, *with_manifest(lab))
    assert "WARNING laboratory: 1 directory under" in wide.stdout

    narrowed = invoke(doctor, *with_manifest(lab, "--experiment-id", 0))
    assert "No problems found." in narrowed.stdout


# --------------------------------------------------------------------------- stats
def test_stats_table(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    set_wall_time(lab, 0, 0, 2.0)
    set_wall_time(lab, 0, 1, 4.0)

    result = invoke(stats, *with_manifest(lab))

    assert result.exit_code == 0, result.output
    assert "experiment  done  median  p95   max   total" in result.stdout
    assert "0           2/6   3.0s    3.9s  4.0s  6.0s" in result.stdout
    assert "1           0/6   -       -     -     0ms" in result.stdout
    assert "Total instance time: 6.0s" in result.stdout
    assert "Suggested" not in result.stdout  # only when --batch-size is given


def test_stats_suggests_a_time_limit_for_a_batch_size(lab, run_instances):
    run_instances(lab, 0, [0])
    set_wall_time(lab, 0, 0, 60.0)

    result = invoke(stats, *with_manifest(lab, "--batch-size", 4))
    assert (
        "Suggested time limit for 4 instance(s) per task: 5 min (slowest p95 x 4 x 1.25"
        in result.stdout
    )

    result = invoke(stats, *with_manifest(lab, "--batch-size", 4, "--margin", 2))
    assert "8 min (slowest p95 x 4 x 2" in result.stdout


def test_stats_without_timings_cannot_suggest_anything(lab):
    result = invoke(stats, *with_manifest(lab, "--batch-size", 4))
    assert "No timings yet" in result.stdout


def test_stats_can_include_disk_usage(lab, run_instances):
    run_instances(lab, 0, [0])

    result = invoke(stats, *with_manifest(lab, "--disk", "--experiment-id", 0))

    header, row = result.stdout.splitlines()[:2]
    assert header == "experiment  done  median  p95  max  total  files  size"
    assert re.search(r"  1  +\d+ B$", row)  # one results.json, a few hundred bytes


# --------------------------------------------------------------------------- list
def test_list_shows_only_the_parameters_that_differ(lab):
    result = invoke(list_experiments, *with_manifest(lab))
    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["id  model.L", "0   2", "1   3"]


def test_list_all_shows_every_parameter(lab):
    header = invoke(list_experiments, *with_manifest(lab, "--all")).stdout.splitlines()[0]
    for column in ("model.L", "model.kind", "solver.sigma", "solver.n_samples", "solver.device"):
        assert column in header


def test_list_a_single_experiment_shows_all_its_parameters(lab):
    header = invoke(
        list_experiments, *with_manifest(lab, "--experiment-id", 0)
    ).stdout.splitlines()[0]
    assert "model.kind" in header


def test_list_can_add_status_and_paths(lab, run_instances, fail_instance):
    run_instances(lab, 0, [0, 1])
    fail_instance(lab, 0, 2)

    result = invoke(list_experiments, *with_manifest(lab, "--status", "--paths"))

    first = result.stdout.splitlines()[1].split()
    assert result.stdout.splitlines()[0].split() == [
        "id",
        "model.L",
        "done",
        "failed",
        "missing",
        "path",
    ]
    assert first[:5] == ["0", "2", "2", "1", "3"]
    assert first[5] == str(lab.experiment(0).instances_dir)


def test_list_json(lab):
    result = invoke(list_experiments, *with_manifest(lab, "--json", "--where", "model.L=3"))
    (row,) = json.loads(result.stdout)
    assert row["experiment_id"] == 1 and row["parameters"]["model.L"] == 3
    assert row["path"] == str(lab.experiment(1).instances_dir)
    assert "status" not in row

    with_status = json.loads(
        invoke(list_experiments, *with_manifest(lab, "--json", "--status")).stdout
    )
    assert with_status[0]["status"] == {"done": 0, "failed": 0, "missing": 6}


def test_list_with_no_match(lab):
    result = invoke(list_experiments, *with_manifest(lab, "--where", "model.L=99"))
    assert "No experiments match." in result.stdout


def test_list_formats_non_string_values_as_json(tmp_path):
    from mechane import grid

    lab = ToyLab.from_configs(
        tmp_path / "r", {"model": grid(L=[2, 3]), "solver": {"sigma": 1.5, "n_samples": 5}}
    )
    lab.setup()
    result = invoke(list_experiments, *with_manifest(lab, "--where", "solver.sigma=1.5", "--all"))
    assert "1.5" in result.stdout and '"gauss"' not in result.stdout  # strings stay unquoted


# --------------------------------------------------------------------------- replay
def test_replay_prints_the_record_and_the_verdict_on_separate_streams(lab, run_instances):
    run_instances(lab, 0, [2])

    result = invoke(replay, *with_manifest(lab, "--experiment-id", 0, "--instance-id", 2))

    assert result.exit_code == 0, result.output
    shown = json.loads(result.stdout)  # stdout is only the JSON
    assert set(shown) == {"seed", "inputs", "outputs"}
    assert shown["seed"] == lab.experiment(0).instance_seed(2)
    assert "Reproduced" in result.stderr


def test_replay_exits_1_when_the_result_differs(lab, run_instances):
    run_instances(lab, 0, [0])
    path = lab.experiment(0).results_path(0)
    record = json.loads(path.read_text())
    record["outputs"]["mean_square"] = 999.0
    path.write_text(json.dumps(record))

    result = invoke(replay, *with_manifest(lab, "--experiment-id", 0, "--instance-id", 0))

    assert result.exit_code == 1
    assert "NOT reproduced" in result.stderr
    assert "outputs.mean_square: replay" in result.stderr
    assert "stored 999.0" in result.stderr


def test_replay_of_an_unfinished_instance_has_nothing_to_compare(lab):
    result = invoke(replay, *with_manifest(lab, "--experiment-id", 0, "--instance-id", 3))
    assert result.exit_code == 0
    assert "No stored result to compare with." in result.stderr


def test_replay_accepts_a_tolerance(lab, run_instances):
    run_instances(lab, 0, [0])
    path = lab.experiment(0).results_path(0)
    record = json.loads(path.read_text())
    record["outputs"]["mean_square"] *= 1 + 1e-9
    path.write_text(json.dumps(record))
    args = with_manifest(lab, "--experiment-id", 0, "--instance-id", 0)

    assert invoke(replay, *args).exit_code == 1
    assert invoke(replay, *args, "--rtol", 1e-6).exit_code == 0


@pytest.mark.parametrize(("experiment_id", "instance_id"), [(9, 0), (0, 6)])
def test_replay_reports_unknown_ids_as_usage_errors(lab, experiment_id, instance_id):
    result = invoke(
        replay, *with_manifest(lab, "--experiment-id", experiment_id, "--instance-id", instance_id)
    )
    assert result.exit_code == 2


def test_replay_can_keep_its_scratch_directory(lab, tmp_path):
    scratch = tmp_path / "scratch" / "here"
    result = invoke(
        replay,
        *with_manifest(lab, "--experiment-id", 0, "--instance-id", 1, "--scratch-dir", scratch),
    )
    assert result.exit_code == 0
    assert list(scratch.rglob("*"))  # the replay's directory is still there


def test_replay_opens_the_debugger_only_when_asked(failing_lab, monkeypatch):
    opened = []
    monkeypatch.setattr("pdb.post_mortem", lambda tb: opened.append(tb))
    args = with_manifest(failing_lab, "--experiment-id", 0, "--instance-id", 0)

    plain = invoke(replay, *args)
    assert isinstance(plain.exception, ValueError) and opened == []

    debugged = invoke(replay, *args, "--pdb")
    assert isinstance(debugged.exception, ValueError)  # still raised after the debugger
    assert len(opened) == 1 and opened[0] is not None


# --------------------------------------------------------------------------- shell
def test_shell_starts_an_interactive_session_on_the_laboratory(lab, monkeypatch):
    started = {}

    def fake_interact(banner, local):
        started.update(banner=banner, local=local)

    monkeypatch.setattr(code, "interact", fake_interact)

    result = invoke(shell, *with_manifest(lab))

    assert result.exit_code == 0, result.output
    assert started["local"]["lab"].num_experiments == 2
    assert "toy, 2 experiments x 6 instances" in started["banner"]
    assert "Available: lab, experiments, experiment, build_instance" in started["banner"]


# --------------------------------------------------------------------------- export
def rows_of(result):
    return list(csv.DictReader(io.StringIO(result.stdout)))


def test_export_writes_csv_to_standard_output(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    run_instances(lab, 1, [0])

    result = invoke(export, *with_manifest(lab))

    assert result.exit_code == 0, result.output
    rows = rows_of(result)
    assert [(r["experiment_id"], r["instance_id"], r["model.L"]) for r in rows] == [
        ("0", "0", "2"),
        ("0", "1", "2"),
        ("1", "0", "3"),
    ]
    assert float(rows[0]["outputs.mean_square"]) > 0


def test_export_can_write_a_file_and_jsonl(lab, run_instances, tmp_path):
    run_instances(lab, 0, [0, 1])
    target = tmp_path / "table.jsonl"

    result = invoke(export, *with_manifest(lab, "--format", "jsonl", "--output", target))

    assert result.exit_code == 0 and result.stdout == ""
    lines = [json.loads(line) for line in target.read_text().splitlines()]
    assert [r["instance_id"] for r in lines] == [0, 1]
    assert lines[0]["model.L"] == 2

    csv_target = tmp_path / "table.csv"
    invoke(export, *with_manifest(lab, "--output", csv_target))
    assert csv_target.read_text().splitlines()[0].startswith("experiment_id,instance_id,seed,")


def test_export_encodes_lists_as_json_and_missing_values_as_empty(lab, run_instances):
    run_instances(lab, 0, [0])
    path = lab.experiment(0).results_path(0)
    record = json.loads(path.read_text())
    record["outputs"] = {"x": None, "v": [1, 2]}
    path.write_text(json.dumps(record))

    (row,) = rows_of(invoke(export, *with_manifest(lab)))

    assert row["outputs.x"] == ""
    assert json.loads(row["outputs.v"]) == [1, 2]


def test_export_can_add_inputs_and_metadata(lab, run_instances):
    run_instances(lab, 0, [0])
    (row,) = rows_of(invoke(export, *with_manifest(lab, "--inputs", "--meta")))
    assert json.loads(row["inputs.first"])  # a list, encoded as JSON
    assert float(row["meta.wall_time_s"]) >= 0


def test_export_with_nothing_finished_says_so_on_stderr(lab):
    result = invoke(export, *with_manifest(lab))
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "No finished instances to export." in result.stderr


def test_export_with_nothing_finished_fails_when_a_file_was_asked_for(lab, tmp_path):
    table = tmp_path / "table.csv"
    result = invoke(export, *with_manifest(lab, "--output", str(table)))
    assert result.exit_code == 1
    assert f"No finished instances to export; {table} not written." in result.stderr
    assert not table.exists()


def test_export_can_be_narrowed(lab, run_instances):
    run_instances(lab, 0, [0])
    run_instances(lab, 1, [0])
    rows = rows_of(invoke(export, *with_manifest(lab, "--where", "model.L=3")))
    assert [r["experiment_id"] for r in rows] == ["1"]


# --------------------------------------------------------------------------- clean
def test_clean_lists_what_it_would_remove_and_deletes_nothing_by_default(lab, fail_instance):
    path = fail_instance(lab, 0, 1)

    result = invoke(clean, *with_manifest(lab, "--what", "errors"))

    assert result.exit_code == 0
    assert f"experiment 0 instance 1: {path}" in result.stdout
    assert "Would remove 1 item(s). Re-run with --yes to delete them." in result.stdout
    assert path.exists()


def test_clean_yes_deletes(lab, fail_instance):
    path = fail_instance(lab, 0, 1)
    result = invoke(clean, *with_manifest(lab, "--what", "errors", "--yes"))
    assert "Removed 1 item(s)." in result.stdout
    assert not path.exists()


def test_clean_with_nothing_to_do(lab):
    assert "Nothing to clean." in invoke(clean, *with_manifest(lab, "--what", "errors")).stdout


def test_clean_shows_only_the_first_ten_targets(lab, fail_instance):
    for experiment_id in (0, 1):
        for i in range(6):
            fail_instance(lab, experiment_id, i)
    result = invoke(clean, *with_manifest(lab, "--what", "errors"))
    assert "... and 2 more" in result.stdout
    assert "Would remove 12 item(s)" in result.stdout


def test_clean_results_must_be_narrowed_to_some_experiments(lab, run_instances):
    run_instances(lab, 0, [0])
    path = lab.experiment(0).results_path(0)

    refused = invoke(clean, *with_manifest(lab, "--what", "results", "--yes"))
    assert refused.exit_code == 2
    assert "refusing to remove the results of every experiment" in refused.stderr
    assert path.exists()

    for narrowing in (["--experiment-id", 0], ["--where", "model.L=2"], ["--all-experiments"]):
        result = invoke(clean, *with_manifest(lab, "--what", "results", *narrowing))
        assert result.exit_code == 0 and "Would remove 1 item(s)" in result.stdout


def test_clean_can_be_narrowed_to_instances(lab, fail_instance):
    for i in range(4):
        fail_instance(lab, 0, i)
    result = invoke(clean, *with_manifest(lab, "--what", "errors", "--instance-ids", "1-2"))
    assert "Would remove 2 item(s)" in result.stdout


def test_clean_stray_directories(lab):
    stray = lab.experiment(0).instances_dir / "8"
    stray.mkdir()

    result = invoke(clean, *with_manifest(lab, "--what", "stray", "--yes"))

    assert "Removed 1 item(s)." in result.stdout
    assert not stray.exists()


def test_clean_rejects_instance_ids_for_stray_and_malformed_ids(lab):
    stray = invoke(clean, *with_manifest(lab, "--what", "stray", "--instance-ids", "1"))
    assert stray.exit_code == 2 and "does not apply to --what stray" in stray.stderr

    malformed = invoke(clean, *with_manifest(lab, "--what", "errors", "--instance-ids", "a"))
    assert malformed.exit_code == 2 and "invalid id or range 'a'" in malformed.stderr


# --------------------------------------------------------------------------- init
def test_init_lab_writes_a_lab_and_its_params(tmp_path):
    result = invoke(init, "lab", "walks", "--directory", tmp_path / "proj", "--section", "walk")

    assert result.exit_code == 0, result.output
    assert (tmp_path / "proj" / "walks.py").exists() and (tmp_path / "proj" / "params.py").exists()
    assert f"[Init] wrote {tmp_path / 'proj' / 'walks.py'}" in result.stdout
    assert "mechane setup --params-file params.py" in result.stdout
    assert "class Walk(Config)" in (tmp_path / "proj" / "walks.py").read_text()


def test_init_lab_defaults_to_the_current_directory_and_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert invoke(init, "lab", "walks").exit_code == 0
    assert (tmp_path / "walks.py").exists()

    again = invoke(init, "lab", "walks")
    assert again.exit_code == 2 and "use --force" in again.stderr

    (tmp_path / "walks.py").write_text("edited")
    assert invoke(init, "lab", "walks", "--force").exit_code == 0
    assert "class WalksLab" in (tmp_path / "walks.py").read_text()


def test_init_lab_rejects_a_bad_name(tmp_path):
    result = invoke(init, "lab", "Bad-Name", "--directory", tmp_path)
    assert result.exit_code == 2
    assert "must be a lowercase identifier" in result.stderr
