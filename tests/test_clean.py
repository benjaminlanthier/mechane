"""`mechane.core.clean`: finding and removing what lets instances run again."""

from mechane.core.clean import find_targets, remove_targets
from mechane.core.runner import run_instance


def where(targets):
    return [(t.kind, t.experiment_id, t.instance_id, t.path.name) for t in targets]


def test_errors_targets_every_job_err(lab, run_instances, fail_instance):
    run_instances(lab, 0, [0])
    fail_instance(lab, 0, 1)  # a failed instance
    fail_instance(lab, 0, 0)  # a stale job.err next to a result
    fail_instance(lab, 1, 5)

    targets = find_targets(lab.experiments, "errors")

    assert where(targets) == [
        ("errors", 0, 0, "job.err"),
        ("errors", 0, 1, "job.err"),
        ("errors", 1, 5, "job.err"),
    ]
    assert all(t.path.is_file() for t in targets)


def test_results_targets_results_and_their_job_err(lab, run_instances, fail_instance):
    run_instances(lab, 0, [0, 1])
    fail_instance(lab, 0, 1)

    targets = find_targets([lab.experiment(0)], "results")

    assert where(targets) == [
        ("results", 0, 0, "results.json"),
        ("results", 0, 1, "results.json"),
        ("results", 0, 1, "job.err"),
    ]


def test_instance_ids_narrow_errors_and_results_and_out_of_range_ids_are_ignored(
    lab, run_instances, fail_instance
):
    run_instances(lab, 0, [0, 1, 2])
    for i in (0, 1, 2):
        fail_instance(lab, 0, i)

    errors = find_targets([lab.experiment(0)], "errors", [1, 99, -3])
    results = find_targets([lab.experiment(0)], "results", [2])

    assert where(errors) == [("errors", 0, 1, "job.err")]
    assert where(results) == [("results", 0, 2, "results.json"), ("results", 0, 2, "job.err")]


def test_stray_targets_numeric_directories_past_the_end_in_numeric_order(lab):
    exp = lab.experiment(0)
    for name in ("10", "7", "6", "5", "notes"):  # 5 is a real instance; "notes" is not numeric
        (exp.instances_dir / name).mkdir()
    (exp.instances_dir / "8").write_text("a file named like an instance")

    targets = find_targets([exp], "stray")

    assert [(t.instance_id, t.path.name) for t in targets] == [(6, "6"), (7, "7"), (10, "10")]
    assert all(t.kind == "stray" and t.path.is_dir() for t in targets)


def test_stray_ignores_instance_ids_and_experiments_without_a_directory(lab, tmp_path):
    (lab.experiment(1).instances_dir / "6").mkdir()
    assert where(find_targets(lab.experiments, "stray", [0])) == [("stray", 1, 6, "6")]

    lab.experiment(0).instances_dir.rmdir()
    assert find_targets([lab.experiment(0)], "stray") == []


def test_removing_files_and_directories_counts_what_it_removed(lab, run_instances, fail_instance):
    exp = lab.experiment(0)
    run_instances(lab, 0, [0])
    fail_instance(lab, 0, 1)
    stray = exp.instances_dir / "9"
    stray.mkdir()
    (stray / "results.json").write_text("{}")

    targets = find_targets([exp], "errors") + find_targets([exp], "stray")

    assert remove_targets(targets) == 2
    assert not exp.error_path(1).exists() and not stray.exists()
    assert exp.results_path(0).exists()  # results were not part of the clean


def test_removing_something_already_gone_is_not_an_error(lab, fail_instance):
    path = fail_instance(lab, 0, 0)
    targets = find_targets(lab.experiments, "errors")
    path.unlink()
    assert remove_targets(targets) == 1


def test_a_failed_instance_becomes_missing_and_a_cleaned_result_runs_again(
    lab, run_instances, fail_instance
):
    exp = lab.experiment(0)
    fail_instance(lab, 0, 1)
    assert exp.status()["failed"] == [1]
    remove_targets(find_targets([exp], "errors"))
    assert exp.status() == {"done": [], "failed": [], "missing": [0, 1, 2, 3, 4, 5]}

    run_instances(lab, 0, [0])
    assert run_instance(lab, 0, 0) is None  # finished work is skipped ...
    remove_targets(find_targets([exp], "results", [0]))
    assert run_instance(lab, 0, 0) == exp.results_path(0)  # ... until its results are cleaned
