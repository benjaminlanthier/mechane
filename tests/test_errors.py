"""`mechane.core.errors`: grouping failed instances by what went wrong."""

from mechane.core.errors import EMPTY_SIGNATURE, collect_failures, error_signature

TRACEBACK = """Traceback (most recent call last):
  File "x.py", line 3, in run
    raise ValueError("negative sigma")
ValueError: negative sigma
"""


def test_the_signature_is_the_last_nonempty_line():
    assert error_signature(TRACEBACK) == "ValueError: negative sigma"
    assert error_signature("first\nlast line  \n\n  \n") == "last line"


def test_an_empty_traceback_has_a_placeholder_signature():
    assert error_signature("") == EMPTY_SIGNATURE
    assert error_signature("\n  \n") == EMPTY_SIGNATURE


def test_no_failures_means_no_groups(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    assert collect_failures(lab.experiments) == []


def test_failures_are_grouped_by_error_and_the_most_frequent_comes_first(lab, fail_instance):
    for i in (0, 1, 2):
        fail_instance(lab, 0, i, "Traceback...\nMemoryError")
    fail_instance(lab, 1, 4, TRACEBACK)
    fail_instance(lab, 1, 5, "Traceback...\nMemoryError")

    groups = collect_failures(lab.experiments)

    assert [(g.signature, g.count) for g in groups] == [
        ("MemoryError", 4),
        ("ValueError: negative sigma", 1),
    ]
    memory = groups[0]
    assert [(f.experiment_id, f.instance_id) for f in memory.instances] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 5),
    ]
    assert memory.instances[0].error_file == lab.experiment(0).error_path(0)
    assert "MemoryError" in memory.instances[0].read_traceback()


def test_equal_counts_are_ordered_by_signature(lab, fail_instance):
    fail_instance(lab, 0, 0, "ZeroDivisionError")
    fail_instance(lab, 0, 1, "AssertionError")
    assert [g.signature for g in collect_failures(lab.experiments)] == [
        "AssertionError",
        "ZeroDivisionError",
    ]


def test_an_instance_with_results_is_not_a_failure_even_with_a_stale_job_err(
    lab, run_instances, fail_instance
):
    run_instances(lab, 0, [0])
    fail_instance(lab, 0, 0, "stale")
    assert collect_failures(lab.experiments) == []


def test_only_the_given_experiments_are_inspected(lab, fail_instance):
    fail_instance(lab, 0, 0, "A")
    fail_instance(lab, 1, 0, "B")
    assert [g.signature for g in collect_failures([lab.experiment(1)])] == ["B"]
