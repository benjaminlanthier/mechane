"""`mechane.core.runner`: running instances and batches with no command line around them."""

import json

import pytest

from mechane.core.runner import (
    BatchResult,
    InstanceFailed,
    InstanceSaved,
    InstanceSkipped,
    InstanceStarted,
    run_batch,
    run_instance,
)
from toy import ToyLab


def make_lab(tmp_path, sigma=1.0) -> ToyLab:
    """One experiment, four instances. A negative `sigma` makes every instance raise."""
    lab = ToyLab.from_configs(
        tmp_path,
        {
            "seed": 1,
            "num_instances": 4,
            "model": {"L": 2},
            "solver": {"sigma": sigma, "n_samples": 10},
        },
    )
    assert isinstance(lab, ToyLab)
    return lab


# --------------------------------------------------------------------------- run_instance
def test_run_instance_persists_the_record_and_reports_progress(tmp_path):
    lab = make_lab(tmp_path)
    events = []

    path = run_instance(lab, 0, 1, listener=events.append)

    experiment = lab.experiment(0)
    assert path == experiment.results_path(1)
    assert json.loads(path.read_text())["seed"] == experiment.instance_seed(1)
    assert events == [InstanceStarted("toy", 0, 1), InstanceSaved(path)]


def test_run_instance_is_silent_without_a_listener(tmp_path, capsys):
    path = run_instance(make_lab(tmp_path), 0, 0)
    assert path is not None and path.exists()
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_finished_instances_are_skipped_unless_overwrite_is_set(tmp_path):
    lab = make_lab(tmp_path)
    first = run_instance(lab, 0, 1)
    assert first is not None
    events = []

    assert run_instance(lab, 0, 1, listener=events.append) is None
    assert events == [InstanceSkipped(first)]

    events.clear()
    assert run_instance(lab, 0, 1, overwrite=True, listener=events.append) == first
    assert [type(e) for e in events] == [InstanceStarted, InstanceSaved]


def test_a_failing_instance_leaves_a_traceback_and_reraises(tmp_path):
    lab = make_lab(tmp_path, sigma=-1.0)
    events = []

    with pytest.raises(ValueError, match="negative sigma"):
        run_instance(lab, 0, 2, listener=events.append)

    experiment = lab.experiment(0)
    assert events == [InstanceStarted("toy", 0, 2)]  # it started, and was never saved
    assert "negative sigma" in experiment.error_path(2).read_text()
    assert not experiment.results_path(2).exists()


def test_a_later_success_removes_the_stale_traceback(tmp_path):
    lab = make_lab(tmp_path)
    experiment = lab.experiment(0)
    experiment.instance_dir(0).mkdir(parents=True)
    experiment.error_path(0).write_text("traceback from an earlier attempt")

    run_instance(lab, 0, 0)

    assert experiment.results_path(0).exists()
    assert not experiment.error_path(0).exists()


# --------------------------------------------------------------------------- run_batch
def test_run_batch_reports_failures_and_keeps_going(tmp_path):
    lab = make_lab(tmp_path, sigma=-1.0)
    events = []

    result = run_batch(lab, 0, [0, 1], listener=events.append)

    assert result == BatchResult(instance_ids=(0, 1), failed=(0, 1))
    assert [type(e) for e in events] == [
        InstanceStarted,
        InstanceFailed,
        InstanceStarted,
        InstanceFailed,
    ]
    failures = [e for e in events if isinstance(e, InstanceFailed)]
    assert [f.instance_id for f in failures] == [0, 1]
    assert all(isinstance(f.error, ValueError) for f in failures)


def test_run_batch_without_continue_on_error_stops_at_the_first_failure(tmp_path):
    lab = make_lab(tmp_path, sigma=-1.0)
    events = []

    with pytest.raises(ValueError, match="negative sigma"):
        run_batch(lab, 0, [0, 1, 2], continue_on_error=False, listener=events.append)

    assert [type(e) for e in events] == [InstanceStarted, InstanceFailed]
    assert lab.experiment(0).status() == {"done": [], "failed": [0], "missing": [1, 2, 3]}


def test_run_batch_accepts_any_iterable_of_ids(tmp_path):
    lab = make_lab(tmp_path)

    result = run_batch(lab, 0, (i for i in range(3)))  # a generator, and no listener

    assert result == BatchResult(instance_ids=(0, 1, 2), failed=())
    assert lab.experiment(0).status()["done"] == [0, 1, 2]


def test_run_batch_skips_what_is_already_done(tmp_path):
    lab = make_lab(tmp_path)
    run_instance(lab, 0, 1)
    events = []

    result = run_batch(lab, 0, [0, 1], listener=events.append)

    assert result.failed == ()
    assert [type(e) for e in events] == [InstanceStarted, InstanceSaved, InstanceSkipped]
