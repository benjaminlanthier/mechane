"""`mechane.core.listing`: describing experiments by their parameters."""

from mechane.core.listing import describe, varying_keys


def test_each_row_describes_one_experiment(lab):
    first, second = describe(lab.experiments)

    assert (first.experiment_id, second.experiment_id) == (0, 1)
    assert first.parameters == {
        "model.L": 2,
        "model.kind": "gauss",
        "solver.sigma": 1.0,
        "solver.n_samples": 20,
        "solver.device": "cpu",
    }
    assert second.parameters["model.L"] == 3
    assert first.content_hash == lab.experiment(0).config.content_hash()
    assert first.path == lab.experiment(0).instances_dir
    assert first.status is None  # reading the status is opt-in


def test_status_counts_are_included_on_request(lab, run_instances, fail_instance):
    run_instances(lab, 0, [0, 1])
    fail_instance(lab, 0, 2)

    first, second = describe(lab.experiments, with_status=True)

    assert first.status == {"done": 2, "failed": 1, "missing": 3}
    assert second.status == {"done": 0, "failed": 0, "missing": 6}


def test_only_parameters_that_differ_tell_experiments_apart(lab):
    assert varying_keys(describe(lab.experiments)) == ["model.L"]


def test_a_single_experiment_or_none_has_no_varying_parameters(lab):
    assert varying_keys(describe([lab.experiment(0)])) == []
    assert varying_keys([]) == []
