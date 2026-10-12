"""`mechane.core.selection`: choosing experiments by id and by parameter."""

import pytest

from mechane.core.selection import SelectionError, parameters_of, queryable, select_experiments


def ids(experiments):
    return [e.config.experiment_id for e in experiments]


def test_no_arguments_selects_everything(lab):
    assert ids(select_experiments(lab)) == [0, 1]


def test_experiment_ids_are_deduplicated_and_sorted(lab):
    assert ids(select_experiments(lab, [1, 0, 1])) == [0, 1]
    assert ids(select_experiments(lab, [1])) == [1]


def test_an_unknown_experiment_id_lists_the_valid_ones(lab):
    with pytest.raises(SelectionError, match=r"No experiment 9; valid ids: 0\.\.1"):
        select_experiments(lab, [9])


@pytest.mark.parametrize(
    ("clause", "expected"),
    [
        ("model.L=3", [1]),  # JSON number
        ("model.kind=gauss", [0, 1]),  # not JSON: plain text
        ("solver.sigma=1.0", [0, 1]),
        ("solver.sigma=1", [0, 1]),  # 1 == 1.0
        ("model.L=5", []),
        ("seed=7", [0, 1]),  # identity keys can be queried too
        ("experiment_id=1", [1]),
    ],
)
def test_filters_compare_parameters_by_value(lab, clause, expected):
    assert ids(select_experiments(lab, where=[clause])) == expected


def test_filters_are_anded_with_each_other_and_with_the_ids(lab):
    assert ids(select_experiments(lab, where=["model.L=2", "solver.n_samples=20"])) == [0]
    assert ids(select_experiments(lab, where=["model.L=2", "solver.n_samples=21"])) == []
    assert ids(select_experiments(lab, [1], where=["model.L=2"])) == []


def test_an_unknown_parameter_lists_what_is_available(lab):
    with pytest.raises(
        SelectionError,
        match=r"unknown parameter 'model.X'; available: \['experiment_id', 'model.L'",
    ):
        select_experiments(lab, where=["model.X=1"])


@pytest.mark.parametrize("clause", ["model.L", "=3", "  =3"])
def test_a_filter_must_be_key_equals_value(lab, clause):
    with pytest.raises(SelectionError, match="expected KEY=VALUE"):
        select_experiments(lab, where=[clause])


def test_parameters_exclude_the_identity_keys_that_queries_include(lab):
    exp = lab.experiment(1)
    assert parameters_of(exp) == {
        "model.L": 3,
        "model.kind": "gauss",
        "solver.sigma": 1.0,
        "solver.n_samples": 20,
        "solver.device": "cpu",
    }
    assert {"experiment_id", "num_instances", "seed"} <= queryable(exp).keys()
