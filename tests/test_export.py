"""`mechane.core.export`: results as a flat table."""

import json

from mechane.core.export import build_table


def test_one_row_per_finished_instance_with_parameters_and_outputs(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    run_instances(lab, 1, [3])

    table = build_table(lab.experiments)

    assert table.columns == (
        "experiment_id",
        "instance_id",
        "seed",
        "model.L",
        "model.kind",
        "solver.sigma",
        "solver.n_samples",
        "solver.device",
        "outputs.mean_square",
    )
    assert [(r["experiment_id"], r["instance_id"]) for r in table.rows] == [(0, 0), (0, 1), (1, 3)]
    row = table.rows[2]
    assert (row["model.L"], row["solver.n_samples"]) == (3, 20)
    assert row["seed"] == lab.experiment(1).instance_seed(3)
    assert isinstance(row["outputs.mean_square"], float)


def test_nothing_finished_means_an_empty_table(lab):
    table = build_table(lab.experiments)
    assert (table.columns, table.rows) == ((), ())


def test_inputs_and_metadata_are_included_only_on_request(lab, run_instances):
    run_instances(lab, 0, [0])

    default = build_table(lab.experiments)
    with_inputs = build_table(lab.experiments, include_inputs=True)
    with_meta = build_table(lab.experiments, include_meta=True)

    assert not any(c.startswith(("inputs.", "meta.")) for c in default.columns)
    assert isinstance(with_inputs.rows[0]["inputs.first"], list)
    assert "wall_time_s" in {c.removeprefix("meta.") for c in with_meta.columns}
    assert with_meta.rows[0]["meta.packages.numpy"]  # nested metadata is flattened too


def test_nested_outputs_are_flattened_and_missing_ones_become_an_empty_mapping(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    exp = lab.experiment(0)
    for i, outputs in ((0, {"energy": {"mean": 1.0, "std": 0.1}, "n": 3}), (1, None)):
        path = exp.results_path(i)
        record = json.loads(path.read_text())
        record["outputs"] = outputs
        path.write_text(json.dumps(record))

    table = build_table([exp])

    first, second = table.rows
    assert (first["outputs.energy.mean"], first["outputs.energy.std"], first["outputs.n"]) == (
        1.0,
        0.1,
        3,
    )
    assert second["outputs"] == {}
    assert "outputs.n" in table.columns and "outputs" in table.columns  # the union of both rows


def test_only_the_given_experiments_are_exported(lab, run_instances):
    run_instances(lab, 0, [0])
    run_instances(lab, 1, [0])
    table = build_table([lab.experiment(1)])
    assert [r["experiment_id"] for r in table.rows] == [1]
