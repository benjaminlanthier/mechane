import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from mechane import ExperimentConfig, expand, grid, stable_hash, zipped
from mechane.cli import (
    aggregate_results,
    main,
    setup_lab,
    simulate,
    simulate_batch,
    status,
)
from toy import Kind, Model, Solver, ToyExperiment, ToyLab


# --------------------------------------------------------------------------- hashing
def test_stable_hash_matches_original_formula():
    data = {"b": [1, 2.5, None], "a": {"x": "y"}}
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    assert stable_hash(data) == hashlib.sha256(blob).hexdigest()[:24]


def test_stable_hash_handles_enums_numpy_and_paths():
    assert stable_hash({"k": Kind.GAUSS, "n": np.int64(3), "p": Path("a")}) == stable_hash(
        {"k": "gauss", "n": 3, "p": "a"}
    )


def test_hash_exclude_and_enum_coercion():
    a, b = Solver(sigma=1.0, device="cpu"), Solver(sigma=1.0, device="cuda")
    assert a.hash() == b.hash() and a.to_dict() != b.to_dict()
    assert Model(L=4, kind=Kind.UNIFORM).kind is Kind.UNIFORM
    assert Model.from_dict({"L": 4, "kind": "gauss"}) == Model(L=4)


# --------------------------------------------------------------------------- sweeps
def test_lists_are_literals_unless_marked_as_axes():
    assert expand({"rates": [0.1, 0.2]}) == [{"rates": [0.1, 0.2]}]


def test_grid_zipped_product_chain_and_nesting():
    assert len(grid(a=[1, 2], b=[3, 4, 5])) == 6
    assert zipped({"c": 0}, a=[1, 2], b=[3, 4]).points() == [
        {"c": 0, "a": 1, "b": 3},
        {"c": 0, "a": 2, "b": 4},
    ]
    assert len(zipped(a=[1, 2]) * zipped(b=[1, 2, 3])) == 6
    assert len(zipped(a=[1, 2]) + zipped(a=[9])) == 3
    nested = expand({"name": "x", "params": zipped(n=[1, 2])})
    assert nested == [
        {"name": "x", "params": {"n": 1}},
        {"name": "x", "params": {"n": 2}},
    ]
    assert zipped().points() == [{}]


def test_sweep_errors():
    with pytest.raises(ValueError, match="equal lengths"):
        zipped(a=[1], b=[1, 2]).points()
    with pytest.raises(TypeError, match="must be a list"):
        grid(a="abc")
    with pytest.raises(ValueError, match="both as fixed"):
        grid({"a": 1}, a=[1, 2])


# --------------------------------------------------------------------------- laboratory
PARAMS = {
    "seed": 7,
    "num_instances": 12,
    "model": grid(L=[2, 3]),
    "solver": {"sigma": 1.0, "n_samples": 50},
}


def make_lab(tmp_path: Path, params=None):
    return ToyLab.from_configs(tmp_path, params or PARAMS)


def test_from_configs_numbering_and_layout(tmp_path):
    PARAMS_2 = {**PARAMS, "solver": zipped({"n_samples": 50}, sigma=[0.5, 1.0])}
    lab = make_lab(tmp_path, PARAMS_2)
    assert [e.config.experiment_id for e in lab.experiments] == [0, 1, 2, 3]
    # directory order follows section_classes: model, then solver
    e = lab.experiments[1]
    assert e.instances_dir == (
        tmp_path / e.config.model.hash() / e.config.solver.hash() / "instances"
    )
    assert e.instance_dir(3).name == "03"  # 12 instances -> 2 digits
    assert e.stage_dir("model") == tmp_path / e.config.model.hash()


def test_unknown_params_key_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown configs keys"):
        make_lab(tmp_path, {**PARAMS, "modle": {}})


def test_directory_collision_is_detected(tmp_path):
    params = {**PARAMS, "solver": zipped({"n_samples": 50}, sigma=[1.0, 1.0])}
    with pytest.raises(ValueError, match="same directory"):
        make_lab(tmp_path, params).setup()


# --------------------------------------------------------------------------- seeds
def test_derived_seed_is_stable_when_sweep_grows(tmp_path):
    small = make_lab(tmp_path, {**PARAMS, "model": grid(L=[3])})
    large = make_lab(tmp_path, {**PARAMS, "model": grid(L=[2, 3])})
    # same physical experiment (L=3) now has a different experiment_id ...
    e_small, e_large = small.experiments[0], large.experiments[1]
    assert e_small.config.experiment_id != e_large.config.experiment_id
    # ... but the same seeds.
    assert [e_small.instance_seed(i) for i in range(5)] == [
        e_large.instance_seed(i) for i in range(5)
    ]
    assert len({e_small.instance_seed(i) for i in range(12)}) == 12
    assert all(0 <= e_small.instance_seed(i) < 2**31 for i in range(12))


def test_legacy_seed_matches_original_formula(tmp_path):
    class LegacyExp(ToyExperiment):
        seed_scheme = "legacy"

    lab = make_lab(tmp_path)
    exp = LegacyExp(tmp_path, lab.experiments[0].config)
    expected = int(stable_hash({"instance_id": 4, **exp.config.to_dict()})[:8], 16) & 0x7FFFFFFF
    assert exp.instance_seed(4) == expected


# --------------------------------------------------------------------------- end to end (CLI)
def write_params_file(tmp_path: Path):
    f = tmp_path / "params.py"
    f.write_text(
        "from pathlib import Path\n"
        "from mechane import grid\n"
        f"ROOT = Path({str(tmp_path / 'out')!r})\n"
        "LABORATORY = 'toy:ToyLab'\n"
        "PARAMS = {'seed': 1, 'num_instances': 4, 'model': grid(L=[2, 3]),\n"
        "          'solver': {'sigma': 1.0, 'n_samples': 20}}\n"
    )
    return f


def test_full_workflow_through_cli(tmp_path: Path):
    runner = CliRunner()
    r = runner.invoke(setup_lab, ["--params-file", str(write_params_file(tmp_path))])
    assert r.exit_code == 0, r.output
    manifest_path = tmp_path / "out" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["laboratory"] == "toy:ToyLab" and manifest["num_experiments"] == 2

    # run a batch for experiment 0 and a single instance for experiment 1
    mp = ["--manifest-path", str(manifest_path)]
    r = runner.invoke(simulate_batch, [*mp, "--experiment-id", "0", "--instance-ids", "0,1,2,3"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(simulate, [*mp, "--experiment-id", "1", "--instance-id", "2"])
    assert r.exit_code == 0, r.output

    lab = ToyLab.from_manifest(manifest_path)
    record = json.loads(lab.experiment(0).results_path(1).read_text())
    assert {"instance_id", "experiment_id", "seed", "inputs", "outputs", "meta"} <= set(record)
    assert isinstance(record["inputs"]["first"], list)  # numpy -> list
    assert "env" not in record and "wall_time_s" in record["meta"]
    assert record["meta"]["packages"]["numpy"]

    # skip-if-done
    r = runner.invoke(simulate, [*mp, "--experiment-id", "1", "--instance-id", "2"])
    assert "already exists" in r.output

    # reproducibility: same instance rerun with --overwrite gives identical outputs
    path = lab.experiment(0).results_path(1)
    before = json.loads(path.read_text())["outputs"]
    assert (
        runner.invoke(
            simulate, [*mp, "--experiment-id", "0", "--instance-id", "1", "--overwrite"]
        ).exit_code
        == 0
    )
    assert json.loads(path.read_text())["outputs"] == before

    # status + aggregation
    r = runner.invoke(status, mp)
    assert "experiment    0: 4 done, 0 failed, 0 missing" in r.output
    assert "experiment    1: 1 done, 0 failed, 3 missing" in r.output
    assert runner.invoke(aggregate_results, mp).exit_code == 0
    agg = json.loads((tmp_path / "out" / "aggregated_results.json").read_text())
    assert agg["0"]["num_done"] == 4 and set(agg["0"]["instances"]) == {
        "0",
        "1",
        "2",
        "3",
    }
    assert agg["1"]["missing"] == [0, 1, 3]

    # the umbrella group exposes the same commands
    assert runner.invoke(main, ["status", *mp]).exit_code == 0


def test_failures_are_isolated_and_traceback_saved(tmp_path: Path):
    runner = CliRunner()
    params = write_params_file(tmp_path)
    params.write_text(params.read_text().replace("'sigma': 1.0", "'sigma': -1.0"))
    assert runner.invoke(setup_lab, ["--params-file", str(params)]).exit_code == 0
    mp = ["--manifest-path", str(tmp_path / "out" / "manifest.json")]
    r = runner.invoke(simulate_batch, [*mp, "--experiment-id", "0", "--instance-ids", "0,1"])
    assert r.exit_code == 1
    lab = ToyLab.from_manifest(tmp_path / "out" / "manifest.json")
    assert "negative sigma" in lab.experiment(0).error_path(0).read_text()
    assert lab.experiment(0).status()["failed"] == [0, 1]


def test_padding_migration(tmp_path):
    lab = make_lab(tmp_path)
    exp = lab.experiments[0]
    exp.instances_dir.mkdir(parents=True)
    (exp.instances_dir / "5").mkdir()
    exp.migrate_padding()
    assert (exp.instances_dir / "05").is_dir()


def test_experiment_config_attribute_access(tmp_path):
    cfg = make_lab(tmp_path).experiments[0].config
    assert isinstance(cfg, ExperimentConfig) and cfg.model == Model(L=2)
    with pytest.raises(AttributeError):
        cfg.nonexistent  # noqa: B018
