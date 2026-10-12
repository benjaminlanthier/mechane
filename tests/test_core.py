"""Error paths and less common behaviour of `Config`, `Configs`, `Experiment`, `Instance` and
`Laboratory`. The happy paths are in test_mechane.py."""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mechane import (
    Config,
    Configs,
    Experiment,
    ExperimentConfig,
    InstanceOutput,
    Laboratory,
    grid,
    load_laboratory,
    zipped,
)
from mechane.core import laboratory as laboratory_module
from mechane.core.laboratory import read_manifest
from toy import Kind, Model, ToyInstance, ToyLab

PARAMS = {
    "seed": 7,
    "num_instances": 12,
    "model": grid(L=[2, 3]),
    "solver": {"sigma": 1.0, "n_samples": 50},
}


def make_lab(tmp_path: Path, params=None) -> ToyLab:
    lab = ToyLab.from_configs(tmp_path, params or PARAMS)
    assert isinstance(lab, ToyLab)
    return lab


# --------------------------------------------------------------------------- Config
@dataclass(frozen=True)
class Flexible(Config):
    mode: float | Kind | None = None


def test_enum_coercion_looks_through_unions_and_optionals():
    assert Flexible.from_dict({"mode": "gauss"}).mode is Kind.GAUSS  # raw value, as in a manifest
    assert Flexible(mode=Kind.UNIFORM).mode is Kind.UNIFORM
    assert Flexible().mode is None  # None is left alone, not passed to the enum
    assert Flexible.from_dict({"mode": "uniform"}).mode is Kind.UNIFORM


def test_unresolvable_annotations_disable_coercion_instead_of_failing():
    @dataclass(frozen=True)
    class Dangling(Config):
        kind: "NotDefinedAnywhere"  # noqa: F821  # ty: ignore[unresolved-reference]

    assert Dangling(kind="gauss").kind == "gauss"  # kept as given


# --------------------------------------------------------------------------- Configs
def test_reserved_section_names_are_rejected():
    @dataclass(frozen=True)
    class Bad(Configs):
        seed: Model

    with pytest.raises(ValueError, match=r"Config names \['seed'\] are reserved"):
        Bad(seed=Model(L=1))


def test_every_section_must_be_a_config_subclass():
    @dataclass(frozen=True)
    class Bad(Configs):
        n: int

    with pytest.raises(TypeError, match=r"Bad\.n must be annotated with a Config subclass"):
        Bad.section_classes()


# --------------------------------------------------------------------------- Experiment
@dataclass(frozen=True)
class EmptyConfigs(Configs):
    pass


@dataclass(frozen=True)
class EmptyExperimentConfig(ExperimentConfig):
    configs: EmptyConfigs


class EmptyExperiment(Experiment):
    config: EmptyExperimentConfig


class EmptyLab(Laboratory):
    name = "empty"
    experiment_class = EmptyExperiment
    instance_class = ToyInstance


def test_stage_dir_of_an_unknown_section_lists_the_valid_ones(tmp_path):
    exp = make_lab(tmp_path).experiments[0]
    with pytest.raises(KeyError, match=r"No section 'nope'; configs are \['model', 'solver'\]"):
        exp.stage_dir("nope")


def test_experiment_without_sections_keeps_instances_directly_under_root(tmp_path):
    config = EmptyExperimentConfig(experiment_id=0, num_instances=3, seed=0, configs=EmptyConfigs())
    exp = EmptyExperiment(tmp_path, config)
    assert exp.instances_dir == tmp_path / "instances"
    assert exp.instance_dir(2) == tmp_path / "instances" / "2"


def test_migrate_padding_is_a_noop_without_an_instances_dir(tmp_path):
    exp = make_lab(tmp_path).experiments[0]
    assert not exp.instances_dir.exists()
    exp.migrate_padding()
    assert not exp.instances_dir.exists()


def test_migrate_padding_only_renames_what_it_should(tmp_path):
    exp = make_lab(tmp_path).experiments[0]  # 12 instances -> width 2
    d = exp.instances_dir
    d.mkdir(parents=True)
    (d / "notes.txt").write_text("a file, not a directory")
    (d / "abc").mkdir()  # not numeric
    (d / "05").mkdir()  # already the right width
    (d / "5").mkdir()  # would become "05", which exists: left alone, never overwritten
    (d / "7").mkdir()  # renamed to "07"
    exp.migrate_padding()
    assert sorted(p.name for p in d.iterdir()) == ["05", "07", "5", "abc", "notes.txt"]


# --------------------------------------------------------------------------- Instance
class MappingInstance(ToyInstance):
    """`run()` returns a plain mapping, which is treated as the outputs."""

    def run(self):
        return {"value": np.float64(1.5), "vector": np.arange(3)}


class MetaInstance(ToyInstance):
    def run(self):
        return InstanceOutput(outputs={"x": 1}, meta={"backend": "numpy"})


def test_instance_may_return_a_plain_mapping(tmp_path):
    exp = make_lab(tmp_path).experiments[0]
    record = MappingInstance(exp, 3).execute(provenance={"hostname": "node-1"})
    assert record["outputs"] == {"value": 1.5, "vector": [0, 1, 2]}
    assert record["inputs"] == {}
    assert record["instance_id"] == 3
    assert record["experiment_id"] == 0
    assert record["seed"] == exp.instance_seed(3)
    json.dumps(record)  # the record is plain JSON
    # execute() creates the instance directory but persists nothing; save() does that.
    assert exp.instance_dir(3).is_dir()
    assert not exp.results_path(3).exists()


def test_instance_meta_is_merged_with_the_provenance(tmp_path):
    exp = make_lab(tmp_path).experiments[0]
    record = MetaInstance(exp, 0).execute(provenance={"hostname": "node-1"})
    assert record["meta"]["hostname"] == "node-1"
    assert record["meta"]["backend"] == "numpy"
    assert {"started_at", "wall_time_s"} <= set(record["meta"])


# --------------------------------------------------------------------------- Laboratory
class SolverFirstLab(ToyLab):
    sweep_order = ("solver", "model")


class BadOrderLab(ToyLab):
    sweep_order = ("model",)


def test_counts(tmp_path):
    lab = make_lab(tmp_path)
    assert (lab.num_experiments, lab.num_instances, lab.total_num_instances) == (2, 12, 24)


def test_seed_and_num_instances_have_defaults(tmp_path):
    lab = ToyLab.from_configs(tmp_path, {"model": {"L": 2}, "solver": {"sigma": 1.0}})
    assert (lab.seed, lab.num_instances) == (0, 1)


def test_sweep_order_controls_experiment_numbering_but_not_directories(tmp_path):
    params = {
        **PARAMS,
        "solver": zipped({"n_samples": 50}, sigma=[0.5, 1.0]),
    }
    default = ToyLab.from_configs(tmp_path, params)
    solver_first = SolverFirstLab.from_configs(tmp_path, params)

    def points(lab):
        return [(e.config.model.L, e.config.solver.sigma) for e in lab.experiments]

    # The first section in `sweep_order` varies slowest.
    assert points(default) == [(2, 0.5), (2, 1.0), (3, 0.5), (3, 1.0)]
    assert points(solver_first) == [(2, 0.5), (3, 0.5), (2, 1.0), (3, 1.0)]
    # Directory nesting still follows the Configs field order.
    exp = solver_first.experiments[1]
    assert exp.instances_dir == (
        tmp_path / exp.config.model.hash() / exp.config.solver.hash() / "instances"
    )


def test_sweep_order_must_list_exactly_the_sections(tmp_path):
    with pytest.raises(ValueError, match="sweep_order must list exactly the sections"):
        BadOrderLab.from_configs(tmp_path, PARAMS)


def test_empty_sweep_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="The sweep is empty"):
        make_lab(tmp_path, {**PARAMS, "model": grid(L=[])})


def test_a_lab_without_sections_explains_what_is_missing(tmp_path):
    with pytest.raises(TypeError, match="declares no sections"):
        EmptyLab.section_classes()
    with pytest.raises(TypeError, match="declares no sections"):
        EmptyLab.from_configs(tmp_path, {})


def test_experiment_lookup_lists_the_valid_ids(tmp_path):
    lab = make_lab(tmp_path)
    assert lab.experiment(1) is lab.experiments[1]
    with pytest.raises(KeyError, match=r"No experiment 5; valid ids: 0\.\.1"):
        lab.experiment(5)


def test_build_instance_binds_the_right_experiment(tmp_path):
    lab = make_lab(tmp_path)
    instance = lab.build_instance(1, 3)
    assert isinstance(instance, ToyInstance)
    assert instance.experiment is lab.experiment(1)
    assert instance.instance_id == 3


def test_provenance_tracks_packages_and_the_git_state_of_the_lab_source(tmp_path):
    info = make_lab(tmp_path).provenance()
    assert set(info["packages"]) == {"mechane", "numpy"}
    assert "git" in info  # a dict, or None when the source is not in a repository


@pytest.mark.parametrize("error", [TypeError, OSError])
def test_provenance_survives_a_lab_whose_source_file_is_unknown(tmp_path, monkeypatch, error):
    lab = make_lab(tmp_path)

    def no_source(_obj):
        raise error("source file not available")

    monkeypatch.setattr(laboratory_module, "inspect", SimpleNamespace(getfile=no_source))
    info = lab.provenance()
    assert "git" not in info
    assert {"hostname", "python", "packages", "slurm"} <= set(info)


def test_setup_refuses_a_lab_that_jobs_could_not_import(tmp_path):
    # Claims to live in `toy`, but there is no `toy.Unreachable`.
    unreachable = type("Unreachable", (ToyLab,), {"__module__": "toy"})
    lab = unreachable.from_configs(tmp_path, PARAMS)
    with pytest.raises(RuntimeError, match="toy:Unreachable is not importable by reference"):
        lab.setup()
    assert not lab.manifest_path.exists()


def test_setup_refuses_a_reference_that_resolves_to_another_class(tmp_path):
    # Its reference, `toy:ToyLab`, resolves to the real ToyLab.
    impostor = type("Impostor", (ToyLab,), {"__module__": "toy", "__qualname__": "ToyLab"})
    lab = impostor.from_configs(tmp_path, PARAMS)
    with pytest.raises(RuntimeError, match="toy:ToyLab resolves to a different class"):
        lab.setup()
    assert not lab.manifest_path.exists()


def test_setup_writes_the_manifest_and_every_instances_directory(tmp_path):
    lab = make_lab(tmp_path)
    lab.setup()
    manifest = read_manifest(lab.manifest_path)
    assert manifest["num_experiments"] == 2
    assert manifest["num_instances_per_experiment"] == 12
    for exp in lab.experiments:
        assert exp.instances_dir.is_dir()
        assert manifest["experiments"][str(exp.config.experiment_id)]["experiment_path"] == str(
            exp.instances_dir
        )


def test_aggregate_refreshes_known_experiments_and_keeps_the_others(tmp_path):
    lab = make_lab(tmp_path)
    lab.setup()
    # An aggregate from an earlier run: a stale entry for experiment 0, plus one for an
    # experiment this lab does not have (say, from a previous sweep).
    previous = {"0": {"num_done": -1}, "42": {"note": "from an earlier sweep"}}
    lab.all_results_path.write_text(json.dumps(previous))
    finished = lab.experiment(0).results_path(0)
    finished.parent.mkdir()  # setup creates `instances/`; an instance's own folder appears on run
    finished.write_text(json.dumps({"outputs": {"v": 1}}))

    lab.aggregate_results()

    aggregated = json.loads(lab.all_results_path.read_text())
    assert aggregated["42"] == {"note": "from an earlier sweep"}
    assert aggregated["0"]["num_done"] == 1
    assert aggregated["0"]["instances"] == {"0": {"v": 1}}
    assert aggregated["0"]["missing"] == list(range(1, 12))
    assert aggregated["1"]["num_done"] == 0


# --------------------------------------------------------------------------- manifests
def test_a_missing_manifest_is_reported_with_its_path(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match=r"No manifest at .*nope\.json"):
        read_manifest(missing)
    with pytest.raises(FileNotFoundError, match="No manifest at"):
        load_laboratory(missing)
    with pytest.raises(FileNotFoundError, match="No manifest at"):
        ToyLab.from_manifest(missing)


def test_load_laboratory_rebuilds_the_lab_from_the_manifest_alone(tmp_path):
    lab = make_lab(tmp_path)
    lab.setup()
    loaded = load_laboratory(lab.manifest_path)
    assert isinstance(loaded, ToyLab)
    assert loaded.root == tmp_path
    assert [e.config for e in loaded.experiments] == [e.config for e in lab.experiments]


def test_old_manifests_name_the_lab_through_an_entry_point(tmp_path, register_laboratory):
    lab = make_lab(tmp_path)
    lab.setup()
    manifest = read_manifest(lab.manifest_path)
    del manifest["laboratory"]  # older manifests only stored a type name
    manifest["experiment_type"] = "toy"
    lab.manifest_path.write_text(json.dumps(manifest))

    register_laboratory(toy="toy:ToyLab")
    loaded = load_laboratory(lab.manifest_path)
    assert isinstance(loaded, ToyLab)
    assert loaded.num_experiments == 2