"""`mechane.core.doctor`: consistency checks on what a laboratory left on disk."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from mechane import Configs, Experiment, ExperimentConfig, Laboratory
from mechane.core.doctor import code_version, diagnose
from toy import ToyLab

PARAMS = {
    "seed": 7,
    "num_instances": 6,
    "model": {"L": 2},
    "solver": {"sigma": 1.0, "n_samples": 20},
}


def edit_record(experiment, index, **fields):
    path = experiment.results_path(index)
    record = json.loads(path.read_text())
    record.update(fields)
    path.write_text(json.dumps(record))


def set_git(experiment, instance_id, commit, dirty=False):
    """Pin the code version of a record: the real one depends on where the tests are run."""
    path = experiment.results_path(instance_id)
    record = json.loads(path.read_text())
    record["meta"]["git"] = None if commit is None else {"commit": commit, "dirty": dirty}
    path.write_text(json.dumps(record))


def codes(findings):
    return sorted(f.code for f in findings)


# --------------------------------------------------------------------------- a healthy laboratory
def test_a_consistent_laboratory_has_no_findings(lab, run_instances):
    run_instances(lab, 0, range(6))
    run_instances(lab, 1, [0, 1])
    assert diagnose(lab) == []


def test_a_laboratory_that_was_never_set_up_has_no_findings(tmp_path):
    lab = ToyLab.from_configs(tmp_path / "never-created", PARAMS)
    assert diagnose(lab) == []


# --------------------------------------------------------------------------- results
def test_a_wrong_seed_or_instance_id_is_an_error(lab, run_instances):
    run_instances(lab, 0, range(4))
    exp = lab.experiment(0)
    edit_record(exp, 1, seed=exp.instance_seed(1) + 1)
    edit_record(exp, 3, instance_id=2)

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code, finding.experiment_id) == (
        "error",
        "record-mismatch",
        0,
    )
    assert finding.instance_ids == (1, 3)
    assert "instances 1,3: the recorded" in finding.message


@pytest.mark.parametrize("content", ["{truncated", "[]", json.dumps({"outputs": {}})])
def test_unreadable_or_incomplete_results_are_errors(lab, run_instances, content):
    run_instances(lab, 0, [0, 1])
    lab.experiment(0).results_path(1).write_text(content)

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code, finding.instance_ids) == (
        "error",
        "unreadable-result",
        (1,),
    )


def test_a_result_next_to_a_job_err_is_a_stale_warning(lab, run_instances, fail_instance):
    run_instances(lab, 0, [0, 1])
    fail_instance(lab, 0, 1, "old traceback")

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code, finding.instance_ids) == (
        "warning",
        "stale-error",
        (1,),
    )


def test_instance_directories_past_the_end_are_reported(lab):
    exp = lab.experiment(1)
    for stray in ("6", "7", "9"):
        (exp.instances_dir / stray).mkdir()
    (exp.instances_dir / "notes").mkdir()  # not numeric: not an instance directory at all

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code, finding.experiment_id) == (
        "warning",
        "stray-instance",
        1,
    )
    assert finding.instance_ids == (6, 7, 9)
    assert "6-7,9 are outside 0..5" in finding.message


# --------------------------------------------------------------------------- code versions
def test_results_from_several_code_versions_are_flagged(lab, run_instances):
    run_instances(lab, 0, range(6))
    exp = lab.experiment(0)
    for i in (0, 1, 2):
        set_git(exp, i, "aaaaaaa1111", dirty=False)
    for i in (3, 4):
        set_git(exp, i, "aaaaaaa1111", dirty=True)
    set_git(exp, 5, "bbbbbbb2222", dirty=False)

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code, finding.instance_ids) == (
        "warning",
        "mixed-code",
        (0, 1, 2, 3, 4, 5),
    )
    assert "3 code versions" in finding.message
    assert "aaaaaaa (clean): 0-2" in finding.message
    assert "aaaaaaa (dirty): 3-4" in finding.message
    assert "bbbbbbb (clean): 5" in finding.message


def test_one_code_version_is_not_flagged(lab, run_instances):
    run_instances(lab, 0, range(3))
    for i in range(3):
        set_git(lab.experiment(0), i, "aaaaaaa1111", dirty=True)
    assert diagnose(lab) == []


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"git": {"commit": "0123456789", "dirty": False}}, "0123456 (clean)"),
        ({"git": {"commit": "0123456789", "dirty": True}}, "0123456 (dirty)"),
        (
            {"git": None, "packages": {"numpy": "2", "mechane": "0.1"}},
            "packages mechane=0.1, numpy=2",
        ),
        ({"git": {"commit": ""}, "packages": {"numpy": "2"}}, "packages numpy=2"),
        ({"git": None, "packages": {}}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_code_version_prefers_git_and_falls_back_to_packages(meta, expected):
    assert code_version({"meta": meta}) == expected


def test_a_record_without_meta_has_an_unknown_code_version():
    assert code_version({}) == "unknown"
    assert code_version({"meta": None}) == "unknown"


# --------------------------------------------------------------------------- lab-wide checks
def test_a_relative_root_is_a_warning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lab = ToyLab.from_configs(Path("results"), PARAMS)

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code, finding.experiment_id) == (
        "warning",
        "relative-root",
        None,
    )
    assert "relative root (results)" in finding.message


def test_directories_the_manifest_does_not_use_are_orphans(lab):
    (lab.root / "0123456789abcdef").mkdir()
    (lab.root / ".cache").mkdir()  # hidden: never reported
    (lab.root / "notes.txt").write_text("a file, not a directory")

    (finding,) = diagnose(lab)

    assert (finding.severity, finding.code) == ("warning", "orphan-directory")
    assert "1 directory under" in finding.message
    assert "is not used by this manifest" in finding.message
    assert "0123456789abcdef" in finding.message


def test_many_orphans_are_summarised(lab):
    for i in range(7):
        (lab.root / f"orphan{i}").mkdir()

    (finding,) = diagnose(lab)

    assert "7 directories under" in finding.message
    assert "are not used by this manifest" in finding.message
    assert "orphan0, orphan1, orphan2, orphan3, orphan4, and 2 more" in finding.message


def test_checking_chosen_experiments_skips_the_laboratory_wide_checks(lab):
    (lab.root / "orphan").mkdir()
    assert diagnose(lab, [lab.experiment(0)]) == []
    assert codes(diagnose(lab)) == ["orphan-directory"]


def test_checking_chosen_experiments_still_checks_those_experiments(lab, run_instances):
    run_instances(lab, 0, [0])
    edit_record(lab.experiment(0), 0, seed=-1)
    assert diagnose(lab, [lab.experiment(1)]) == []
    assert codes(diagnose(lab, [lab.experiment(0)])) == ["record-mismatch"]


@dataclass(frozen=True)
class NoSections(Configs):
    pass


@dataclass(frozen=True)
class NoSectionsConfig(ExperimentConfig):
    configs: NoSections


class NoSectionsExperiment(Experiment):
    config: NoSectionsConfig


def test_an_experiment_without_sections_does_not_make_its_own_directory_an_orphan(tmp_path):
    config = NoSectionsConfig(experiment_id=0, num_instances=3, seed=0, configs=NoSections())
    experiment = NoSectionsExperiment(tmp_path, config)
    experiment.instances_dir.mkdir(parents=True)  # <root>/instances: no section hash above it
    lab = Laboratory(root=tmp_path, experiments=[experiment], num_instances=3, seed=0)

    assert diagnose(lab) == []

    (tmp_path / "leftover").mkdir()
    assert codes(diagnose(lab)) == ["orphan-directory"]
