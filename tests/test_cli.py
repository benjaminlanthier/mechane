"""Command-line behaviour that the end-to-end workflow test does not reach: the `python -m`
entry point, the different ways a params file can name its laboratory, and `run-batch`
argument handling and failure modes."""

import json
import runpy
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from mechane.cli import setup_lab, simulate_batch
from toy import ToyLab


def write_params(
    tmp_path: Path,
    *,
    header: str = "LABORATORY = 'toy:ToyLab'\n",
    sigma: float = 1.0,
    name: str = "params.py",
) -> Path:
    """A params file for the toy lab (1 experiment, 4 instances). `header` supplies LABORATORY."""
    f = tmp_path / name
    f.write_text(
        "from pathlib import Path\n"
        f"ROOT = Path({str(tmp_path / 'out')!r})\n"
        f"{header}"
        "PARAMS = {'seed': 1, 'num_instances': 4, 'model': {'L': 2},\n"
        f"          'solver': {{'sigma': {sigma}, 'n_samples': 10}}}}\n"
    )
    return f


def run_setup(tmp_path: Path, **kwargs) -> Path:
    """Run `setup` on a freshly written params file and return the manifest path."""
    params = write_params(tmp_path, **kwargs)
    result = CliRunner().invoke(setup_lab, ["--params-file", str(params)])
    assert result.exit_code == 0, result.output
    return tmp_path / "out" / "manifest.json"


# --------------------------------------------------------------------------- python -m
def test_python_dash_m_runs_the_cli(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mechane", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("mechane.cli", run_name="__main__")
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for command in ("setup", "run", "run-batch", "aggregate", "status"):
        assert command in out


# --------------------------------------------------------------------------- setup
def test_setup_reports_what_it_created(tmp_path):
    params = write_params(tmp_path)
    result = CliRunner().invoke(setup_lab, ["--params-file", str(params)])
    assert result.exit_code == 0, result.output
    assert "[Setup] 1 experiments x 4 instances" in result.output
    assert str(tmp_path / "out" / "manifest.json") in result.output


def test_setup_accepts_the_laboratory_class_itself(tmp_path):
    manifest = run_setup(tmp_path, header="from toy import ToyLab\nLABORATORY = ToyLab\n")
    assert json.loads(manifest.read_text())["laboratory"] == "toy:ToyLab"


def test_setup_accepts_the_legacy_experiment_type_name(tmp_path):
    manifest = run_setup(tmp_path, header="EXPERIMENT_TYPE = 'toy:ToyLab'\n")
    assert json.loads(manifest.read_text())["laboratory"] == "toy:ToyLab"


def test_setup_accepts_an_entry_point_name(tmp_path, register_laboratory):
    register_laboratory(toy="toy:ToyLab")
    manifest = run_setup(tmp_path, header="LABORATORY = 'toy'\n")
    # The manifest stores the importable reference, not the short name.
    assert json.loads(manifest.read_text())["laboratory"] == "toy:ToyLab"


def test_setup_requires_a_laboratory(tmp_path):
    params = write_params(tmp_path, header="")
    result = CliRunner().invoke(setup_lab, ["--params-file", str(params)])
    assert result.exit_code == 2
    assert "must define LABORATORY" in result.output
    assert not (tmp_path / "out").exists()


def test_setup_rejects_a_params_file_python_cannot_load(tmp_path):
    params = write_params(tmp_path, name="params.txt")  # no loader for a .txt suffix
    result = CliRunner().invoke(setup_lab, ["--params-file", str(params)])
    assert result.exit_code == 1
    assert isinstance(result.exception, ImportError)
    assert "Could not load module" in str(result.exception)


# --------------------------------------------------------------------------- run-batch
@pytest.mark.parametrize("ids", ["", ",", " , ", "-1,2", "3,-4"])
def test_run_batch_rejects_empty_or_negative_ids(tmp_path, ids):
    manifest = run_setup(tmp_path)
    result = CliRunner().invoke(
        simulate_batch,
        ["--manifest-path", str(manifest), "--experiment-id", "0", f"--instance-ids={ids}"],
    )
    assert result.exit_code == 2
    assert "non-empty list of non-negative integers" in result.output
    # Validation happens before anything runs.
    assert ToyLab.from_manifest(manifest).experiment(0).status()["done"] == []


def test_run_batch_reports_every_failure_and_still_runs_the_rest(tmp_path):
    manifest = run_setup(tmp_path, sigma=-1.0)  # every instance raises
    result = CliRunner().invoke(
        simulate_batch,
        ["--manifest-path", str(manifest), "--experiment-id", "0", "--instance-ids", "0,1"],
    )
    assert result.exit_code == 1
    assert "instance_id=0 FAILED" in result.output
    assert "instance_id=1 FAILED" in result.output
    assert "2/2 failed: [0, 1]" in result.output
    assert ToyLab.from_manifest(manifest).experiment(0).status()["failed"] == [0, 1]


def test_run_batch_fail_fast_stops_at_the_first_error(tmp_path):
    manifest = run_setup(tmp_path, sigma=-1.0)
    result = CliRunner().invoke(
        simulate_batch,
        [
            "--manifest-path",
            str(manifest),
            "--experiment-id",
            "0",
            "--instance-ids",
            "0,1,2",
            "--fail-fast",
        ],
    )
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)  # re-raised, not swallowed
    assert "negative sigma" in str(result.exception)
    # Instance 0 failed and left its traceback; 1 and 2 were never attempted.
    status = ToyLab.from_manifest(manifest).experiment(0).status()
    assert status == {"done": [], "failed": [0], "missing": [1, 2, 3]}
