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
from mechane.cli._output import (
    duration,
    echo_event,
    finish_batch,
    format_plan,
    format_table,
    size,
    slurm_time,
)
from mechane.core.plan import Plan, Task
from mechane.core.runner import (
    BatchResult,
    InstanceFailed,
    InstanceSaved,
    InstanceSkipped,
    InstanceStarted,
)
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


# --------------------------------------------------------------------------- wording
@pytest.mark.parametrize(
    ("event", "stream", "line"),
    [
        (
            InstanceSkipped(Path("/r/results.json")),
            "out",
            "[Job] /r/results.json already exists, skipping.",
        ),
        (InstanceStarted("toy", 1, 2), "out", "[Job] lab=toy experiment_id=1 instance_id=2"),
        (InstanceSaved(Path("/r/results.json")), "out", "[Save] /r/results.json"),
        (
            InstanceFailed(3, ValueError("boom")),
            "err",
            "[BatchJob] instance_id=3 FAILED: ValueError('boom')",
        ),
    ],
    ids=["skipped", "started", "saved", "failed"],
)
def test_run_events_are_worded_and_routed_by_the_cli(capsys, event, stream, line):
    echo_event(event)
    captured = capsys.readouterr()
    other = "err" if stream == "out" else "out"
    assert getattr(captured, stream) == line + "\n"
    assert getattr(captured, other) == ""


# --------------------------------------------------------------------------- shared formatting
@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (None, "-"),
        (0.0, "0ms"),
        (0.012, "12ms"),
        (0.9996, "1000ms"),
        (1, "1.0s"),
        (5.55, "5.5s"),
        (59.9, "59.9s"),
        (60, "1m00s"),
        (125, "2m05s"),
        (3599, "59m59s"),
        (3600, "1h00m"),
        (7300, "2h01m"),
    ],
)
def test_duration(seconds, text):
    assert duration(seconds) == text


@pytest.mark.parametrize(
    ("n", "text"),
    [
        (None, "-"),
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (5 * 1024**2, "5.0 MiB"),
        (3 * 1024**3, "3.0 GiB"),
        (2 * 1024**4, "2.0 TiB"),
        (5000 * 1024**4, "5000.0 TiB"),  # the largest unit is the last one
    ],
)
def test_size(n, text):
    assert size(n) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "00:01:00"),  # SLURM needs at least a minute
        (1, "00:01:00"),
        (60, "00:01:00"),
        (61, "00:02:00"),  # always rounded up
        (300, "00:05:00"),
        (3600, "01:00:00"),
        (86400, "1-00:00:00"),
        (90000, "1-01:00:00"),
    ],
)
def test_slurm_time(seconds, text):
    assert slurm_time(seconds) == text


def test_format_table_aligns_columns_without_trailing_spaces():
    text = format_table(("id", "name"), [(1, "a"), (10, "longer")])
    assert text.splitlines() == ["id  name", "1   a", "10  longer"]
    assert format_table(("only", "header"), []) == "only  header"
    assert format_table(("a", "b"), [("x", "")]).splitlines()[1] == "x"


def test_format_plan():
    empty = Plan(Path("m.json"), 4, ())
    assert format_plan(empty) == "Nothing to run: the plan has no tasks."

    plan = Plan(Path("m.json"), 4, (Task(0, (0, 1, 2, 3)), Task(1, (4, 6))))
    lines = format_plan(plan).splitlines()
    assert lines[0] == "6 instance(s) in 2 task(s) of up to 4: sbatch --array=0-1"
    assert lines[2] == "task  experiment  count  instances"
    assert lines[3].split() == ["0", "0", "4", "0-3"]
    assert lines[4].split() == ["1", "1", "2", "4,6"]


def test_finish_batch_reports_success_on_stdout(capsys):
    finish_batch(BatchResult(instance_ids=(0, 1, 2), failed=()))
    captured = capsys.readouterr()
    assert captured.out == "[BatchJob] Completed 3/3 instance(s).\n"
    assert captured.err == ""


def test_finish_batch_reports_failures_on_stderr_and_exits_with_status_1(capsys):
    with pytest.raises(SystemExit) as exit_info:
        finish_batch(BatchResult(instance_ids=(0, 1, 2), failed=(1,)))
    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == "[BatchJob] 1/3 failed: [1]\n"
    assert captured.out == ""
