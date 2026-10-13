"""The Snakemake example must keep working as mechane changes.

Snakemake itself is not a dependency of mechane, so these tests check the parts that can drift:
the example's laboratory and params file, its analysis script against the real `export` columns,
and the mechane commands and options that its Snakefile calls.
"""

import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from mechane.cli import export, main
from mechane.core.params import load_params_file
from mechane.core.plan import build_plan, plan_to_dict
from mechane.core.runner import run_batch

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "snakemake"


@pytest.fixture
def example(tmp_path, monkeypatch):
    """A copy of the example project, importable, with the current directory inside it."""
    project = tmp_path / "project"
    shutil.copytree(EXAMPLE, project)
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.chdir(project)
    yield project
    sys.modules.pop("decay", None)
    sys.modules.pop("external_params", None)


def test_the_example_lab_estimates_the_exact_answer(example):
    lab = load_params_file(Path("params.py")).build_laboratory()
    lab.setup()
    assert (lab.num_experiments, lab.num_instances) == (6, 20)

    first = lab.experiments[0]  # rate 0.5, 1000 atoms
    assert not run_batch(lab, 0, range(20)).failed
    estimates = [r["outputs"]["surviving_fraction"] for _, r in first.iter_results()]

    assert sum(estimates) / len(estimates) == pytest.approx(math.exp(-0.5), abs=0.02)


def test_the_analysis_script_reads_what_export_writes(example):
    lab = load_params_file(Path("params.py")).build_laboratory()
    lab.setup()
    for experiment_id in range(lab.num_experiments):
        run_batch(lab, experiment_id, range(lab.num_instances))
    table = example / "table.csv"
    exported = CliRunner().invoke(
        export, ["--manifest-path", str(lab.manifest_path), "--output", str(table)]
    )
    assert exported.exit_code == 0, exported.output

    result = subprocess.run(
        [sys.executable, "summarize.py", str(table)], capture_output=True, text=True, check=True
    )

    lines = result.stdout.splitlines()
    assert lines[0].split() == ["rate", "n_atoms", "estimate", "+/-", "exact"]
    assert len(lines) == 1 + 6  # one row per experiment
    for line in lines[1:]:
        rate, _, estimate, _, exact = (float(x) for x in line.split())
        assert exact == pytest.approx(math.exp(-rate), abs=1e-3)
        assert estimate == pytest.approx(exact, abs=0.05)


def test_the_snakefile_calls_only_mechane_commands_and_options_that_exist():
    snakefile = (EXAMPLE / "Snakefile").read_text()
    calls = re.findall(r'"mechane ([a-z-]+)([^"]*)"', snakefile)

    assert {name for name, _ in calls} >= {"setup", "plan", "run-plan", "status", "export"}
    for name, arguments in calls:
        assert name in main.commands, f"`mechane {name}` does not exist"
        command = main.commands[name]
        known = {
            opt
            for param in command.params
            if isinstance(param, click.Option)
            for opt in param.opts + param.secondary_opts
        }
        for flag in re.findall(r"--[a-z][a-z-]*", arguments):
            assert flag in known, f"`mechane {name}` has no option {flag}"


def test_the_snakefile_counts_tasks_with_num_tasks_and_reads_only_keys_a_plan_has(example):
    snakefile = (example / "Snakefile").read_text()
    lab = load_params_file(Path("params.py")).build_laboratory()
    plan = plan_to_dict(build_plan(lab, 5))

    keys = set(re.findall(r'plan\["(\w+)"\]', snakefile))

    assert "num_tasks" in keys
    assert keys <= plan.keys()
    # `tasks` is grouped by experiment, so its length is the number of groups, not of tasks.
    assert 'len(plan["tasks"])' not in snakefile
    assert plan["num_tasks"] == 24 and len(plan["tasks"]) == 6
