"""The Snakemake example must keep working as mechane changes.

Snakemake itself is not a dependency of mechane, so these tests check the parts that can drift:
the example's laboratory and params file, its analysis script against the real `export` columns,
and the mechane commands and options that its Snakefile calls.
"""

import ast
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
from mechane.core.params import load_labs
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


def example_labs():
    return {lab.name: lab for lab in (p.build_laboratory() for p in load_labs(Path("params.py")))}


def test_the_params_file_defines_two_labs_with_their_own_settings(example):
    labs = example_labs()

    assert sorted(labs) == ["decay", "half_life"]
    shapes = {
        name: (lab.num_experiments, lab.num_instances, lab.seed) for name, lab in labs.items()
    }
    assert shapes == {"decay": (6, 20, 1), "half_life": (3, 50, 2)}
    for name, lab in labs.items():
        assert lab.root == example / "outputs" / name  # the Snakefile relies on this layout


def test_the_snakefile_lists_the_labs_of_the_params_file(example):
    snakefile = (example / "Snakefile").read_text()
    listed = re.search(r"^LABS = (\[.*\])", snakefile, re.MULTILINE)

    assert listed is not None
    assert sorted(ast.literal_eval(listed.group(1))) == sorted(example_labs())


@pytest.mark.parametrize(
    ("name", "output", "exact"),
    [
        ("decay", "surviving_fraction", math.exp(-0.5)),
        ("half_life", "half_life", math.log(2) / 0.1),
    ],
)
def test_each_example_lab_estimates_the_exact_answer(example, name, output, exact):
    lab = example_labs()[name]
    lab.setup()
    first = lab.experiments[0]  # the smallest rate
    assert not run_batch(lab, 0, range(lab.num_instances)).failed
    estimates = [r["outputs"][output] for _, r in first.iter_results()]

    assert sum(estimates) / len(estimates) == pytest.approx(exact, rel=0.03)


def test_the_analysis_script_reads_what_export_writes(example):
    tables = []
    for name, lab in example_labs().items():
        lab.setup()
        for experiment_id in range(lab.num_experiments):
            run_batch(lab, experiment_id, range(lab.num_instances))
        table = example / f"{name}.csv"
        exported = CliRunner().invoke(
            export, ["--manifest-path", str(lab.manifest_path), "--output", str(table)]
        )
        assert exported.exit_code == 0, exported.output
        tables.append(str(table))

    result = subprocess.run(
        [sys.executable, "summarize.py", *tables], capture_output=True, text=True, check=True
    )

    decay, half_life = (block.splitlines() for block in result.stdout.split("\n\n"))
    assert (decay[0], half_life[0]) == ("decay: surviving_fraction", "half_life: half_life")
    assert decay[1].split() == ["rate", "n_atoms", "t", "estimate", "+/-", "exact"]
    assert half_life[1].split() == ["rate", "n_atoms", "estimate", "+/-", "exact"]
    assert (len(decay), len(half_life)) == (2 + 6, 2 + 3)  # one row per experiment
    for line in decay[2:]:
        rate, _, t, estimate, _, exact = (float(x) for x in line.split())
        assert exact == pytest.approx(math.exp(-rate * t), abs=1e-3)
        assert estimate == pytest.approx(exact, abs=0.05)
    for line in half_life[2:]:
        rate, _, estimate, _, exact = (float(x) for x in line.split())
        assert exact == pytest.approx(math.log(2) / rate, rel=1e-3)
        assert estimate == pytest.approx(exact, rel=0.05)


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
    plans = {name: plan_to_dict(build_plan(lab, 5)) for name, lab in example_labs().items()}

    keys = set(re.findall(r'plan\["(\w+)"\]', snakefile))

    assert "num_tasks" in keys
    assert all(keys <= plan.keys() for plan in plans.values())
    # `tasks` is grouped by experiment, so its length is the number of groups, not of tasks.
    assert 'len(plan["tasks"])' not in snakefile
    assert (plans["decay"]["num_tasks"], len(plans["decay"]["tasks"])) == (24, 6)
    assert (plans["half_life"]["num_tasks"], len(plans["half_life"]["tasks"])) == (30, 3)
