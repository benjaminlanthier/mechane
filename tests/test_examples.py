"""The example project must keep working as mechane changes.

`examples/radioactivity` is a uv workspace: its root is a science package (`radioactivity`) and
the workflow that runs the experiments (Snakefile, params file, analysis); its `experiments`
member holds the mechane labs. Neither package is installed here, so these tests put both `src/`
directories on the path and register the member's entry points from its own `pyproject.toml`.
They check the parts that can drift: the workspace wiring, the labs and their params file, the
analysis script against the real `export` columns, and the mechane commands and options that the
Snakefile calls.
"""

import ast
import math
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from mechane.cli import export, main
from mechane.core.params import load_labs
from mechane.core.plan import build_plan, plan_to_dict
from mechane.core.runner import run_batch

REPOSITORY = Path(__file__).resolve().parent.parent
EXAMPLE = REPOSITORY / "examples" / "radioactivity"
EXPERIMENTS = EXAMPLE / "experiments"


def pyproject(directory: Path) -> dict:
    return tomllib.loads((directory / "pyproject.toml").read_text())


@pytest.fixture
def example(tmp_path, monkeypatch, register_laboratory):
    """A copy of the example as `uv run --group workflow` would run it: both packages importable,
    the labs' entry points registered, and the current directory at the workspace root."""
    project = tmp_path / "radioactivity"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    monkeypatch.syspath_prepend(str(project / "src"))
    monkeypatch.syspath_prepend(str(project / "experiments" / "src"))
    register_laboratory(**pyproject(EXPERIMENTS)["project"]["entry-points"]["mechane.laboratories"])
    monkeypatch.chdir(project)
    yield project
    for module in [m for m in sys.modules if m.startswith("radioactivity")]:
        sys.modules.pop(module)
    sys.modules.pop("external_params", None)


def example_labs():
    return {lab.name: lab for lab in (p.build_laboratory() for p in load_labs(Path("params.py")))}


# --------------------------------------------------------------------------- the workspace
def test_the_experiments_are_a_member_of_the_science_packages_workspace():
    root, member = pyproject(EXAMPLE), pyproject(EXPERIMENTS)

    assert root["tool"]["uv"]["workspace"]["members"] == ["experiments"]
    # The root's `workflow` group installs the labs next to the science package.
    assert "radioactivity-experiments" in root["dependency-groups"]["workflow"]
    assert root["tool"]["uv"]["sources"]["radioactivity-experiments"] == {"workspace": True}
    assert set(member["project"]["dependencies"]) == {"radioactivity", "mechane"}
    sources = member["tool"]["uv"]["sources"]
    assert sources["radioactivity"] == {"workspace": True}
    # The example runs the mechane of this repository, not a released one.
    assert (EXPERIMENTS / sources["mechane"]["path"]).resolve() == REPOSITORY


def test_the_science_package_does_not_depend_on_mechane():
    assert pyproject(EXAMPLE)["project"]["dependencies"] == ["numpy>=1.26"]
    for source in (EXAMPLE / "src").rglob("*.py"):
        imported = set()
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "mechane" not in imported, source


def test_the_science_packages_own_tests_pass(tmp_path):
    project = tmp_path / "radioactivity"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist"],
        cwd=project,
        env={**os.environ, "PYTHONPATH": str(project / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_entry_points_name_the_labs_they_point_to(example):
    for name, lab in example_labs().items():
        target = pyproject(EXPERIMENTS)["project"]["entry-points"]["mechane.laboratories"][name]
        assert target == f"{type(lab).__module__}:{type(lab).__qualname__}"


# --------------------------------------------------------------------------- the labs
def test_the_params_file_defines_two_labs_with_their_own_settings(example):
    labs = example_labs()

    assert sorted(labs) == ["decay", "half_life"]
    shapes = {
        name: (lab.num_experiments, lab.num_instances, lab.seed) for name, lab in labs.items()
    }
    assert shapes == {"decay": (6, 20, 1), "half_life": (3, 50, 2)}
    for name, lab in labs.items():
        assert lab.root == example / "outputs" / name  # the Snakefile relies on this layout


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
        [sys.executable, "summarize.py", *tables],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(example / "src")},  # it imports the science
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


# --------------------------------------------------------------------------- the Snakefile
def test_the_snakefile_lists_the_labs_of_the_params_file(example):
    snakefile = (example / "Snakefile").read_text()
    listed = re.search(r"^LABS = (\[.*\])", snakefile, re.MULTILINE)

    assert listed is not None
    assert sorted(ast.literal_eval(listed.group(1))) == sorted(example_labs())


def test_the_snakefile_names_files_that_exist():
    snakefile = (EXAMPLE / "Snakefile").read_text()
    for path in re.findall(r'^\s+"([\w/.]+\.py)",', snakefile, re.MULTILINE):
        assert (EXAMPLE / path).is_file(), path


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
