"""`mechane.core.scaffold`: starting files for a laboratory."""

import sys

import pytest
from click.testing import CliRunner

from mechane.cli import aggregate_results, setup_lab, simulate_batch
from mechane.core.manifest import load_laboratory
from mechane.core.scaffold import ScaffoldError, render_lab, write_files


@pytest.fixture
def scaffolded(tmp_path, monkeypatch):
    """A generated lab, written to disk and importable, with the current directory inside it."""
    write_files(tmp_path, render_lab("scaffolded_lab"))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    sys.modules.pop("scaffolded_lab", None)
    sys.modules.pop("external_params", None)


# --------------------------------------------------------------------------- the lab skeleton
def test_the_generated_lab_runs_as_it_is(scaffolded):
    runner = CliRunner()
    result = runner.invoke(setup_lab, ["--params-file", "params.py"])
    assert result.exit_code == 0, result.output
    assert "2 experiments x 10 instances" in result.output  # the first section is a sweep

    manifest = scaffolded / "results" / "scaffolded_lab" / "manifest.json"
    args = ["--manifest-path", str(manifest), "--experiment-id", "0"]
    result = runner.invoke(simulate_batch, [*args, "--instance-ids", "0,1,2"])
    assert result.exit_code == 0, result.output
    assert runner.invoke(aggregate_results, ["--manifest-path", str(manifest)]).exit_code == 0

    lab = load_laboratory(manifest)
    assert lab.root.is_absolute()  # the generated params resolve ROOT
    (_, record), *_ = lab.experiment(0).iter_results()
    assert isinstance(record["outputs"]["sample"], float)


def test_default_sections_and_names():
    files = render_lab("walks")
    assert set(files) == {"walks.py", "params.py"}
    module = files["walks.py"]
    for expected in ("class Model(Config)", "class Solver(Config)", "class WalksLab(Laboratory)"):
        assert expected in module
    assert 'LABORATORY = "walks:WalksLab"' in files["params.py"]
    for text in files.values():
        compile(text, "<generated>", "exec")


def test_custom_sections_are_used_in_order_and_duplicates_are_dropped():
    files = render_lab("walks", ["walk", "sampling", "walk"])
    module, params = files["walks.py"], files["params.py"]
    assert module.index("class Walk(Config)") < module.index("class Sampling(Config)")
    assert module.count("class Walk(Config)") == 1
    assert "    walk: Walk\n    sampling: Sampling\n" in module
    assert '"walk": grid(example=[1, 2])' in params  # the first section is the swept one
    assert '"sampling": {"example": 1}' in params


@pytest.mark.parametrize("name", ["My-Lab", "1lab", "Lab", "class", "", "a b"])
def test_a_laboratory_name_must_be_a_lowercase_identifier(name):
    with pytest.raises(ScaffoldError, match=r"laboratory name .* must be a lowercase identifier"):
        render_lab(name)


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ([], "at least one section"),
        (["Bad"], "section name 'Bad' must be a lowercase identifier"),
        (["seed"], "'seed' is reserved"),
        (["configs"], "'configs' is reserved"),
        (["zip"], "'zip' is reserved"),
        (["config"], "class named Config, which clashes"),  # would shadow mechane's own Config
        (["instance"], "class named Instance, which clashes"),
        (["walk_lab"], "class named WalkLab, which clashes"),  # the lab's own class
        (["a_b", "a__b"], "class named AB, which clashes"),  # two sections, one class
    ],
)
def test_sections_that_would_produce_broken_code_are_rejected(sections, message):
    with pytest.raises(ScaffoldError, match=message):
        render_lab("walk", sections)


# --------------------------------------------------------------------------- writing files
def test_write_files_creates_the_directory_and_returns_the_paths(tmp_path):
    written = write_files(tmp_path / "new" / "dir", {"a.py": "A", "b.py": "B"})
    assert [p.name for p in written] == ["a.py", "b.py"]
    assert (tmp_path / "new" / "dir" / "b.py").read_text() == "B"


def test_write_files_refuses_to_overwrite_unless_forced(tmp_path):
    write_files(tmp_path, {"a.py": "old", "b.py": "old"})

    with pytest.raises(ScaffoldError, match=r"a\.py, b\.py already exist\(s\)"):
        write_files(tmp_path, {"a.py": "new", "b.py": "new"})
    assert (tmp_path / "a.py").read_text() == "old"  # nothing was written

    write_files(tmp_path, {"a.py": "new"}, force=True)
    assert (tmp_path / "a.py").read_text() == "new"
