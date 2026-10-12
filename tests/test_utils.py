"""Tests for `mechane.utils`: serialization, sweeps, laboratory resolution and provenance."""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mechane import Laboratory, expand, grid, to_jsonable, zipped
from mechane.core.registry import laboratory_ref, resolve_laboratory
from mechane.utils import provenance
from toy import Kind, ToyLab


# --------------------------------------------------------------------------- serialization
@dataclass
class Point:
    x: int
    kind: Kind


def test_to_jsonable_converts_every_supported_type():
    obj = {
        "enum": Kind.UNIFORM,
        "dataclass": Point(1, Kind.GAUSS),
        "tuple": (1, 2),
        "set": {3, 1, 2},
        "frozenset": frozenset({"b", "a"}),
        "path": Path("a/b"),
        "array": np.arange(3),
        "scalar": np.float32(0.5),
        "none": None,
    }
    assert to_jsonable(obj) == {
        "enum": "uniform",
        "dataclass": {"x": 1, "kind": "gauss"},
        "tuple": [1, 2],
        "set": [1, 2, 3],  # sets are sorted so the output is deterministic
        "frozenset": ["a", "b"],
        "path": "a/b",
        "array": [0, 1, 2],
        "scalar": 0.5,
        "none": None,
    }


def test_to_jsonable_stringifies_non_string_keys():
    assert to_jsonable({1: "a", np.int64(2): "b"}) == {"1": "a", "2": "b"}


@pytest.mark.parametrize("unsupported", [object(), len, {"nested": [object()]}])
def test_to_jsonable_rejects_what_it_cannot_serialize(unsupported):
    with pytest.raises(TypeError, match="Cannot serialize"):
        to_jsonable(unsupported)


# --------------------------------------------------------------------------- sweeps
@pytest.mark.parametrize(("not_a_spec", "type_name"), [(42, "int"), ([{"a": 1}], "list")])
def test_expand_rejects_what_is_not_a_spec(not_a_spec, type_name):
    with pytest.raises(TypeError, match=f"Expected a dict or Sweep, got {type_name}"):
        expand(not_a_spec)


def test_product_of_sweeps_rejects_shared_keys():
    with pytest.raises(ValueError, match=r"Cannot multiply sweeps sharing keys: \['a'\]"):
        (zipped(a=[1, 2]) * zipped(a=[3])).points()


@pytest.mark.parametrize("bad_axis", [5, {"x": 1}, "ab", b"ab"])
def test_axes_must_be_sequences_of_values(bad_axis):
    with pytest.raises(TypeError, match="must be a list/tuple/array of values"):
        grid(a=bad_axis)


# --------------------------------------------------------------------------- registry
class Holder:
    class Inner(ToyLab):
        pass


def test_laboratory_references_round_trip():
    assert laboratory_ref(ToyLab) == "toy:ToyLab"
    assert resolve_laboratory("toy:ToyLab") is ToyLab
    # Nested classes are reached through their dotted qualified name.
    assert laboratory_ref(Holder.Inner).endswith(":Holder.Inner")
    assert resolve_laboratory(laboratory_ref(Holder.Inner)) is Holder.Inner


def test_entry_point_names_resolve(register_laboratory):
    register_laboratory(toy="toy:ToyLab")
    lab: type[Laboratory] = resolve_laboratory("toy")
    assert lab is ToyLab


def test_unknown_entry_point_lists_the_registered_names(register_laboratory):
    register_laboratory(toy="toy:ToyLab", other="toy:ToyLab")
    with pytest.raises(
        KeyError, match=r"Unknown laboratory 'nope'.*Registered: \['other', 'toy'\]"
    ):
        resolve_laboratory("nope")


def test_unknown_entry_point_without_any_registered_lab(register_laboratory):
    register_laboratory()
    with pytest.raises(KeyError, match=r"Registered: \[\]"):
        resolve_laboratory("toy")


# --------------------------------------------------------------------------- provenance
@pytest.fixture(autouse=True)
def fresh_git_cache():
    """`_git_info` is memoized per directory: start and finish each test with a clean cache."""
    provenance._git_info.cache_clear()
    yield
    provenance._git_info.cache_clear()


def test_collect_provenance_records_an_allow_list_only(monkeypatch):
    for key in provenance.SLURM_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")

    info = provenance.collect_provenance(("numpy", "no-such-distribution-xyz"))

    assert info["slurm"] == {"SLURM_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "7"}
    assert info["packages"]["numpy"] == np.__version__
    assert info["packages"]["no-such-distribution-xyz"] is None  # not installed
    assert {"hostname", "platform", "python"} <= set(info)
    assert "git" not in info  # no directory to inspect
    assert "hunter2" not in json.dumps(info)  # the environment is never dumped


@pytest.mark.parametrize(("status_output", "dirty"), [("", False), (" M results.py\n", True)])
def test_git_info_reports_the_commit_and_whether_the_tree_is_dirty(
    tmp_path, monkeypatch, status_output, dirty
):
    outputs = {"rev-parse": "0123abc\n", "status": status_output}

    def fake_run(command, **kwargs):
        assert command[0] == "git"
        assert kwargs["cwd"] == str(tmp_path)
        return SimpleNamespace(stdout=outputs[command[1]])

    # Replace the name inside `provenance` only: the real `subprocess.run` is shared with
    # the standard library (platform.platform() uses it lazily).
    monkeypatch.setattr(provenance, "subprocess", SimpleNamespace(run=fake_run))

    assert provenance._git_info(str(tmp_path)) == {"commit": "0123abc", "dirty": dirty}
    info = provenance.collect_provenance(git_dir=tmp_path)
    assert info["git"] == {"commit": "0123abc", "dirty": dirty}


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, "git"),
        subprocess.TimeoutExpired("git", 5),
    ],
    ids=["git-not-installed", "not-a-repository", "timeout"],
)
def test_git_info_is_none_when_git_cannot_answer(tmp_path, monkeypatch, error):
    def failing_run(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(provenance, "subprocess", SimpleNamespace(run=failing_run))

    assert provenance._git_info(str(tmp_path)) is None
    assert provenance.collect_provenance(git_dir=tmp_path)["git"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_info_reads_a_real_repository(tmp_path, monkeypatch):
    for variable, value in {
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(variable, value)

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one")
    git("init", "-q")
    git("add", "tracked.txt")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "initial")

    clean = provenance._git_info(str(tmp_path))
    assert clean is not None
    assert re.fullmatch(r"[0-9a-f]{40,64}", clean["commit"])
    assert clean["dirty"] is False

    provenance._git_info.cache_clear()
    tracked.write_text("two")
    dirty = provenance._git_info(str(tmp_path))
    assert dirty is not None
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True
