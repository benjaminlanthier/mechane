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

from mechane import expand, grid, to_jsonable, zipped
from mechane.utils import provenance
from mechane.utils.ranges import compact_ranges, parse_ranges
from mechane.utils.serialization import flatten
from toy import Kind


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
        "path": str(Path("a") / Path("b")),
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
        "path": str(Path("a") / Path("b")),
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
        if command[1] == "status":
            assert "--untracked-files=no" in command
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

    # Untracked files (logs, results, __pycache__) say nothing about the code that ran.
    provenance._git_info.cache_clear()
    (tmp_path / "untracked.log").write_text("noise")
    untracked = provenance._git_info(str(tmp_path))
    assert untracked is not None and untracked["dirty"] is False

    provenance._git_info.cache_clear()
    tracked.write_text("two")
    dirty = provenance._git_info(str(tmp_path))
    assert dirty is not None
    assert dirty["commit"] == clean["commit"]
    assert dirty["dirty"] is True


# --------------------------------------------------------------------------- ranges
@pytest.mark.parametrize(
    ("ids", "text"),
    [
        ([], ""),
        ([3], "3"),
        ([0, 1, 2, 5, 7, 8], "0-2,5,7-8"),
        ([8, 7, 5, 2, 1, 0, 1], "0-2,5,7-8"),  # unsorted, with a duplicate
        ([4, 5], "4-5"),
    ],
)
def test_compact_ranges(ids, text):
    assert compact_ranges(ids) == text


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("", []),
        ("3", [3]),
        ("0-2,5", [0, 1, 2, 5]),
        (" 3 , 4-5 ,, ", [3, 4, 5]),  # blanks and empty parts are ignored
        ("5,1-2,2", [5, 1, 2]),  # first-seen order, no duplicates
        ("7-7", [7]),
    ],
)
def test_parse_ranges(text, ids):
    assert parse_ranges(text) == ids


@pytest.mark.parametrize("text", ["a", "-1", "1-", "1--3", "0x10", "1.5"])
def test_parse_ranges_rejects_what_is_not_an_id_or_range(text):
    with pytest.raises(ValueError, match="invalid id or range"):
        parse_ranges(text)


def test_parse_ranges_rejects_descending_ranges():
    with pytest.raises(ValueError, match="empty range '3-1'"):
        parse_ranges("0,3-1")


def test_the_two_notations_are_inverses():
    ids = [0, 1, 2, 5, 9, 10, 11]
    assert parse_ranges(compact_ranges(ids)) == ids


# --------------------------------------------------------------------------- flatten
def test_flatten_joins_nested_keys_with_dots():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_flatten_leaves_lists_and_empty_mappings_alone():
    data = {"a": [1, {"b": 2}], "empty": {}, "x": {"y": []}}
    assert flatten(data) == {"a": [1, {"b": 2}], "empty": {}, "x.y": []}


def test_flatten_accepts_a_prefix():
    assert flatten({"b": {"c": 1}}, "a.") == {"a.b.c": 1}
