"""How an experiment's instances are found on disk.

`status`, `iter_results`, `doctor` and `clean` look at the instances that have a directory, from
one directory listing, instead of probing every id below `num_instances`. The reference
implementations below are the probing versions, kept here to prove that nothing else changed.
"""

import json
import os
import random
import time

import pytest

from mechane.core.clean import find_targets
from mechane.core.doctor import diagnose
from toy import ToyLab


def make_experiment(tmp_path, num_instances):
    params = {"num_instances": num_instances, "model": {"L": 2}, "solver": {"sigma": 1.0}}
    lab = ToyLab.from_configs(tmp_path / "results", params)
    assert isinstance(lab, ToyLab)
    return lab, lab.experiments[0]


def reference_status(exp):
    """The probing implementation: one `exists` per instance and per file."""
    done, failed, missing = [], [], []
    for i in range(exp.config.num_instances):
        if exp.results_path(i).exists():
            done.append(i)
        elif exp.error_path(i).exists():
            failed.append(i)
        else:
            missing.append(i)
    return {"done": done, "failed": failed, "missing": missing}


def reference_results(exp):
    out = []
    for i in range(exp.config.num_instances):
        path = exp.results_path(i)
        if path.exists():
            with path.open() as f:
                out.append((i, json.load(f)))
    return out


def put(exp, i, results=False, error=False):
    directory = exp.instance_dir(i)
    directory.mkdir(parents=True, exist_ok=True)
    if results:
        exp.results_path(i).write_text(json.dumps({"instance": i}))
    if error:
        exp.error_path(i).write_text("Traceback\nBoom")


# --------------------------------------------------------------------------- equivalence
def test_a_messy_tree_is_read_exactly_as_probing_would_read_it(tmp_path):
    _, exp = make_experiment(tmp_path, 25)  # ids 00..24, two digits
    put(exp, 0, results=True)
    put(exp, 1, error=True)
    put(exp, 2, results=True, error=True)  # a stale job.err next to a result: done
    put(exp, 3)  # a directory with nothing in it: missing
    put(exp, 24, results=True)  # the last valid id
    exp.instance_dir(6).parent.joinpath("06").write_text("a file named like an instance")
    exp.instance_dir(10).mkdir()
    exp.results_path(
        10
    ).mkdir()  # `results.json` is a directory: it exists, so the instance is done
    base = exp.instances_dir
    for stray in ("7", "007", "25", "99", "ab", "\u00b2\u00b3"):  # none is an instance
        (base / stray).mkdir()
        (base / stray / "results.json").write_text("{}")

    assert exp.status() == reference_status(exp)
    assert exp.status()["done"] == [0, 2, 10, 24]
    assert exp.status()["failed"] == [1]


def test_status_and_results_match_probing_on_random_trees(tmp_path):
    rng = random.Random(2026)
    for trial in range(15):
        _, exp = make_experiment(
            tmp_path / f"trial{trial}", rng.choice([1, 2, 9, 10, 11, 100, 101])
        )
        for i in range(exp.config.num_instances):
            kind = rng.choice(["none", "dir", "results", "error", "both"])
            if kind != "none":
                put(exp, i, results=kind in ("results", "both"), error=kind in ("error", "both"))
        assert exp.status() == reference_status(exp), trial
        assert list(exp.iter_results()) == reference_results(exp), trial


def test_an_experiment_whose_directory_does_not_exist_has_everything_missing(tmp_path):
    _, exp = make_experiment(tmp_path, 4)
    assert exp.present_instance_ids() == []
    assert exp.status() == {"done": [], "failed": [], "missing": [0, 1, 2, 3]}
    assert list(exp.iter_results()) == []


def test_an_instances_path_that_is_a_file_has_everything_missing(tmp_path):
    _, exp = make_experiment(tmp_path, 3)
    exp.instances_dir.parent.mkdir(parents=True)
    exp.instances_dir.write_text("not a directory")
    assert exp.present_instance_ids() == []
    assert exp.status()["missing"] == [0, 1, 2]


# --------------------------------------------------------------------------- present_instance_ids
def test_present_ids_are_ascending_whatever_order_they_were_created_in(tmp_path):
    _, exp = make_experiment(tmp_path, 30)
    for i in (17, 3, 29, 0, 11):
        put(exp, i)
    assert exp.present_instance_ids() == [0, 3, 11, 17, 29]


def test_only_canonically_named_directories_below_num_instances_are_present(tmp_path):
    _, exp = make_experiment(tmp_path, 12)  # width 2
    put(exp, 5)
    for name in ("5", "005", "12", "13", "ab", "1a"):
        (exp.instances_dir / name).mkdir()
    (exp.instances_dir / "07").write_text("a file")

    assert exp.present_instance_ids() == [5]


# --------------------------------------------------------------------------- results
def test_iter_results_skips_a_directory_whose_instance_has_not_finished(tmp_path):
    _, exp = make_experiment(tmp_path, 5)
    put(exp, 1, results=True)
    put(exp, 2)  # started, never finished
    put(exp, 3, error=True)
    put(exp, 4, results=True)

    assert [i for i, _ in exp.iter_results()] == [1, 4]


# --------------------------------------------------------------------------- not O(num_instances)
@pytest.fixture
def huge(tmp_path):
    """One million instances, three of which have a directory."""
    lab, exp = make_experiment(tmp_path, 1_000_000)
    put(exp, 0, results=True)
    put(exp, 1, error=True)
    put(exp, 999_999)
    return lab, exp


def timed(function, *args):
    start = time.perf_counter()
    value = function(*args)
    return value, time.perf_counter() - start


def test_status_does_not_probe_every_instance(huge):
    # Probing a million instances one by one took about a minute. Listing takes well under a second.
    _, exp = huge

    status, seconds = timed(exp.status)

    assert (len(status["done"]), len(status["failed"]), len(status["missing"])) == (1, 1, 999_998)
    assert seconds < 15


def count_calls(monkeypatch, exp, *names):
    """Wrap the given path methods of `exp` and return the list that records every call."""
    calls = []
    for name in names:
        real = getattr(exp, name)
        monkeypatch.setattr(exp, name, lambda i, real=real: calls.append(i) or real(i))
    return calls


def test_doctor_does_not_probe_every_instance(huge, monkeypatch):
    lab, exp = huge
    calls = count_calls(monkeypatch, exp, "results_path", "error_path")

    diagnose(lab, [exp])

    assert len(calls) <= 2 * 3  # a path per instance that has a directory, not per instance


def test_clean_does_not_probe_every_instance(huge, monkeypatch):
    _, exp = huge
    calls = count_calls(monkeypatch, exp, "results_path", "error_path")

    targets = find_targets([exp], "errors")

    assert [(t.instance_id, t.path.name) for t in targets] == [(1, "job.err")]
    assert len(calls) <= 2 * 3  # a path or two per instance that has a directory


def test_present_ids_come_from_a_listing_not_from_probing(huge, monkeypatch):
    _, exp = huge
    probed = []
    real_exists = os.path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: probed.append(p) or real_exists(p))

    exp.status()

    # At most two probes (results, then error) per directory that exists, never one per instance.
    assert len(probed) <= 2 * 3
