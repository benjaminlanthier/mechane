"""`mechane.core.replay`: re-running an instance and comparing it with what was stored."""

import json

import pytest

from mechane.core.replay import ReplayError, compare_records, replay_instance
from toy import ToyInstance, ToyLab


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def edit_stored(lab, experiment_id, instance_id, edit):
    path = lab.experiment(experiment_id).results_path(instance_id)
    record = json.loads(path.read_text())
    edit(record)
    path.write_text(json.dumps(record))


# --------------------------------------------------------------------------- replaying
def test_a_replay_reproduces_the_stored_result(lab, run_instances, tmp_path):
    run_instances(lab, 0, [2])

    result = replay_instance(lab, 0, 2, tmp_path / "scratch")

    assert result.reproduced is True
    assert result.differences == ()
    assert result.saved is not None
    assert result.record["seed"] == result.saved["seed"]
    assert result.record["outputs"] == result.saved["outputs"]


def test_a_replay_leaves_the_results_tree_alone(lab, run_instances, tmp_path):
    run_instances(lab, 0, [1])
    before = tree(lab.root)

    replay_instance(lab, 0, 1, tmp_path / "scratch")
    replay_instance(lab, 0, 4, tmp_path / "scratch")  # an instance that never ran

    assert tree(lab.root) == before
    assert (tmp_path / "scratch").is_dir()  # the replays happened here instead


def test_an_unfinished_instance_is_replayed_with_nothing_to_compare(lab, tmp_path):
    result = replay_instance(lab, 1, 3, tmp_path / "scratch")

    assert result.saved is None
    assert result.reproduced is None
    assert result.differences == ()
    assert set(result.record) >= {"seed", "inputs", "outputs", "meta"}


def test_a_replay_that_disagrees_lists_the_differences(lab, run_instances, tmp_path):
    run_instances(lab, 0, [0])
    edit_stored(lab, 0, 0, lambda r: r["outputs"].update(mean_square=123.0))

    result = replay_instance(lab, 0, 0, tmp_path / "scratch")

    assert result.reproduced is False
    (difference,) = result.differences
    assert difference.startswith("outputs.mean_square: replay ")
    assert "!= stored 123.0" in difference


def test_a_relative_tolerance_absorbs_small_numerical_noise(lab, run_instances, tmp_path):
    run_instances(lab, 0, [0])
    edit_stored(
        lab,
        0,
        0,
        lambda r: r["outputs"].update(mean_square=r["outputs"]["mean_square"] * (1 + 1e-9)),
    )

    assert replay_instance(lab, 0, 0, tmp_path / "a").reproduced is False
    assert replay_instance(lab, 0, 0, tmp_path / "b", rtol=1e-6).reproduced is True


def test_an_unknown_experiment_or_instance_is_a_replay_error(lab, tmp_path):
    with pytest.raises(ReplayError, match=r"No experiment 9; valid ids: 0\.\.1"):
        replay_instance(lab, 9, 0, tmp_path)
    for bad in (-1, 6):
        with pytest.raises(ReplayError, match=r"instance_id .* out of range .*valid: 0\.\.5"):
            replay_instance(lab, 0, bad, tmp_path)


class WritingInstance(ToyInstance):
    """An instance that, like many real ones, writes a file next to its results."""

    def run(self):
        (self.dir / "big_array.npy").write_bytes(b"data")
        return super().run()


class WritingLab(ToyLab):
    instance_class = WritingInstance


def test_files_an_instance_writes_land_in_the_scratch_root(tmp_path):
    lab = WritingLab.from_configs(
        tmp_path / "results", {"model": {"L": 2}, "solver": {"sigma": 1.0, "n_samples": 5}}
    )
    real_instance_dir = lab.experiment(0).instance_dir(0)

    replay_instance(lab, 0, 0, tmp_path / "scratch")

    assert not real_instance_dir.exists()
    assert list((tmp_path / "scratch").rglob("big_array.npy"))


# --------------------------------------------------------------------------- comparing records
def record(**fields):
    return {"seed": 1, "inputs": {}, "outputs": {}, **fields}


def test_equal_records_have_no_differences():
    r = record(inputs={"a": [1, 2]}, outputs={"x": 1.5, "nested": {"y": "z"}})
    assert compare_records(r, json.loads(json.dumps(r))) == []


def test_only_seed_inputs_and_outputs_are_compared():
    replay = record(meta={"hostname": "node-1"}, wall_time_s=1)
    saved = record(meta={"hostname": "node-2"}, wall_time_s=2)
    assert compare_records(replay, saved) == []


def test_differences_name_the_field_that_differs():
    assert compare_records(record(seed=1), record(seed=2)) == ["seed: replay 1 != stored 2"]
    assert compare_records(record(outputs={"s": "a"}), record(outputs={"s": "b"})) == [
        "outputs.s: replay 'a' != stored 'b'"
    ]


def test_keys_present_on_one_side_only_are_reported():
    differences = compare_records(
        record(outputs={"only_replay": 1, "both": 1}),
        record(outputs={"only_stored": 2, "both": 1}),
    )
    assert differences == [
        "outputs.only_replay: only in the replay",
        "outputs.only_stored: only in the stored result",
    ]


def test_lists_are_compared_item_by_item_and_by_length():
    assert compare_records(record(outputs={"v": [1, 2, 3]}), record(outputs={"v": [1, 9, 3]})) == [
        "outputs.v[1]: replay 2 != stored 9"
    ]
    assert compare_records(record(outputs={"v": [1, 2]}), record(outputs={"v": [1]})) == [
        "outputs.v: replay has 2 items, stored result has 1"
    ]


def test_values_of_different_kinds_differ():
    (difference,) = compare_records(record(outputs={"x": [1]}), record(outputs={"x": {"a": 1}}))
    assert difference.startswith("outputs.x: replay [1] != stored")


def test_nan_equals_nan_and_tolerance_is_relative():
    nan = float("nan")
    assert compare_records(record(outputs={"x": nan}), record(outputs={"x": nan})) == []
    assert compare_records(record(outputs={"x": nan}), record(outputs={"x": 1.0})) != []
    assert compare_records(record(outputs={"x": 1000.0}), record(outputs={"x": 1000.5}), 1e-3) == []
    assert compare_records(record(outputs={"x": 1000.0}), record(outputs={"x": 1002.0}), 1e-3) != []
