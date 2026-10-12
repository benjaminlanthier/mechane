"""`mechane.core.stats`: timings and disk usage of finished instances."""

import json
import os

import pytest

from mechane.core.stats import ExperimentStats, compute_stats, suggested_walltime


def set_wall_times(experiment, seconds):
    for i, wall in enumerate(seconds):
        path = experiment.results_path(i)
        record = json.loads(path.read_text())
        record["meta"]["wall_time_s"] = wall
        path.write_text(json.dumps(record))


def test_an_experiment_with_no_results_has_no_timings(lab):
    (first, _) = compute_stats(lab.experiments)
    assert first == ExperimentStats(0, 6, 0, None, None, None, 0.0)


def test_timings_come_from_the_recorded_wall_times(lab, run_instances):
    run_instances(lab, 0, [0, 1, 2, 3])
    set_wall_times(lab.experiment(0), [1.0, 2.0, 3.0, 10.0])

    first, second = compute_stats(lab.experiments)

    assert (first.num_done, first.num_expected) == (4, 6)
    assert first.wall_median == pytest.approx(2.5)
    assert first.wall_p95 == pytest.approx(8.95)  # linear interpolation: 3 + 0.85 * (10 - 3)
    assert first.wall_max == 10.0
    assert first.wall_total == 16.0
    assert second.num_done == 0 and second.wall_median is None


def test_records_without_a_wall_time_are_counted_as_done_but_not_timed(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    set_wall_times(lab.experiment(0), [4.0])
    path = lab.experiment(0).results_path(1)
    record = json.loads(path.read_text())
    del record["meta"]["wall_time_s"]
    path.write_text(json.dumps(record))

    (first,) = compute_stats([lab.experiment(0)])

    assert first.num_done == 2
    assert first.wall_median == 4.0 and first.wall_total == 4.0


def test_disk_usage_is_only_measured_on_request(lab, run_instances):
    run_instances(lab, 0, [0, 1])
    exp = lab.experiment(0)
    (exp.instance_dir(0) / "extra.bin").write_bytes(b"x" * 100)

    (without,) = compute_stats([exp])
    (with_disk,) = compute_stats([exp], disk=True)

    assert (without.num_files, without.size_bytes) == (None, None)
    files = [os.path.join(d, f) for d, _, names in os.walk(exp.instances_dir) for f in names]
    assert with_disk.num_files == len(files) == 3  # two results.json and extra.bin
    assert with_disk.size_bytes == sum(os.path.getsize(f) for f in files)
    assert with_disk.size_bytes >= 100


def test_disk_usage_of_an_experiment_with_no_directory_is_zero(tmp_path):
    from toy import ToyLab

    lab = ToyLab.from_configs(
        tmp_path / "never-set-up",
        {"model": {"L": 2}, "solver": {"sigma": 1.0}},
    )
    (only,) = compute_stats(lab.experiments, disk=True)
    assert (only.num_files, only.size_bytes) == (0, 0)


def stats_with_p95(*p95s):
    return [ExperimentStats(i, 1, 1, None, p, None, 0.0) for i, p in enumerate(p95s)]


def test_the_suggested_walltime_uses_the_slowest_experiment(lab):
    assert suggested_walltime(stats_with_p95(10.0, 30.0, None), batch_size=4) == 30 * 4 * 1.25
    assert suggested_walltime(stats_with_p95(10.0), batch_size=2, margin=2.0) == 40.0


def test_there_is_no_suggestion_before_anything_has_finished():
    assert suggested_walltime(stats_with_p95(None, None), batch_size=4) is None
    assert suggested_walltime([], batch_size=4) is None
