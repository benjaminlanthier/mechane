"""How long instances took, and (optionally) how much disk they use.

Timings come from `meta.wall_time_s` in each results file, so they are what the instances really
spent, not what a job requested. The numbers here are in seconds and bytes: wording is the CLI's.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment


@dataclass(frozen=True)
class ExperimentStats:
    experiment_id: int
    num_expected: int
    num_done: int
    wall_median: float | None  # None while no instance has finished
    wall_p95: float | None
    wall_max: float | None
    wall_total: float  # sum over instances, in seconds
    num_files: int | None = None  # only with `disk=True`
    size_bytes: int | None = None


def _disk_usage(exp: Experiment) -> tuple[int, int]:
    files = (
        [p for p in exp.instances_dir.rglob("*") if p.is_file()]
        if exp.instances_dir.is_dir()
        else []
    )
    return len(files), sum(p.stat().st_size for p in files)


def compute_stats(experiments: Iterable[Experiment], disk: bool = False) -> list[ExperimentStats]:
    """Timing statistics per experiment. `disk=True` also walks every instance directory,
    which is slow on large trees and on parallel filesystems."""
    out: list[ExperimentStats] = []
    for exp in experiments:
        times = np.array(
            [
                r["meta"]["wall_time_s"]
                for _, r in exp.iter_results()
                if "wall_time_s" in (r.get("meta") or {})
            ],
            dtype=float,
        )
        num_done = len(exp.status()["done"])
        files, size = _disk_usage(exp) if disk else (None, None)
        out.append(
            ExperimentStats(
                experiment_id=exp.config.experiment_id,
                num_expected=exp.config.num_instances,
                num_done=num_done,
                wall_median=float(np.median(times)) if times.size else None,
                wall_p95=float(np.percentile(times, 95)) if times.size else None,
                wall_max=float(times.max()) if times.size else None,
                wall_total=float(times.sum()),
                num_files=files,
                size_bytes=size,
            )
        )
    return out


def suggested_walltime(
    stats: Sequence[ExperimentStats], batch_size: int, margin: float = 1.25
) -> float | None:
    """A time limit, in seconds, for an array task running `batch_size` instances back to back:
    the slowest experiment's p95 per instance, times `batch_size`, times a safety `margin`.
    None when nothing has finished yet. A heuristic: it cannot know about a slower, unrun case."""
    p95s = [s.wall_p95 for s in stats if s.wall_p95 is not None]
    return max(p95s) * batch_size * margin if p95s else None
