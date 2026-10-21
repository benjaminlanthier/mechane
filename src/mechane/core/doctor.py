"""Consistency checks on what a laboratory has left on disk.

`diagnose` reads the results of every instance once and reports, per experiment:

  * results whose recorded instance id or seed does not match the configuration (error);
  * results that cannot be read, or lack the fields every record has (error);
  * results next to a stale `job.err` (warning);
  * instance directories past `num_instances`, which `status` and `aggregate` never look at,
    typically left by a batch that overshot the last instance (warning);
  * results produced by more than one code version (warning).

and, for the laboratory as a whole, a relative root and directories under the root that this
manifest does not use (warnings). It never modifies anything.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mechane.utils.ranges import compact_ranges

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment
    from mechane.core.laboratory import Laboratory

Severity = Literal["error", "warning"]

#: What `Instance.execute` writes into every record.
REQUIRED_FIELDS = frozenset({"instance_id", "experiment_id", "seed", "inputs", "outputs", "meta"})


@dataclass(frozen=True)
class Finding:
    severity: Severity
    code: str
    message: str
    experiment_id: int | None = None  # None: about the laboratory as a whole
    instance_ids: tuple[int, ...] = ()


def diagnose(lab: Laboratory, experiments: Sequence[Experiment] | None = None) -> list[Finding]:
    """Findings for `experiments`, or for the whole laboratory when `experiments` is None
    (only then are the lab-wide checks run)."""
    findings: list[Finding] = []
    if experiments is None:
        findings += _check_root(lab)
        chosen: Sequence[Experiment] = lab.experiments
    else:
        chosen = experiments
    for exp in chosen:
        findings += _check_experiment(exp)
    if experiments is None:
        findings += _check_orphans(lab)
    return findings


def _check_root(lab: Laboratory) -> list[Finding]:
    if lab.root.is_absolute():
        return []
    message = (
        f"the manifest stores a relative root ({lab.root}): commands run from another directory "
        "will not find the results"
    )
    return [Finding("warning", "relative-root", message)]


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not record.keys() >= REQUIRED_FIELDS:
        return None
    return record


def code_version(record: dict[str, Any]) -> str:
    """Which code produced a record: its git commit when recorded, else its package versions."""
    meta = record.get("meta") or {}
    git = meta.get("git")
    if isinstance(git, dict) and git.get("commit"):
        return f"{str(git['commit'])[:7]} ({'dirty' if git.get('dirty') else 'clean'})"
    packages = meta.get("packages")
    if isinstance(packages, dict) and packages:
        return "packages " + ", ".join(f"{k}={v}" for k, v in sorted(packages.items()))
    return "unknown"


def _instances(ids: list[int]) -> str:
    """`instance 3` or `instances 1-2,5`."""
    return f"{'instance' if len(ids) == 1 else 'instances'} {compact_ranges(ids)}"


def _check_experiment(exp: Experiment) -> list[Finding]:
    eid, n = exp.config.experiment_id, exp.config.num_instances
    unreadable: list[int] = []
    mismatched: list[int] = []
    stale: list[int] = []
    versions: dict[str, list[int]] = {}

    for i in exp.present_instance_ids():
        results = exp.results_path(i)
        if not results.exists():
            continue
        if exp.error_path(i).exists():
            stale.append(i)
        record = _read_record(results)
        if record is None:
            unreadable.append(i)
            continue
        if record["instance_id"] != i or record["seed"] != exp.instance_seed(i):
            mismatched.append(i)
        versions.setdefault(code_version(record), []).append(i)

    stray: list[int] = []
    if exp.instances_dir.is_dir():
        stray = sorted(
            int(p.name)
            for p in exp.instances_dir.iterdir()
            if p.name.isdigit() and int(p.name) >= n
        )

    findings: list[Finding] = []

    def add(severity: Severity, code: str, ids: list[int], message: str) -> None:
        if ids:
            findings.append(Finding(severity, code, message, eid, tuple(ids)))

    add(
        "error",
        "record-mismatch",
        mismatched,
        f"{_instances(mismatched)}: the recorded seed or instance id does not match "
        "the configuration",
    )
    add(
        "error",
        "unreadable-result",
        unreadable,
        f"{_instances(unreadable)}: results.json is not valid JSON or lacks required fields",
    )
    add(
        "warning",
        "stale-error",
        stale,
        f"{_instances(stale)}: results.json exists next to a stale job.err",
    )
    add(
        "warning",
        "stray-instance",
        stray,
        f"instance directories {compact_ranges(stray)} are outside 0..{n - 1}; "
        "status and aggregate ignore them",
    )
    if len(versions) > 1:
        detail = "; ".join(f"{v}: {compact_ranges(ids)}" for v, ids in sorted(versions.items()))
        every = sorted(i for ids in versions.values() for i in ids)
        add(
            "warning",
            "mixed-code",
            every,
            f"results come from {len(versions)} code versions: {detail}",
        )
    return findings


def _check_orphans(lab: Laboratory) -> list[Finding]:
    if not lab.root.is_dir():
        return []
    # The first directory below the root on the way to each experiment's instances.
    used = {exp.instances_dir.relative_to(lab.root).parts[0] for exp in lab.experiments}
    orphans = sorted(
        p.name
        for p in lab.root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in used
    )
    if not orphans:
        return []
    shown = ", ".join(orphans[:5]) + (f", and {len(orphans) - 5} more" if len(orphans) > 5 else "")
    message = (
        f"{len(orphans)} director{'y' if len(orphans) == 1 else 'ies'} under {lab.root} "
        f"{'is' if len(orphans) == 1 else 'are'} not used by this manifest "
        f"(results of an earlier sweep?): {shown}"
    )
    return [Finding("warning", "orphan-directory", message)]
