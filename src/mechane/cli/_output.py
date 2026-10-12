"""The CLI's wording: turn events from `mechane.core.runner` into the lines printed on a terminal
(and in job logs), and the formatting shared by commands that print tables."""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence

import click

from mechane.core.plan import Plan
from mechane.core.runner import (
    BatchResult,
    InstanceSaved,
    InstanceSkipped,
    InstanceStarted,
    RunEvent,
)
from mechane.utils.ranges import compact_ranges


def echo_event(event: RunEvent) -> None:
    if isinstance(event, InstanceSkipped):
        click.echo(f"[Job] {event.results_file} already exists, skipping.")
    elif isinstance(event, InstanceStarted):
        click.echo(
            f"[Job] lab={event.laboratory} experiment_id={event.experiment_id} "
            f"instance_id={event.instance_id}"
        )
    elif isinstance(event, InstanceSaved):
        click.echo(f"[Save] {event.results_file}")
    else:  # InstanceFailed: the only remaining event
        click.echo(f"[BatchJob] instance_id={event.instance_id} FAILED: {event.error!r}", err=True)


def finish_batch(result: BatchResult) -> None:
    """Print the summary of a batch, and exit with status 1 if any instance failed."""
    total = len(result.instance_ids)
    if result.failed:
        click.echo(
            f"[BatchJob] {len(result.failed)}/{total} failed: {list(result.failed)}", err=True
        )
        sys.exit(1)
    click.echo(f"[BatchJob] Completed {total}/{total} instance(s).")


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Left-aligned columns separated by two spaces, no trailing whitespace."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    return "\n".join(
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in cells
    )


def duration(seconds: float | None) -> str:
    """`0.012` -> `12ms`, `5.5` -> `5.5s`, `125` -> `2m05s`, `7300` -> `2h01m`; None -> `-`."""
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


_SIZE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")


def size(num_bytes: int | None) -> str:
    """`2048` -> `2.0 KiB`; None -> `-`."""
    if num_bytes is None:
        return "-"
    exponent = min((num_bytes.bit_length() - 1) // 10, len(_SIZE_UNITS) - 1) if num_bytes else 0
    if exponent <= 0:
        return f"{num_bytes} B"
    return f"{num_bytes / 1024**exponent:.1f} {_SIZE_UNITS[exponent]}"


def minutes(seconds: float) -> int:
    """Whole minutes, rounded up, at least one: a time limit that any scheduler can take."""
    return max(1, math.ceil(seconds / 60))


def format_plan(plan: Plan) -> str:
    if not plan.tasks:
        return "Nothing to run: the plan has no tasks."
    head = (
        f"{plan.num_instances} instance(s) in {len(plan.tasks)} task(s) of up to "
        f"{plan.batch_size}; run task N (0 to {len(plan.tasks) - 1}) with `mechane run-plan`"
    )
    rows = [
        (i, t.experiment_id, len(t.instance_ids), compact_ranges(t.instance_ids))
        for i, t in enumerate(plan.tasks)
    ]
    return head + "\n\n" + format_table(("task", "experiment", "count", "instances"), rows)
