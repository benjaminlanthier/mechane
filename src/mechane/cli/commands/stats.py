from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.cli._output import duration, format_table, size, slurm_time
from mechane.core.manifest import load_laboratory
from mechane.core.stats import compute_stats, suggested_walltime


@click.command("stats")
@MANIFEST_PATH
@selection_options
@click.option("--disk", is_flag=True, help="Also count files and bytes (slow on big trees).")
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Suggest a SLURM --time for array tasks of this many instances.",
)
@click.option(
    "--margin",
    type=click.FloatRange(min=1.0),
    default=1.25,
    show_default=True,
    help="Safety factor for the suggested --time.",
)
def stats(
    manifest_path: Path,
    experiment_ids: tuple[int, ...],
    where: tuple[str, ...],
    disk: bool,
    batch_size: int | None,
    margin: float,
) -> None:
    """How long finished instances took (and how much disk they use), per experiment."""
    lab = load_laboratory(manifest_path)
    results = compute_stats(selected(lab, experiment_ids, where), disk=disk)
    headers = ["experiment", "done", "median", "p95", "max", "total"]
    if disk:
        headers += ["files", "size"]
    rows = []
    for s in results:
        row: list[object] = [
            s.experiment_id,
            f"{s.num_done}/{s.num_expected}",
            duration(s.wall_median),
            duration(s.wall_p95),
            duration(s.wall_max),
            duration(s.wall_total),
        ]
        if disk:
            row += [s.num_files, size(s.size_bytes)]
        rows.append(row)
    click.echo(format_table(headers, rows))
    click.echo(f"\nTotal instance time: {duration(sum(s.wall_total for s in results))}")
    if batch_size is not None:
        seconds = suggested_walltime(results, batch_size, margin)
        if seconds is None:
            click.echo("No timings yet: nothing has finished, so no --time can be suggested.")
        else:
            click.echo(
                f"Suggested --time for {batch_size} instance(s) per task: {slurm_time(seconds)} "
                f"(slowest p95 x {batch_size} x {margin:g}; a heuristic)"
            )
