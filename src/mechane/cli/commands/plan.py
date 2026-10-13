from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.cli._output import format_plan
from mechane.core.manifest import load_laboratory
from mechane.core.plan import build_plan, plan_to_json, write_plan


@click.command("plan")
@MANIFEST_PATH
@selection_options
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Instances per task.",
)
@click.option(
    "--unfinished",
    is_flag=True,
    help="Plan only instances that are missing or failed.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Save the plan here: `mechane run-plan` runs one of its tasks.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="`json` is the plan file format, for scripts and workflow tools.",
)
def plan(
    manifest_path: Path,
    experiment_ids: tuple[int, ...],
    where: tuple[str, ...],
    batch_size: int,
    unfinished: bool,
    output: Path | None,
    fmt: str,
) -> None:
    """Split the work into tasks of a few instances each (a unit of scheduled work)."""
    lab = load_laboratory(manifest_path)
    the_plan = build_plan(
        lab,
        batch_size,
        experiments=selected(lab, experiment_ids, where),
        unfinished_only=unfinished,
    )
    if output is not None:
        write_plan(the_plan, output)
    if fmt == "json":
        click.echo(plan_to_json(the_plan), nl=False)
    else:
        click.echo(format_plan(the_plan))
