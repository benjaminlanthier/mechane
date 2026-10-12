from __future__ import annotations

import json
from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.cli._output import format_plan
from mechane.core.manifest import load_laboratory
from mechane.core.plan import build_plan, plan_to_dict, write_plan


@click.command("plan")
@MANIFEST_PATH
@selection_options
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Instances per array task.",
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
    help="Save the plan here: array tasks run it with `mechane run-plan`.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "array", "json"]),
    default="table",
    show_default=True,
    help="`array` prints just the value for `sbatch --array=`.",
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
    """Split the work into SLURM array tasks and say how to submit them."""
    lab = load_laboratory(manifest_path)
    the_plan = build_plan(
        lab,
        batch_size,
        experiments=selected(lab, experiment_ids, where),
        unfinished_only=unfinished,
    )
    if output is not None:
        write_plan(the_plan, output)
    if fmt == "array":
        click.echo(the_plan.array_spec)
    elif fmt == "json":
        click.echo(json.dumps(plan_to_dict(the_plan), indent=2))
    else:
        click.echo(format_plan(the_plan))
