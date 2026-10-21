from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import OVERWRITE
from mechane.cli._output import echo_event, finish_batch
from mechane.core.manifest import load_laboratory
from mechane.core.plan import PlanError, read_task
from mechane.core.runner import run_batch


@click.command("run-plan")
@click.option(
    "--plan-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="A plan written by `mechane plan --output`.",
)
@click.option(
    "--task-id",
    type=click.IntRange(min=0),
    required=True,
    help="Which task of the plan to run (0 to the number of tasks minus one).",
)
@OVERWRITE
@click.option("--continue-on-error/--fail-fast", default=True, show_default=True)
def run_plan(plan_file: Path, task_id: int, overwrite: bool, continue_on_error: bool) -> None:
    """Run one task of a plan: a batch of instances of one experiment."""
    try:
        manifest_path, task = read_task(plan_file, task_id)
    except PlanError as exc:
        raise click.UsageError(str(exc)) from exc
    lab = load_laboratory(manifest_path)
    result = run_batch(
        lab,
        task.experiment_id,
        task.instance_ids,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
        listener=echo_event,
    )
    finish_batch(result)
