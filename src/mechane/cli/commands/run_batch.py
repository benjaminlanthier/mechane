from __future__ import annotations

import sys
from pathlib import Path

import click

from mechane.cli._options import EXPERIMENT_ID, INSTANCE_IDS, MANIFEST_PATH, OVERWRITE
from mechane.cli._output import echo_event
from mechane.core.manifest import load_laboratory
from mechane.core.runner import run_batch


@click.command("run-batch")
@MANIFEST_PATH
@EXPERIMENT_ID
@INSTANCE_IDS
@OVERWRITE
@click.option("--continue-on-error/--fail-fast", default=True, show_default=True)
def simulate_batch(
    manifest_path: Path,
    experiment_id: int,
    instance_ids: str,
    overwrite: bool,
    continue_on_error: bool,
) -> None:
    """Run several instances of one experiment in a single process."""
    ids = [int(i) for i in instance_ids.split(",") if i.strip()]
    if not ids or any(i < 0 for i in ids):
        raise click.UsageError("--instance-ids must be a non-empty list of non-negative integers.")
    lab = load_laboratory(manifest_path)  # loaded once for the whole batch
    result = run_batch(
        lab,
        experiment_id,
        ids,
        overwrite=overwrite,
        continue_on_error=continue_on_error,
        listener=echo_event,
    )
    if result.failed:
        click.echo(
            f"[BatchJob] {len(result.failed)}/{len(ids)} failed: {list(result.failed)}", err=True
        )
        sys.exit(1)
    click.echo(f"[BatchJob] Completed {len(ids)}/{len(ids)} instance(s).")
