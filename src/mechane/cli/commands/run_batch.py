from __future__ import annotations

import sys
from pathlib import Path

import click

from mechane.cli._options import EXPERIMENT_ID, INSTANCE_IDS, MANIFEST_PATH, OVERWRITE
from mechane.core.laboratory import load_laboratory
from mechane.core.runner import run_instance


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
    failed: list[int] = []
    for instance_id in ids:
        try:
            run_instance(lab, experiment_id, instance_id, overwrite=overwrite)
        except Exception as exc:
            failed.append(instance_id)
            click.echo(f"[BatchJob] instance_id={instance_id} FAILED: {exc!r}", err=True)
            if not continue_on_error:
                raise
    if failed:
        click.echo(f"[BatchJob] {len(failed)}/{len(ids)} failed: {failed}", err=True)
        sys.exit(1)
    click.echo(f"[BatchJob] Completed {len(ids)}/{len(ids)} instance(s).")
