from __future__ import annotations

import traceback
from pathlib import Path

import click

from mechane.laboratory import Laboratory


def run_instance(
    lab: Laboratory, experiment_id: int, instance_id: int, overwrite: bool = False
) -> Path | None:
    """Run one instance and persist it. Returns the results path, or None if skipped."""
    instance = lab.build_instance(experiment_id, instance_id)
    if not overwrite and instance.results_file.exists():
        click.echo(f"[Job] {instance.results_file} already exists, skipping.")
        return None
    click.echo(f"[Job] lab={lab.name} experiment_id={experiment_id} instance_id={instance_id}")
    try:
        record = instance.execute(provenance=lab.provenance())
        instance.save(record)
        instance.error_file.unlink(missing_ok=True)  # clear a stale traceback from an earlier try
    except Exception:
        instance.dir.mkdir(parents=True, exist_ok=True)
        instance.error_file.write_text(traceback.format_exc())
        raise
    click.echo(f"[Save] {instance.results_file}")
    return instance.results_file
