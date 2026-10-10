from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import EXPERIMENT_ID, INSTANCE_ID, MANIFEST_PATH, OVERWRITE
from mechane.cli.runner import run_instance
from mechane.core.laboratory import load_laboratory


@click.command("run")
@MANIFEST_PATH
@EXPERIMENT_ID
@INSTANCE_ID
@OVERWRITE
def simulate(manifest_path: Path, experiment_id: int, instance_id: int, overwrite: bool) -> None:
    """Run a single instance (one SLURM array task)."""
    run_instance(load_laboratory(manifest_path), experiment_id, instance_id, overwrite=overwrite)
