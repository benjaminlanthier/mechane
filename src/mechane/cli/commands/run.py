from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import EXPERIMENT_ID, INSTANCE_ID, MANIFEST_PATH, OVERWRITE
from mechane.cli._output import echo_event
from mechane.core.manifest import load_laboratory
from mechane.core.runner import run_instance


@click.command("run")
@MANIFEST_PATH
@EXPERIMENT_ID
@INSTANCE_ID
@OVERWRITE
def simulate(manifest_path: Path, experiment_id: int, instance_id: int, overwrite: bool) -> None:
    """Run a single instance."""
    lab = load_laboratory(manifest_path)
    run_instance(lab, experiment_id, instance_id, overwrite=overwrite, listener=echo_event)
