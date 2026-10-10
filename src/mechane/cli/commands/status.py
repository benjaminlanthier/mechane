from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH
from mechane.core.laboratory import load_laboratory


@click.command("status")
@MANIFEST_PATH
def status(manifest_path: Path) -> None:
    """Show done / failed / missing instance counts per experiment."""
    lab = load_laboratory(manifest_path)
    for exp in lab.experiments:
        s = exp.status()
        click.echo(
            f"experiment {exp.config.experiment_id:>4}: "
            f"{len(s['done'])} done, {len(s['failed'])} failed, {len(s['missing'])} missing"
        )
