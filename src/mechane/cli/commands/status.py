from __future__ import annotations

import sys
from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH
from mechane.core.manifest import load_laboratory


@click.command("status")
@MANIFEST_PATH
@click.option(
    "--check",
    is_flag=True,
    help="Exit with status 1 unless every instance of every experiment is done.",
)
def status(manifest_path: Path, check: bool) -> None:
    """Show done / failed / missing instance counts per experiment."""
    lab = load_laboratory(manifest_path)
    not_done = 0
    for exp in lab.experiments:
        s = exp.status()
        not_done += len(s["failed"]) + len(s["missing"])
        click.echo(
            f"experiment {exp.config.experiment_id:>4}: "
            f"{len(s['done'])} done, {len(s['failed'])} failed, {len(s['missing'])} missing"
        )
    if check and not_done:
        click.echo(f"{not_done} instance(s) are not done.", err=True)
        sys.exit(1)
