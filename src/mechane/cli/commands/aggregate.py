from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH
from mechane.core.laboratory import load_laboratory


@click.command("aggregate")
@MANIFEST_PATH
def aggregate_results(manifest_path: Path) -> None:
    """Collect every experiment's instance results into aggregated_results.json."""
    lab = load_laboratory(manifest_path)
    lab.aggregate_results()
    click.echo(f"[Aggregation] Wrote {lab.all_results_path}")
