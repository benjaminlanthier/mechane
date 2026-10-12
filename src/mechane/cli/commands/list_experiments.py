from __future__ import annotations

import json
from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.cli._output import format_table
from mechane.core.listing import describe, varying_keys
from mechane.core.manifest import load_laboratory


def _cell(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value)


@click.command("list")
@MANIFEST_PATH
@selection_options
@click.option("--status", "with_status", is_flag=True, help="Add done/failed/missing counts.")
@click.option("--paths", is_flag=True, help="Add each experiment's instances directory.")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show every parameter, not only those that differ between experiments.",
)
@click.option("--json", "as_json", is_flag=True, help="Print JSON instead of a table.")
def list_experiments(
    manifest_path: Path,
    experiment_ids: tuple[int, ...],
    where: tuple[str, ...],
    with_status: bool,
    paths: bool,
    show_all: bool,
    as_json: bool,
) -> None:
    """List experiments with their parameters."""
    lab = load_laboratory(manifest_path)
    rows = describe(selected(lab, experiment_ids, where), with_status=with_status)
    if as_json:
        click.echo(
            json.dumps(
                [
                    {
                        "experiment_id": r.experiment_id,
                        "parameters": r.parameters,
                        "content_hash": r.content_hash,
                        "path": str(r.path),
                        **({"status": r.status} if r.status is not None else {}),
                    }
                    for r in rows
                ],
                indent=2,
            )
        )
        return
    if not rows:
        click.echo("No experiments match.")
        return
    keys = list(rows[0].parameters) if show_all or len(rows) == 1 else varying_keys(rows)
    headers = ["id", *keys]
    if with_status:
        headers += ["done", "failed", "missing"]
    if paths:
        headers.append("path")
    table = []
    for r in rows:
        row: list[object] = [r.experiment_id, *(_cell(r.parameters[k]) for k in keys)]
        if r.status is not None:
            row += [r.status["done"], r.status["failed"], r.status["missing"]]
        if paths:
            row.append(r.path)
        table.append(row)
    click.echo(format_table(headers, table))
