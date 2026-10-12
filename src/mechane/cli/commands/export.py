from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.core.export import build_table
from mechane.core.manifest import load_laboratory


def _cell(value: Any) -> Any:
    """A table cell for CSV: lists and mappings as JSON, None as empty."""
    if value is None:
        return ""
    return json.dumps(value) if isinstance(value, dict | list) else value


@click.command("export")
@MANIFEST_PATH
@selection_options
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "jsonl"]),
    default="csv",
    show_default=True,
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write here instead of to standard output.",
)
@click.option("--inputs", "include_inputs", is_flag=True, help="Include the recorded inputs.")
@click.option(
    "--meta", "include_meta", is_flag=True, help="Include run metadata (host, time, ...)."
)
def export(
    manifest_path: Path,
    experiment_ids: tuple[int, ...],
    where: tuple[str, ...],
    fmt: str,
    output: Path | None,
    include_inputs: bool,
    include_meta: bool,
) -> None:
    """Write finished instances as a table: one row each, with parameters and outputs."""
    lab = load_laboratory(manifest_path)
    table = build_table(
        selected(lab, experiment_ids, where),
        include_inputs=include_inputs,
        include_meta=include_meta,
    )
    if not table.rows:
        click.echo("No finished instances to export.", err=True)
        return
    stream = output.open("w") if output is not None else sys.stdout
    try:
        if fmt == "csv":
            writer = csv.DictWriter(stream, fieldnames=table.columns, lineterminator="\n")
            writer.writeheader()
            for row in table.rows:
                writer.writerow({k: _cell(v) for k, v in row.items()})
        else:
            for row in table.rows:
                stream.write(json.dumps(row) + "\n")
    finally:
        if output is not None:
            stream.close()
