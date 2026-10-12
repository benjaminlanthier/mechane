from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import PARAMS_PATH
from mechane.core.params import ParamsFileError, load_params_file


@click.command("setup")
@PARAMS_PATH
def setup_lab(params_file: Path) -> None:
    """Expand the sweep in PARAMS_FILE, write the manifest and create the directories."""
    try:
        params = load_params_file(params_file)
    except ParamsFileError as exc:
        raise click.UsageError(str(exc)) from exc
    lab = params.build_laboratory()
    lab.setup()
    click.echo(
        f"[Setup] {lab.num_experiments} experiments x {lab.num_instances} instances. "
        f"Manifest: {lab.manifest_path}"
    )
