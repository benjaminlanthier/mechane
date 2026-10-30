from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import PARAMS_PATH
from mechane.core.params import ParamsFileError, load_labs


@click.command("setup")
@PARAMS_PATH
def setup_lab(params_file: Path) -> None:
    """Expand the sweep in PARAMS_FILE, write the manifest and create the directories.

    A file that defines LABS sets up each of its labs, each with its own manifest."""
    try:
        params = load_labs(params_file)
    except ParamsFileError as exc:
        raise click.UsageError(str(exc)) from exc
    # Expand every sweep before writing anything, so a mistake in one lab sets up none of them.
    labs = [p.build_laboratory() for p in params]
    for lab in labs:
        lab.setup()
        click.echo(
            f"[Setup] {lab.num_experiments} experiments x {lab.num_instances} instances. "
            f"Manifest: {lab.manifest_path}"
        )
