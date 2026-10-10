"""Click options shared by several commands. They keep the option names of the original
`experiments` scripts so existing sbatch scripts work unchanged."""

from __future__ import annotations

from pathlib import Path

import click

MANIFEST_PATH = click.option(
    "--manifest-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the manifest.json written by `setup`.",
)
PARAMS_PATH = click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
EXPERIMENT_ID = click.option("--experiment-id", type=click.IntRange(min=0), required=True)
INSTANCE_ID = click.option("--instance-id", type=click.IntRange(min=0), required=True)
INSTANCE_IDS = click.option(
    "--instance-ids", type=str, required=True, help="Comma-separated, e.g. 10,11,12"
)
OVERWRITE = click.option("--overwrite/--no-overwrite", default=False, show_default=True)
