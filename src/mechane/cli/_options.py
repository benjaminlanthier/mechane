"""Click options shared by several commands. They keep the option names of the original
`experiments` scripts so existing sbatch scripts work unchanged."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from mechane.core.selection import SelectionError, select_experiments
from mechane.utils.ranges import parse_ranges

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment
    from mechane.core.laboratory import Laboratory

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

# ---- choosing experiments (the commands that act on a subset of the sweep)
EXPERIMENT_IDS = click.option(
    "--experiment-id",
    "experiment_ids",
    type=click.IntRange(min=0),
    multiple=True,
    help="Only this experiment (repeatable).",
)
WHERE = click.option(
    "--where",
    "where",
    multiple=True,
    metavar="KEY=VALUE",
    help="Only experiments whose parameter KEY (e.g. model.L) equals VALUE (repeatable).",
)
INSTANCE_RANGES = click.option(
    "--instance-ids",
    type=str,
    default=None,
    help="Only these instances, e.g. 0-9,15 (default: all).",
)


def selection_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """`--experiment-id` and `--where`: the command receives `experiment_ids` and `where`."""
    return EXPERIMENT_IDS(WHERE(command))


def selected(
    lab: Laboratory, experiment_ids: Sequence[int], where: Sequence[str]
) -> list[Experiment]:
    """The experiments chosen by `selection_options`, with bad filters reported as usage errors."""
    try:
        return select_experiments(lab, experiment_ids, where)
    except SelectionError as exc:
        raise click.UsageError(str(exc)) from exc


def parse_instance_ids(text: str | None) -> list[int] | None:
    """The ids of `INSTANCE_RANGES` (None when the option was not given)."""
    if text is None:
        return None
    try:
        return parse_ranges(text)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--instance-ids") from exc
