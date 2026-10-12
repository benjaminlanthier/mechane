from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import (
    INSTANCE_RANGES,
    MANIFEST_PATH,
    parse_instance_ids,
    selected,
    selection_options,
)
from mechane.core.clean import CleanKind, find_targets, remove_targets
from mechane.core.manifest import load_laboratory

SHOWN = 10


@click.command("clean")
@MANIFEST_PATH
@selection_options
@click.option(
    "--what",
    type=click.Choice(["errors", "results", "stray"]),
    required=True,
    help=(
        "errors: job.err files (failed instances become missing). "
        "results: results.json, so finished instances run again. "
        "stray: instance directories past num_instances."
    ),
)
@INSTANCE_RANGES
@click.option(
    "--all-experiments",
    is_flag=True,
    help="Allow --what results without narrowing to some experiments.",
)
@click.option("--yes", is_flag=True, help="Really delete. Without it, only list what would go.")
def clean(
    manifest_path: Path,
    experiment_ids: tuple[int, ...],
    where: tuple[str, ...],
    what: CleanKind,
    instance_ids: str | None,
    all_experiments: bool,
    yes: bool,
) -> None:
    """Remove job.err files, results, or stray directories. Lists them unless --yes is given."""
    ids = parse_instance_ids(instance_ids)
    if what == "stray" and ids is not None:
        raise click.UsageError("--instance-ids does not apply to --what stray.")
    if what == "results" and not (experiment_ids or where or all_experiments):
        raise click.UsageError(
            "refusing to remove the results of every experiment: narrow with --experiment-id or "
            "--where, or pass --all-experiments."
        )
    lab = load_laboratory(manifest_path)
    targets = find_targets(selected(lab, experiment_ids, where), what, ids)
    if not targets:
        click.echo("Nothing to clean.")
        return
    for target in targets[:SHOWN]:
        click.echo(
            f"  experiment {target.experiment_id} instance {target.instance_id}: {target.path}"
        )
    if len(targets) > SHOWN:
        click.echo(f"  ... and {len(targets) - SHOWN} more")
    if yes:
        click.echo(f"Removed {remove_targets(targets)} item(s).")
    else:
        click.echo(f"Would remove {len(targets)} item(s). Re-run with --yes to delete them.")
