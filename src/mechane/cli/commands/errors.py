from __future__ import annotations

from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.core.errors import collect_failures
from mechane.core.manifest import load_laboratory


@click.command("errors")
@MANIFEST_PATH
@selection_options
@click.option(
    "--examples",
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Instances to list for each distinct error.",
)
@click.option(
    "--traceback",
    "show_traceback",
    is_flag=True,
    help="Print the full traceback of the first instance of each error.",
)
def errors(
    manifest_path: Path,
    experiment_ids: tuple[int, ...],
    where: tuple[str, ...],
    examples: int,
    show_traceback: bool,
) -> None:
    """Group failed instances by their error, the most frequent first."""
    lab = load_laboratory(manifest_path)
    groups = collect_failures(selected(lab, experiment_ids, where))
    if not groups:
        click.echo("No failed instances.")
        return
    click.echo(
        f"{sum(g.count for g in groups)} failed instance(s), {len(groups)} distinct error(s)"
    )
    for group in groups:
        click.echo(f"\n{group.count:>5}x  {group.signature}")
        for failed in group.instances[:examples]:
            click.echo(
                f"        experiment {failed.experiment_id} instance {failed.instance_id}: "
                f"{failed.error_file}"
            )
        if group.count > examples:
            click.echo(f"        ... and {group.count - examples} more")
        if show_traceback:
            traceback_text = group.instances[0].read_traceback().rstrip()
            click.echo("\n".join(f"        | {line}" for line in traceback_text.splitlines()))
