"""The CLI's wording: turn events from `mechane.core.runner` into the lines printed on a terminal
(and in SLURM logs). Everything user-visible about running instances is defined here."""

from __future__ import annotations

import click

from mechane.core.runner import (
    InstanceSaved,
    InstanceSkipped,
    InstanceStarted,
    RunEvent,
)


def echo_event(event: RunEvent) -> None:
    if isinstance(event, InstanceSkipped):
        click.echo(f"[Job] {event.results_file} already exists, skipping.")
    elif isinstance(event, InstanceStarted):
        click.echo(
            f"[Job] lab={event.laboratory} experiment_id={event.experiment_id} "
            f"instance_id={event.instance_id}"
        )
    elif isinstance(event, InstanceSaved):
        click.echo(f"[Save] {event.results_file}")
    else:  # InstanceFailed: the only remaining event
        click.echo(f"[BatchJob] instance_id={event.instance_id} FAILED: {event.error!r}", err=True)
