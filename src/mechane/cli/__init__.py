"""Console entry point: the `mechane` group and its sub-commands.

Each command lives in its own module under `mechane.cli.commands`; this file only assembles
them. Command names come from each command's own `click.command("<name>")`.
"""

from __future__ import annotations

import click

from mechane.cli.commands import (
    aggregate_results,
    clean,
    doctor,
    errors,
    export,
    init,
    list_experiments,
    plan,
    replay,
    run_plan,
    setup_lab,
    shell,
    simulate,
    simulate_batch,
    stats,
    status,
)


@click.group()
def main() -> None:
    """mechane: declarative sweeps and job running."""


for _cmd in (
    aggregate_results,
    clean,
    doctor,
    errors,
    export,
    init,
    list_experiments,
    plan,
    replay,
    run_plan,
    setup_lab,
    shell,
    simulate,
    simulate_batch,
    stats,
    status,
):
    main.add_command(_cmd)

__all__ = [
    "aggregate_results",
    "clean",
    "doctor",
    "errors",
    "export",
    "init",
    "list_experiments",
    "main",
    "plan",
    "replay",
    "run_plan",
    "setup_lab",
    "shell",
    "simulate",
    "simulate_batch",
    "stats",
    "status",
]
