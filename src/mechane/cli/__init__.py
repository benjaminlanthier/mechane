"""Console entry point: the `mechane` group and its sub-commands.

Each command lives in its own module under `mechane.cli.commands`; this file only assembles
them. Command names come from each command's own `click.command("<name>")`.
"""

from __future__ import annotations

import click

from mechane.cli.commands import (
    aggregate_results,
    setup_lab,
    simulate,
    simulate_batch,
    status,
)


@click.group()
def main() -> None:
    """mechane: declarative sweeps and job running."""


for _cmd in (setup_lab, simulate, simulate_batch, aggregate_results, status):
    main.add_command(_cmd)

__all__ = [
    "aggregate_results",
    "load_params_file",
    "main",
    "setup_lab",
    "simulate",
    "simulate_batch",
    "status",
]
