"""One module per console command. Each defines a single `click.Command`."""
from mechane.cli.commands.aggregate import aggregate_results
from mechane.cli.commands.run import simulate
from mechane.cli.commands.run_batch import simulate_batch
from mechane.cli.commands.setup import setup_lab
from mechane.cli.commands.status import status

__all__ = [
    "aggregate_results",
    "setup_lab",
    "simulate",
    "simulate_batch",
    "status"
]
