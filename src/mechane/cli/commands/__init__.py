"""One module per console command. Each defines a single `click.Command` (or group)."""

from mechane.cli.commands.aggregate import aggregate_results
from mechane.cli.commands.clean import clean
from mechane.cli.commands.doctor import doctor
from mechane.cli.commands.errors import errors
from mechane.cli.commands.export import export
from mechane.cli.commands.init import init
from mechane.cli.commands.list_experiments import list_experiments
from mechane.cli.commands.plan import plan
from mechane.cli.commands.replay import replay
from mechane.cli.commands.run import simulate
from mechane.cli.commands.run_batch import simulate_batch
from mechane.cli.commands.run_plan import run_plan
from mechane.cli.commands.setup import setup_lab
from mechane.cli.commands.shell import shell
from mechane.cli.commands.stats import stats
from mechane.cli.commands.status import status

__all__ = [
    "aggregate_results",
    "clean",
    "doctor",
    "errors",
    "export",
    "init",
    "list_experiments",
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
