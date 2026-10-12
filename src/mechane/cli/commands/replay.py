from __future__ import annotations

import json
import pdb
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path

import click

from mechane.cli._options import EXPERIMENT_ID, INSTANCE_ID, MANIFEST_PATH
from mechane.core.manifest import load_laboratory
from mechane.core.replay import ReplayError, replay_instance


@click.command("replay")
@MANIFEST_PATH
@EXPERIMENT_ID
@INSTANCE_ID
@click.option(
    "--rtol",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help="Relative tolerance when comparing numbers with the stored result.",
)
@click.option(
    "--scratch-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Run here and keep what is written (default: a temporary directory, removed afterwards).",
)
@click.option("--pdb", "use_pdb", is_flag=True, help="Open the debugger if the instance raises.")
def replay(
    manifest_path: Path,
    experiment_id: int,
    instance_id: int,
    rtol: float,
    scratch_dir: Path | None,
    use_pdb: bool,
) -> None:
    """Re-run one instance with its original seed, and compare with its stored result.

    The replayed inputs and outputs go to standard output; the verdict goes to standard error,
    and the exit status is 1 if the replay differs from the stored result. Nothing is saved in
    the results tree.
    """
    lab = load_laboratory(manifest_path)
    if scratch_dir is not None:
        scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = (
        tempfile.TemporaryDirectory(prefix="mechane-replay-")
        if scratch_dir is None
        else nullcontext(str(scratch_dir))
    )
    with scratch as root:
        try:
            result = replay_instance(lab, experiment_id, instance_id, Path(root), rtol=rtol)
        except ReplayError as exc:
            raise click.UsageError(str(exc)) from exc
        except Exception as exc:
            if use_pdb:
                pdb.post_mortem(exc.__traceback__)
            raise
    shown = {key: result.record[key] for key in ("seed", "inputs", "outputs")}
    click.echo(json.dumps(shown, indent=2))
    if result.reproduced is None:
        click.echo("No stored result to compare with.", err=True)
    elif result.reproduced:
        click.echo("Reproduced: the replay matches the stored result.", err=True)
    else:
        click.echo("NOT reproduced: the replay differs from the stored result:", err=True)
        for difference in result.differences:
            click.echo(f"  {difference}", err=True)
        sys.exit(1)
