from __future__ import annotations

from pathlib import Path

import click

from mechane.core.scaffold import (
    DEFAULT_SECTIONS,
    ScaffoldError,
    render_lab,
    render_slurm,
    write_files,
)


@click.group("init")
def init() -> None:
    """Create starting files: a laboratory (`init lab`) or a SLURM script (`init slurm`)."""


@init.command("lab")
@click.argument("name")
@click.option(
    "--section",
    "sections",
    multiple=True,
    help=f"A section of parameters (repeatable; default: {', '.join(DEFAULT_SECTIONS)}).",
)
@click.option(
    "--directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path(),
    show_default=True,
    help="Where to write NAME.py and params.py.",
)
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def init_lab(name: str, sections: tuple[str, ...], directory: Path, force: bool) -> None:
    """Write a runnable laboratory module called NAME and a matching params.py."""
    try:
        written = write_files(directory, render_lab(name, sections or DEFAULT_SECTIONS), force)
    except ScaffoldError as exc:
        raise click.UsageError(str(exc)) from exc
    for path in written:
        click.echo(f"[Init] wrote {path}")
    click.echo(
        f"\nNext: cd {directory} && PYTHONPATH=. mechane setup --params-file params.py\n"
        "then edit `run` in the generated module."
    )


@init.command("slurm")
@click.option("--job-name", default="mechane", show_default=True)
@click.option("--time", default="01:00:00", show_default=True, help="Time limit per array task.")
@click.option("--cpus-per-task", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--mem", default="4G", show_default=True)
@click.option("--account", default=None)
@click.option("--partition", default=None)
@click.option("--container", default=None, help="An Apptainer image to run `mechane` in.")
@click.option("--bind", "binds", multiple=True, help="Bind mount for the container (repeatable).")
@click.option("--module", "modules", multiple=True, help="`module load` this (repeatable).")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the script here instead of to standard output.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing --output file.")
def init_slurm(
    job_name: str,
    time: str,
    cpus_per_task: int,
    mem: str,
    account: str | None,
    partition: str | None,
    container: str | None,
    binds: tuple[str, ...],
    modules: tuple[str, ...],
    output: Path | None,
    force: bool,
) -> None:
    """Print an sbatch script whose array tasks run the tasks of a `mechane plan`."""
    script = render_slurm(
        job_name=job_name,
        time=time,
        cpus_per_task=cpus_per_task,
        mem=mem,
        account=account,
        partition=partition,
        container=container,
        binds=binds,
        modules=modules,
    )
    if output is None:
        click.echo(script, nl=False)
        return
    try:
        write_files(output.parent, {output.name: script}, force)
    except ScaffoldError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(f"[Init] wrote {output}")
