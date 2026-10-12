from __future__ import annotations

from pathlib import Path

import click

from mechane.core.scaffold import (
    DEFAULT_SECTIONS,
    ScaffoldError,
    render_lab,
    write_files,
)


@click.group("init")
def init() -> None:
    """Create starting files for a new project (currently: `init lab`)."""


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
