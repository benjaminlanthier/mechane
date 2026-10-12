from __future__ import annotations

import sys
from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH, selected, selection_options
from mechane.core.doctor import diagnose
from mechane.core.manifest import load_laboratory


@click.command("doctor")
@MANIFEST_PATH
@selection_options
@click.option("--strict", is_flag=True, help="Exit with status 1 on warnings too.")
def doctor(
    manifest_path: Path, experiment_ids: tuple[int, ...], where: tuple[str, ...], strict: bool
) -> None:
    """Check the results on disk for inconsistencies. Changes nothing.

    Exits with status 1 if it finds errors (and, with --strict, warnings).
    """
    lab = load_laboratory(manifest_path)
    narrowed = bool(experiment_ids or where)
    findings = diagnose(lab, selected(lab, experiment_ids, where) if narrowed else None)
    for finding in findings:
        scope = (
            "laboratory" if finding.experiment_id is None else f"experiment {finding.experiment_id}"
        )
        click.echo(f"{finding.severity.upper():<7} {scope}: {finding.message}")
    n_errors = sum(f.severity == "error" for f in findings)
    n_warnings = len(findings) - n_errors
    if not findings:
        click.echo("No problems found.")
    else:
        click.echo(f"\n{n_errors} error(s), {n_warnings} warning(s)")
    if n_errors or (strict and n_warnings):
        sys.exit(1)
