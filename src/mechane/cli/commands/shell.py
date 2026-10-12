from __future__ import annotations

import code
from pathlib import Path

import click

from mechane.cli._options import MANIFEST_PATH
from mechane.core.manifest import load_laboratory
from mechane.core.shell import shell_namespace


@click.command("shell")
@MANIFEST_PATH
def shell(manifest_path: Path) -> None:
    """Open a Python prompt with the laboratory loaded as `lab`."""
    lab = load_laboratory(manifest_path)
    namespace = shell_namespace(lab)
    banner = (
        f"mechane shell: {lab.name}, {lab.num_experiments} experiments x "
        f"{lab.num_instances} instances\nAvailable: {', '.join(namespace)}"
    )
    code.interact(banner=banner, local=namespace)
