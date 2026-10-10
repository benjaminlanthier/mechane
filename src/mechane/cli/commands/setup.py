from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import click

from mechane.cli._options import PARAMS_PATH
from mechane.core.laboratory import Laboratory
from mechane.utils.registry import resolve_laboratory


def load_params_file(params_file: Path) -> tuple[Path, type[Laboratory], dict]:
    """Execute a params file and return (ROOT, laboratory class, PARAMS).

    The file defines ROOT, PARAMS and LABORATORY ("module:Class", an entry-point name or the
    class itself). Legacy files defining EXPERIMENT_TYPE instead are accepted.
    """
    spec = importlib.util.spec_from_file_location("external_params", params_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {params_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["external_params"] = module
    spec.loader.exec_module(module)

    ref = getattr(module, "LABORATORY", None) or getattr(module, "EXPERIMENT_TYPE", None)
    if ref is None:
        raise click.UsageError(f"{params_file} must define LABORATORY (or legacy EXPERIMENT_TYPE).")
    lab_cls = ref if isinstance(ref, type) else resolve_laboratory(str(ref))
    return Path(module.ROOT), lab_cls, module.PARAMS


@click.command("setup")
@PARAMS_PATH
def setup_lab(params_file: Path) -> None:
    """Expand the sweep in PARAMS_FILE, write the manifest and create the directories."""
    root, lab_cls, configs = load_params_file(params_file)
    lab = lab_cls.from_configs(root=root, configs=configs)
    lab.setup()
    click.echo(
        f"[Setup] {lab.num_experiments} experiments x {lab.num_instances} instances. "
        f"Manifest: {lab.manifest_path}"
    )
