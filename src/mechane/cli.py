"""Console commands. They keep the option names of the original `experiments` scripts
(`--params-file`, `--manifest-path`, `--experiment-id`, `--instance-id(s)`) so existing
sbatch scripts work unchanged."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import click

from mechane.laboratory import Laboratory, load_laboratory
from mechane.registry import resolve_laboratory
from mechane.runner import run_instance

MANIFEST_PATH = click.option(
    "--manifest-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the manifest.json written by `setup`.",
)
PARAMS_PATH = click.option(
    "--params-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
EXPERIMENT_ID = click.option("--experiment-id", type=click.IntRange(min=0), required=True)
INSTANCE_ID = click.option("--instance-id", type=click.IntRange(min=0), required=True)
INSTANCE_IDS = click.option(
    "--instance-ids", type=str, required=True, help="Comma-separated, e.g. 10,11,12"
)
OVERWRITE = click.option("--overwrite/--no-overwrite", default=False, show_default=True)


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


@click.command("run")
@MANIFEST_PATH
@EXPERIMENT_ID
@INSTANCE_ID
@OVERWRITE
def simulate(manifest_path: Path, experiment_id: int, instance_id: int, overwrite: bool) -> None:
    """Run a single instance (one SLURM array task)."""
    run_instance(load_laboratory(manifest_path), experiment_id, instance_id, overwrite=overwrite)


@click.command("run-batch")
@MANIFEST_PATH
@EXPERIMENT_ID
@INSTANCE_IDS
@OVERWRITE
@click.option("--continue-on-error/--fail-fast", default=True, show_default=True)
def simulate_batch(
    manifest_path: Path,
    experiment_id: int,
    instance_ids: str,
    overwrite: bool,
    continue_on_error: bool,
) -> None:
    """Run several instances of one experiment in a single process."""
    ids = [int(i) for i in instance_ids.split(",") if i.strip()]
    if not ids or any(i < 0 for i in ids):
        raise click.UsageError("--instance-ids must be a non-empty list of non-negative integers.")
    lab = load_laboratory(manifest_path)  # loaded once for the whole batch
    failed: list[int] = []
    for instance_id in ids:
        try:
            run_instance(lab, experiment_id, instance_id, overwrite=overwrite)
        except Exception as exc:
            failed.append(instance_id)
            click.echo(f"[BatchJob] instance_id={instance_id} FAILED: {exc!r}", err=True)
            if not continue_on_error:
                raise
    if failed:
        click.echo(f"[BatchJob] {len(failed)}/{len(ids)} failed: {failed}", err=True)
        sys.exit(1)
    click.echo(f"[BatchJob] Completed {len(ids)}/{len(ids)} instance(s).")


@click.command("aggregate")
@MANIFEST_PATH
def aggregate_results(manifest_path: Path) -> None:
    """Collect every experiment's instance results into aggregated_results.json."""
    lab = load_laboratory(manifest_path)
    lab.aggregate_results()
    click.echo(f"[Aggregation] Wrote {lab.all_results_path}")


@click.command("status")
@MANIFEST_PATH
def status(manifest_path: Path) -> None:
    """Show done / failed / missing instance counts per experiment."""
    lab = load_laboratory(manifest_path)
    for exp in lab.experiments:
        s = exp.status()
        click.echo(
            f"experiment {exp.config.experiment_id:>4}: "
            f"{len(s['done'])} done, {len(s['failed'])} failed, {len(s['missing'])} missing"
        )


@click.group()
def main() -> None:
    """mechane: declarative sweeps and job running."""


for _name, _cmd in [
    ("setup", setup_lab),
    ("run", simulate),
    ("run-batch", simulate_batch),
    ("aggregate", aggregate_results),
    ("status", status),
]:
    main.add_command(_cmd, _name)

if __name__ == "__main__":
    main()
