"""Starting files for a new laboratory and for running it on SLURM.

These functions only produce text (and `write_files` puts it on disk). The generated lab runs as
it is: `setup`, `run-batch` and `aggregate` work immediately, and only the TODOs are yours.
"""

from __future__ import annotations

import keyword
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

DEFAULT_SECTIONS = ("model", "solver")

#: Names a section may not take: top-level keys of the parameter dict and manifest ...
RESERVED_SECTION_NAMES = frozenset({"experiment_id", "num_instances", "seed", "configs"})
#: ... and the names the generated module imports from mechane, which a section class must not
#: shadow.
MECHANE_NAMES = frozenset(
    {"Config", "Configs", "Experiment", "ExperimentConfig", "Instance", "InstanceOutput"}
    | {"Laboratory"}
)


class ScaffoldError(ValueError):
    """Names or paths that would produce broken or overwritten files."""


def _identifier(name: str, what: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        raise ScaffoldError(
            f"{what} {name!r} must be a lowercase identifier: letters, digits and underscores, "
            "starting with a letter"
        )
    return name


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def render_lab(name: str, sections: Sequence[str] = DEFAULT_SECTIONS) -> dict[str, str]:
    """`{filename: text}` for a laboratory module called `name` and a matching `params.py`."""
    _identifier(name, "laboratory name")
    names = list(dict.fromkeys(sections))
    if not names:
        raise ScaffoldError("a laboratory needs at least one section")
    base = _camel(name)
    generated = {f"{base}{suffix}" for suffix in ("Configs", "ExperimentConfig", "Experiment")}
    generated |= {f"{base}Instance", f"{base}Lab"}
    classes: dict[str, str] = {}
    for section in names:
        _identifier(section, "section name")
        if section in RESERVED_SECTION_NAMES:
            raise ScaffoldError(f"{section!r} is reserved: choose another section name")
        cls = _camel(section)
        if cls in MECHANE_NAMES or cls in generated or cls in classes.values():
            raise ScaffoldError(
                f"section {section!r} would produce a class named {cls}, which clashes"
            )
        classes[section] = cls

    section_classes = "\n\n".join(
        f"@dataclass(frozen=True)\nclass {cls}(Config):\n"
        "    example: int = 1  # TODO: replace with your parameters"
        for cls in classes.values()
    )
    fields = "\n".join(f"    {section}: {cls}" for section, cls in classes.items())
    module = f'''"""{name}: a mechane laboratory. Edit the sections and `run`; see params.py."""

from dataclasses import dataclass

from mechane import (
    Config,
    Configs,
    Experiment,
    ExperimentConfig,
    Instance,
    InstanceOutput,
    Laboratory,
)


{section_classes}


@dataclass(frozen=True)
class {base}Configs(Configs):  # field order = directory nesting order
{fields}


@dataclass(frozen=True)
class {base}ExperimentConfig(ExperimentConfig):
    configs: {base}Configs


class {base}Experiment(Experiment):
    config: {base}ExperimentConfig


class {base}Instance(Instance):
    experiment: {base}Experiment

    def run(self):
        # Your parameters: self.experiment.config.configs.<section>.<field>
        # TODO: replace this with your simulation. `self.rng` is seeded from this instance alone.
        sample = float(self.rng.normal())
        return InstanceOutput(outputs={{"sample": sample}})


class {base}Lab(Laboratory):
    name = "{name}"
    experiment_class = {base}Experiment
    instance_class = {base}Instance
'''
    first, *rest = classes
    section_params = [f'    "{first}": grid(example=[1, 2]),  # a sweep: 2 experiments']
    section_params += [f'    "{section}": {{"example": 1}},' for section in rest]
    params = f'''# Run mechane here with PYTHONPATH=. so that {name}.py can be imported.
from pathlib import Path

from mechane import grid

# Absolute, so that every mechane command finds the results wherever it is run from.
ROOT = Path("results/{name}").resolve()
LABORATORY = "{name}:{base}Lab"
PARAMS = {{
    "seed": 0,
    "num_instances": 10,
{chr(10).join(section_params)}
}}
'''
    return {f"{name}.py": module, "params.py": params}


def render_slurm(
    job_name: str = "mechane",
    time: str = "01:00:00",
    cpus_per_task: int = 1,
    mem: str = "4G",
    account: str | None = None,
    partition: str | None = None,
    container: str | None = None,
    binds: Sequence[str] = (),
    modules: Sequence[str] = (),
) -> str:
    """An sbatch script whose array tasks each run one task of a `mechane plan` file.

    `--array` is deliberately not in the script: `mechane plan --format array` gives it at
    submission time, so there is no number to keep in sync by hand.
    """
    directives = [f"--job-name={job_name}"]
    if account:
        directives.append(f"--account={account}")
    if partition:
        directives.append(f"--partition={partition}")
    directives += [
        "--ntasks=1",
        f"--cpus-per-task={cpus_per_task}",
        f"--mem={mem}",
        f"--time={time}",
        "--output=logs/%x_%A_%a.out",
        "--error=logs/%x_%A_%a.err",
    ]
    prefix = ""
    if container:
        bind_flags = "".join(f"--bind {b} " for b in binds)
        prefix = f"apptainer exec {bind_flags}{container} "
    module_lines = "".join(f"module load {m}\n" for m in modules)
    sbatch = "".join(f"#SBATCH {d}\n" for d in directives)
    return f"""#!/bin/bash
{sbatch}
# Each array task runs one task of a plan made with `mechane plan`:
#
#   mkdir -p logs
#   mechane setup --params-file params.py
#   ARRAY=$(mechane plan --manifest-path MANIFEST --batch-size 10 \\
#           --output plan.json --format array)
#   sbatch --array="$ARRAY" THIS_SCRIPT plan.json
#
# Add --unfinished to `mechane plan` to resubmit only what is missing or failed.

set -eo pipefail
{module_lines}set -u  # after `module load`: module systems often read unset variables

PLAN="${{1:?usage: sbatch --array=... THIS_SCRIPT plan.json}}"

{prefix}mechane run-plan --plan-file "$PLAN" --task-id "$SLURM_ARRAY_TASK_ID"
"""


def write_files(directory: Path, files: Mapping[str, str], force: bool = False) -> list[Path]:
    """Write `files` into `directory`, refusing to overwrite anything unless `force`."""
    directory = Path(directory)
    existing = [name for name in files if (directory / name).exists()]
    if existing and not force:
        raise ScaffoldError(f"{', '.join(existing)} already exist(s) in {directory}; use --force")
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in files.items():
        path = directory / name
        path.write_text(text)
        written.append(path)
    return written
