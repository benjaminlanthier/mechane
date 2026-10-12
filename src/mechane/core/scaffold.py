"""Starting files for a new laboratory.

`render_lab` only produces text (and `write_files` puts it on disk). The generated lab runs as
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
