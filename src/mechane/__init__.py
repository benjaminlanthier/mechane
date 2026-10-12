"""mechane: organise numerical experiments as Laboratory -> Experiment -> Instance."""

from mechane.core.config import Config, Configs, ExperimentConfig
from mechane.core.experiment import Experiment
from mechane.core.instance import Instance, InstanceOutput
from mechane.core.laboratory import Laboratory
from mechane.core.manifest import load_laboratory
from mechane.utils.serialization import stable_hash, to_jsonable
from mechane.utils.sweep import Sweep, expand, grid, zipped

__all__ = [
    "Config",
    "Configs",
    "Experiment",
    "ExperimentConfig",
    "Instance",
    "InstanceOutput",
    "Laboratory",
    "Sweep",
    "expand",
    "grid",
    "load_laboratory",
    "stable_hash",
    "to_jsonable",
    "zipped",
]
