"""mechane: organise numerical experiments as Laboratory -> Experiment -> Instance."""

from mechane.config import Config
from mechane.experiment import Experiment, ExperimentConfig
from mechane.instance import Instance, InstanceResult
from mechane.laboratory import Laboratory, load_laboratory
from mechane.serialization import stable_hash, to_jsonable
from mechane.sweep import Sweep, expand, grid, zipped

__all__ = [
    "Config",
    "Experiment",
    "ExperimentConfig",
    "Instance",
    "InstanceResult",
    "Laboratory",
    "Sweep",
    "expand",
    "grid",
    "load_laboratory",
    "stable_hash",
    "to_jsonable",
    "zipped",
]
