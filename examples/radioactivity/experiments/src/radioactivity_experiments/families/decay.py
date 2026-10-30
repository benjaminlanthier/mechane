"""The `decay` family: the fraction of atoms still alive at time `t`.

The science stays in `radioactivity`: an instance samples lifetimes and measures them with it.
"""

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
from radioactivity import sample_lifetimes, surviving_fraction


@dataclass(frozen=True)
class Decay(Config):
    rate: float
    n_atoms: int = 10_000


@dataclass(frozen=True)
class Observation(Config):
    t: float = 1.0


@dataclass(frozen=True)
class DecayConfigs(Configs):  # field order = directory nesting order
    decay: Decay
    observation: Observation


@dataclass(frozen=True)
class DecayExperimentConfig(ExperimentConfig):
    configs: DecayConfigs


class DecayExperiment(Experiment):
    config: DecayExperimentConfig


class DecayInstance(Instance):
    experiment: DecayExperiment

    def run(self):
        configs = self.experiment.config.configs
        lifetimes = sample_lifetimes(self.rng, configs.decay.rate, configs.decay.n_atoms)
        fraction = surviving_fraction(lifetimes, configs.observation.t)
        return InstanceOutput(outputs={"surviving_fraction": fraction})


class DecayLab(Laboratory):
    name = "decay"
    experiment_class = DecayExperiment
    instance_class = DecayInstance
