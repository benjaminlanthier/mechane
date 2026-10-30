"""The `half_life` family: the half-life, measured as the median lifetime.

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
from radioactivity import half_life, sample_lifetimes


@dataclass(frozen=True)
class Decay(Config):
    rate: float
    n_atoms: int = 10_000


@dataclass(frozen=True)
class HalfLifeConfigs(Configs):
    decay: Decay


@dataclass(frozen=True)
class HalfLifeExperimentConfig(ExperimentConfig):
    configs: HalfLifeConfigs


class HalfLifeExperiment(Experiment):
    config: HalfLifeExperimentConfig


class HalfLifeInstance(Instance):
    experiment: HalfLifeExperiment

    def run(self):
        decay = self.experiment.config.configs.decay
        lifetimes = sample_lifetimes(self.rng, decay.rate, decay.n_atoms)
        return InstanceOutput(outputs={"half_life": half_life(lifetimes)})


class HalfLifeLab(Laboratory):
    name = "half_life"
    experiment_class = HalfLifeExperiment
    instance_class = HalfLifeInstance
