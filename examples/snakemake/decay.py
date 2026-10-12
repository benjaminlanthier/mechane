"""A small laboratory for the Snakemake example: Monte Carlo radioactive decay.

Each instance draws the lifetime of `n_atoms` atoms from an exponential distribution and reports
the fraction still alive at time `t`. The exact answer is exp(-rate * t), which `summarize.py`
compares the estimates with.
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
        lifetimes = self.rng.exponential(1.0 / configs.decay.rate, size=configs.decay.n_atoms)
        alive = float((lifetimes > configs.observation.t).mean())
        return InstanceOutput(outputs={"surviving_fraction": alive})


class DecayLab(Laboratory):
    name = "decay"
    experiment_class = DecayExperiment
    instance_class = DecayInstance
