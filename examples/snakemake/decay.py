"""Two small laboratories for the Snakemake example: Monte Carlo radioactive decay.

Each instance of either lab draws the lifetimes of `n_atoms` atoms from an exponential
distribution. `DecayLab` reports the fraction still alive at time `t` (exact: exp(-rate * t));
`HalfLifeLab` reports the median lifetime (exact: ln 2 / rate). `summarize.py` compares the
estimates with these exact answers.
"""

from dataclasses import dataclass

import numpy as np

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
        lifetimes = self.rng.exponential(1.0 / decay.rate, size=decay.n_atoms)
        return InstanceOutput(outputs={"half_life": float(np.median(lifetimes))})


class HalfLifeLab(Laboratory):
    name = "half_life"
    experiment_class = HalfLifeExperiment
    instance_class = HalfLifeInstance
