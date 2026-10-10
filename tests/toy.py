"""A deliberately non-QEC lab: Monte-Carlo estimate of E[x^2] for x ~ N(0, sigma) on a
lattice of size L. Used by the tests and as the template for new labs."""

from dataclasses import dataclass
from enum import StrEnum

from mechane import (
    Config,
    Configs,
    Experiment,
    ExperimentConfig,
    Instance,
    InstanceOutput,
    Laboratory,
)


class Kind(StrEnum):
    GAUSS = "gauss"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class Model(Config):
    L: int
    kind: Kind = Kind.GAUSS


@dataclass(frozen=True)
class Solver(Config):
    sigma: float
    n_samples: int = 1000
    device: str = "cpu"
    hash_exclude = ("device",)


@dataclass(frozen=True)
class ToyConfigs(Configs):
    model: Model
    solver: Solver


@dataclass(frozen=True)
class ToyExperimentConfig(ExperimentConfig):
    configs: ToyConfigs


class ToyExperiment(Experiment):
    config: ToyExperimentConfig


class ToyInstance(Instance):
    experiment: ToyExperiment

    def run(self):
        configs = self.experiment.config.configs  # typed path: ToyConfigs
        model, solver = configs.model, configs.solver
        if solver.sigma < 0:
            raise ValueError("negative sigma")
        x = self.rng.normal(0.0, solver.sigma, size=(solver.n_samples, model.L))
        return InstanceOutput(
            inputs={"first": x[0, :2]},  # numpy array: must serialize
            outputs={"mean_square": float((x**2).mean())},
        )


class ToyLab(Laboratory):
    name = "toy"
    experiment_class = ToyExperiment
    instance_class = ToyInstance
