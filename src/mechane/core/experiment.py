from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, get_type_hints

from mechane.core.config import Config
from mechane.utils.serialization import stable_hash

RESERVED = frozenset({"experiment_id", "num_instances", "seed", "configs"})


@dataclass(frozen=True)
class Configs:
    """Base class for the user's collection of sections. Field order = nesting order."""

    def __post_init__(self) -> None:
        clash = RESERVED & set(self.names())
        if clash:
            raise ValueError(f"Config names {sorted(clash)} are reserved")

    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    def as_dict(self) -> dict[str, Config]:
        return {name: getattr(self, name) for name in self.names()}

    @classmethod
    def section_classes(cls) -> dict[str, type[Config]]:
        """Section name -> `Config` subclass, read from the field annotations (in field order)."""
        hints = get_type_hints(cls)
        out: dict[str, type[Config]] = {}
        for f in fields(cls):
            tp = hints[f.name]
            if not (isinstance(tp, type) and issubclass(tp, Config)):
                raise TypeError(
                    f"{cls.__name__}.{f.name} must be annotated with a Config subclass, got {tp!r}"
                )
            out[f.name] = tp
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**{n: tp.from_dict(data[n]) for n, tp in cls.section_classes().items()})


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: int
    num_instances: int
    seed: int
    configs: Configs

    def __getattr__(self, name: str) -> Config:
        configs = self.__dict__.get("configs")
        if configs is not None and name in configs.names():
            return getattr(configs, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "num_instances": self.num_instances,
            "seed": self.seed,
            **{name: cfg.to_dict() for name, cfg in self.configs.as_dict().items()},
        }

    def content_hash(self) -> str:
        """Hash of everything that influences results; independent of `experiment_id`,
        `num_instances` and every section's `hash_exclude` fields."""
        return stable_hash(
            {name: cfg.hash_payload() for name, cfg in self.configs.as_dict().items()}
        )

    @classmethod
    def configs_class(cls) -> type[Configs]:
        """The `Configs` subclass, read from this class's `configs` annotation."""
        return get_type_hints(cls)["configs"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            experiment_id=data["experiment_id"],
            num_instances=data["num_instances"],
            seed=data["seed"],
            configs=cls.configs_class().from_dict(data),
        )


class Experiment:
    seed_scheme: ClassVar[Literal["derived", "legacy"]] = "derived"
    config: ExperimentConfig  # subclasses narrow this; `config_class()` reads it back

    @classmethod
    def config_class(cls) -> type[ExperimentConfig]:
        return get_type_hints(cls)["config"]

    def __init__(self, root: Path, config: ExperimentConfig) -> None:
        self.root = Path(root)
        self.config = config

    def stage_dir(self, section: str) -> Path:
        path = self.root
        for name, cfg in self.config.configs.as_dict().items():
            path = path.joinpath(*cfg.path_parts())
            if name == section:
                return path
        raise KeyError(f"No section {section!r}; configs are {list(self.config.configs.names())}")

    @property
    def instances_dir(self) -> Path:
        names = self.config.configs.names()
        base = self.stage_dir(names[-1]) if names else self.root
        return base / "instances"

    @property
    def pad_length(self) -> int:
        return len(str(max(self.config.num_instances - 1, 0)))

    def instance_dir(self, instance_id: int) -> Path:
        return self.instances_dir / f"{instance_id:0{self.pad_length}d}"

    # ---- seeding --------------------------------------------------------------------------
    def instance_seed(self, instance_id: int) -> int:
        """Unique, reproducible seed in the signed 32-bit range (some C++ libs bind to `int`)."""
        if self.seed_scheme == "legacy":
            payload = stable_hash({"instance_id": instance_id, **self.config.to_dict()})
        else:
            payload = stable_hash(
                {
                    "seed": self.config.seed,
                    "experiment": self.config.content_hash(),
                    "instance_id": instance_id,
                }
            )
        return int(payload[:8], 16) & 0x7FFFFFFF

    # ---- results & aggregation -------------------------------------------------------------
    def results_path(self, instance_id: int) -> Path:
        return self.instance_dir(instance_id) / "results.json"

    def error_path(self, instance_id: int) -> Path:
        return self.instance_dir(instance_id) / "job.err"

    def iter_results(self) -> Iterator[tuple[int, dict[str, Any]]]:
        for i in range(self.config.num_instances):
            path = self.results_path(i)
            if path.exists():
                with path.open() as f:
                    yield i, json.load(f)

    def status(self) -> dict[str, list[int]]:
        done, failed, missing = [], [], []
        for i in range(self.config.num_instances):
            if self.results_path(i).exists():
                done.append(i)
            elif self.error_path(i).exists():
                failed.append(i)
            else:
                missing.append(i)
        return {"done": done, "failed": failed, "missing": missing}

    def summarize(self, result: dict[str, Any]) -> Any:
        """Reduce one instance's result record to what the aggregate file should keep.
        Override this: it is the only experiment-specific part of aggregation."""
        return result.get("outputs", {})

    def aggregate_instance_results(self) -> dict[str, Any]:
        status = self.status()
        return {
            "num_expected": self.config.num_instances,
            "num_done": len(status["done"]),
            "failed": status["failed"],
            "missing": status["missing"],
            "instances": {str(i): self.summarize(r) for i, r in self.iter_results()},
        }

    # ---- maintenance ----------------------------------------------------------------------
    def migrate_padding(self) -> None:
        """Rename numeric instance dirs ('5' -> '005') after `num_instances` grew a digit."""
        if not self.instances_dir.exists():
            return
        for path in list(self.instances_dir.iterdir()):
            if path.is_dir() and path.name.isdigit() and len(path.name) != self.pad_length:
                target = self.instance_dir(int(path.name))
                if not target.exists():
                    path.rename(target)
