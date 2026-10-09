from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

from mechane.config import Config
from mechane.serialization import stable_hash


@dataclass(frozen=True)
class ExperimentConfig:
    """One point of the parameter space: an ordered collection of named `Config` sections.

    The order of `sections` is the directory nesting order (see `Experiment.stage_dir`).
    Sections are reachable as attributes: `config.code`, `config.noise`, ...
    """

    experiment_id: int
    num_instances: int
    seed: int
    sections: dict[str, Config]

    def __getattr__(self, name: str) -> Config:
        sections = self.__dict__.get("sections")
        if sections is not None and name in sections:
            return sections[name]
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "num_instances": self.num_instances,
            "seed": self.seed,
            **{name: cfg.to_dict() for name, cfg in self.sections.items()},
        }

    def content_hash(self) -> str:
        """Hash of everything that influences results; independent of `experiment_id`,
        `num_instances` and every section's `hash_exclude` fields."""
        return stable_hash({name: cfg.hash_payload() for name, cfg in self.sections.items()})

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], section_classes: dict[str, type[Config]]
    ) -> ExperimentConfig:
        return cls(
            experiment_id=data["experiment_id"],
            num_instances=data["num_instances"],
            seed=data["seed"],
            sections={name: sc.from_dict(data[name]) for name, sc in section_classes.items()},
        )


class Experiment:
    """A single point of the sweep plus the directory tree it owns.

    Layout:  root / <section 1 path parts> / <section 2 path parts> / ... / instances / <id> /
    Every prefix of that path is a *stage directory*: put an artifact there (a compiled
    object, a partitioning, ...) and every experiment sharing that prefix shares the artifact.
    """

    #: "derived": seed = hash(base seed, content hash, instance id) -- stable when the sweep changes.
    #: "legacy":  seed = hash of the full config dict incl. experiment_id/num_instances, as in the
    #:            original `experiments` package. Use it to keep reproducing existing result trees.
    seed_scheme: ClassVar[Literal["derived", "legacy"]] = "derived"

    def __init__(self, root: Path, config: ExperimentConfig) -> None:
        self.root = Path(root)
        self.config = config

    # ---- paths ----------------------------------------------------------------------------
    def stage_dir(self, section: str) -> Path:
        path = self.root
        for name, cfg in self.config.sections.items():
            path = path.joinpath(*cfg.path_parts())
            if name == section:
                return path
        raise KeyError(f"No section {section!r}; sections are {list(self.config.sections)}")

    @property
    def instances_dir(self) -> Path:
        last = next(reversed(self.config.sections), None)
        base = self.stage_dir(last) if last else self.root
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
