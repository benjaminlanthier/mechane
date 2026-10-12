"""`Experiment`: one point of the sweep. It owns the directory tree its instances live in, their
seeds, and reading their results back. It does not run anything: see `mechane.core.runner`."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Literal, get_type_hints

from mechane.core.config import ExperimentConfig
from mechane.utils.seeding import derived_base, derived_seed, legacy_seed


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
    @cached_property
    def _seed_base(self) -> int:
        return derived_base(self.config.seed, self.config.content_hash())

    def instance_seed(self, instance_id: int) -> int:
        """Reproducible seed in [0, 2**31) (some C++ libs bind to a signed `int`).

        `seed_scheme = "derived"` (default) or `"legacy"`; see `mechane.utils.seeding` for what
        each guarantees.
        """
        if self.seed_scheme == "legacy":
            return legacy_seed(self.config.to_dict(), instance_id)
        return derived_seed(self._seed_base, instance_id)

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
