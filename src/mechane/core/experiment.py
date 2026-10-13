"""`Experiment`: one point of the sweep. It owns the directory tree its instances live in, their
seeds, and reading their results back. It does not run anything: see `mechane.core.runner`."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Literal, get_type_hints

from mechane.core.config import ExperimentConfig
from mechane.utils.seeding import derived_base, derived_seed, legacy_seed


class Experiment:
    seed_scheme: ClassVar[Literal["derived", "legacy"]] = "derived"
    RESULTS_FILE: ClassVar[str] = "results.json"  # what an instance leaves when it finishes
    ERROR_FILE: ClassVar[str] = "job.err"  # what it leaves when it raises
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

    @cached_property
    def instances_dir(self) -> Path:
        # Computed once: it hashes every section, and almost everything else is built from it.
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
        return self.instance_dir(instance_id) / self.RESULTS_FILE

    def error_path(self, instance_id: int) -> Path:
        return self.instance_dir(instance_id) / self.ERROR_FILE

    def present_instance_ids(self) -> list[int]:
        """The ids below `num_instances` that have a directory under `instances/`, ascending.

        One directory listing, not one probe per instance: with a million instances of which a
        thousand have run, looking at the instances costs a thousand probes, not a million. A
        directory counts only if it is named the way `instance_dir` names it (so a stray
        `instances/5` next to `instances/05`, or a file called `07`, is not an instance).
        """
        width, n = self.pad_length, self.config.num_instances
        ids: list[int] = []
        try:
            with os.scandir(self.instances_dir) as entries:
                for entry in entries:
                    name = entry.name
                    is_instance_name = len(name) == width and name.isascii() and name.isdigit()
                    if is_instance_name and int(name) < n and entry.is_dir():
                        ids.append(int(name))
        except (FileNotFoundError, NotADirectoryError):
            return []
        ids.sort()
        return ids

    def iter_results(self) -> Iterator[tuple[int, dict[str, Any]]]:
        base, width = os.fspath(self.instances_dir), self.pad_length
        for i in self.present_instance_ids():
            try:
                with open(
                    os.path.join(base, f"{i:0{width}d}", self.RESULTS_FILE), encoding="utf-8"
                ) as f:
                    record = json.load(f)
            except FileNotFoundError:  # the directory exists, the instance has not finished
                continue
            yield i, record

    def status(self) -> dict[str, list[int]]:
        base, width = os.fspath(self.instances_dir), self.pad_length
        done: list[int] = []
        failed: list[int] = []
        for i in self.present_instance_ids():
            directory = os.path.join(base, f"{i:0{width}d}")
            if os.path.exists(os.path.join(directory, self.RESULTS_FILE)):
                done.append(i)
            elif os.path.exists(os.path.join(directory, self.ERROR_FILE)):
                failed.append(i)
        seen = set(done)
        seen.update(failed)
        missing = [i for i in range(self.config.num_instances) if i not in seen]
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
