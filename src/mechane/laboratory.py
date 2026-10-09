from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any, ClassVar

from mechane.config import Config
from mechane.experiment import Experiment, ExperimentConfig
from mechane.instance import Instance
from mechane.io import atomic_write_json
from mechane.provenance import collect_provenance
from mechane.registry import laboratory_ref, resolve_laboratory
from mechane.sweep import expand


class Laboratory:
    """Declares a family of experiments and owns everything generic about running them:
    sweep expansion, the manifest, directory setup and aggregation.

    Subclass it and set the class variables:

        class MyLab(Laboratory):
            name = "my-lab"
            configs_classes = {"model": Model, "solver": Solver}   # order = directory nesting
            instance_class = MyInstance
    """

    name: ClassVar[str]
    configs_classes: ClassVar[dict[str, type[Config]]]
    instance_class: ClassVar[type[Instance]]
    experiment_class: ClassVar[type[Experiment]] = Experiment
    #: Order in which sections are multiplied to number the experiments. Defaults to the
    #: directory order. Set it explicitly to keep experiment ids stable when porting.
    sweep_order: ClassVar[tuple[str, ...] | None] = None
    track_packages: ClassVar[tuple[str, ...]] = ("mechane", "numpy")

    def __init__(
        self,
        root: Path,
        experiments: Sequence[Experiment],
        num_instances: int,
        seed: int,
        manifest_path: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.experiments = list(experiments)
        self.num_instances = num_instances
        self.seed = seed
        self._manifest_path = manifest_path or self.default_manifest_path(
            self.root, self.experiments
        )
        self._by_id = {e.config.experiment_id: e for e in self.experiments}

    # ---- construction ---------------------------------------------------------------------
    @classmethod
    def from_configs(cls, root: Path, configs: dict[str, Any]) -> Laboratory:
        """Build from a configs dict: {"seed": ..., "num_instances": ..., <section>: spec, ...}."""
        allowed = {"seed", "num_instances", *cls.configs_classes}
        unknown = set(configs) - allowed
        if unknown:
            raise ValueError(
                f"Unknown configs keys {sorted(unknown)}; expected a subset of {sorted(allowed)}"
            )
        seed = configs.get("seed", 0)
        num_instances = configs.get("num_instances", 1)

        order = cls.sweep_order or tuple(cls.configs_classes)
        if set(order) != set(cls.configs_classes):
            raise ValueError("sweep_order must list exactly the keys of configs_classes")

        axes = {
            name: [cls.configs_classes[name].from_dict(p) for p in expand(configs.get(name, {}))]
            for name in order
        }
        experiments = []
        for experiment_id, combo in enumerate(product(*(axes[n] for n in order))):
            chosen = dict(zip(order, combo, strict=True))
            config = ExperimentConfig(
                experiment_id=experiment_id,
                num_instances=num_instances,
                seed=seed,
                configs={n: chosen[n] for n in cls.configs_classes},
            )
            experiments.append(cls.experiment_class(root=Path(root), config=config))
        if not experiments:
            raise ValueError("The sweep is empty: some section expanded to zero points.")
        return cls(
            root=Path(root),
            experiments=experiments,
            num_instances=num_instances,
            seed=seed,
        )

    @classmethod
    def from_manifest(cls, manifest_path: Path) -> Laboratory:
        manifest = read_manifest(manifest_path)
        root = Path(manifest["root"])
        experiments = [
            cls.experiment_class(
                root=root, config=ExperimentConfig.from_dict(data, cls.configs_classes)
            )
            for data in manifest["experiments"].values()
        ]
        return cls(
            root=root,
            experiments=experiments,
            num_instances=manifest["num_instances_per_experiment"],
            seed=manifest["seed"],
            manifest_path=Path(manifest_path),
        )

    @classmethod
    def default_manifest_path(cls, root: Path, experiments: Sequence[Experiment]) -> Path:
        return Path(root) / "manifest.json"

    # ---- lookups --------------------------------------------------------------------------
    def experiment(self, experiment_id: int) -> Experiment:
        try:
            return self._by_id[experiment_id]
        except KeyError:
            raise KeyError(
                f"No experiment {experiment_id}; valid ids: 0..{len(self._by_id) - 1}"
            ) from None

    def build_instance(self, experiment_id: int, instance_id: int) -> Instance:
        return self.instance_class(self.experiment(experiment_id), instance_id)

    def provenance(self) -> dict[str, Any]:
        try:
            git_dir: Path | None = Path(inspect.getfile(type(self))).resolve().parent
        except (TypeError, OSError):
            git_dir = None
        return collect_provenance(self.track_packages, git_dir)

    # ---- setup / manifest -----------------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def all_results_path(self) -> Path:
        return self.manifest_path.parent / "aggregated_results.json"

    @property
    def num_experiments(self) -> int:
        return len(self.experiments)

    @property
    def total_num_instances(self) -> int:
        return self.num_experiments * self.num_instances

    def _check_unique_directories(self) -> None:
        seen: dict[Path, int] = {}
        for exp in self.experiments:
            other = seen.setdefault(exp.instances_dir, exp.config.experiment_id)
            if other != exp.config.experiment_id:
                raise ValueError(
                    f"Experiments {other} and {exp.config.experiment_id} map to the same directory "
                    f"({exp.instances_dir}). They differ only in fields listed in `hash_exclude`, "
                    "or the sweep contains a duplicate point."
                )

    def manifest(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "laboratory": laboratory_ref(type(self)),
            "experiment_type": self.name,
            "num_experiments": self.num_experiments,
            "num_instances_per_experiment": self.num_instances,
            "seed": self.seed,
            "experiments": {
                str(e.config.experiment_id): {
                    **e.config.to_dict(),
                    "experiment_path": str(e.instances_dir),
                }
                for e in self.experiments
            },
        }

    def setup(self) -> None:
        """Write the manifest and create every instances directory up front, so array tasks
        never race on `mkdir`."""
        ref = laboratory_ref(type(self))
        try:
            resolved = resolve_laboratory(ref)
        except Exception as exc:
            raise RuntimeError(
                f"{ref} is not importable by reference (defined in __main__ or a local scope?). "
                "Define the Laboratory in an importable module so jobs can find it."
            ) from exc
        if resolved is not type(self):
            raise RuntimeError(f"{ref} resolves to a different class than {type(self)}")
        self._check_unique_directories()
        atomic_write_json(self.manifest_path, self.manifest())
        for exp in self.experiments:
            exp.instances_dir.mkdir(parents=True, exist_ok=True)
            exp.migrate_padding()

    # ---- aggregation ----------------------------------------------------------------------
    def aggregate_results(self) -> None:
        all_results: dict[str, Any] = {}
        if self.all_results_path.exists():
            with self.all_results_path.open() as f:
                all_results = json.load(f)
        for exp in self.experiments:
            all_results[str(exp.config.experiment_id)] = exp.aggregate_instance_results()
        atomic_write_json(self.all_results_path, all_results)


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    with manifest_path.open() as f:
        return json.load(f)


def load_laboratory(manifest_path: Path) -> Laboratory:
    """Rebuild the right Laboratory subclass from a manifest alone."""
    manifest = read_manifest(manifest_path)
    ref = manifest.get("laboratory") or manifest["experiment_type"]  # older manifests: type name
    return resolve_laboratory(ref).from_manifest(manifest_path)
