"""The params file: an ordinary Python file that names a laboratory and its sweep.

It defines three names:

  * `ROOT`: directory that will hold the manifest and all results;
  * `LABORATORY`: the lab, as "package.module:ClassName", an entry-point name or the class
    itself (legacy files define `EXPERIMENT_TYPE` instead);
  * `PARAMS`: `{"seed": ..., "num_instances": ..., "<section>": <dict or sweep>, ...}`.

Or, to set up several labs at once, it defines `LABS` instead: a list of dicts, each with the
keys `ROOT`, `LABORATORY` and `PARAMS` above. Every lab has its own root, so its own manifest.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechane.core.registry import resolve_laboratory

if TYPE_CHECKING:
    from mechane.core.laboratory import Laboratory

SINGLE_LAB_NAMES = ("ROOT", "LABORATORY", "EXPERIMENT_TYPE", "PARAMS")


class ParamsFileError(ValueError):
    """The params file loaded, but does not say what a params file must say."""


@dataclass(frozen=True)
class ParamsFile:
    """One lab of a params file: where it lives, what it is, and its sweep."""

    root: Path
    laboratory: type[Laboratory]
    params: dict[str, Any]

    def build_laboratory(self) -> Laboratory:
        """Expand the sweep into a `Laboratory` (nothing is written: call `.setup()` for that)."""
        return self.laboratory.from_configs(root=self.root, configs=self.params)


def load_params_file(params_file: Path) -> ParamsFile:
    """Execute a params file that defines exactly one lab and return it."""
    labs = load_labs(params_file)
    if len(labs) != 1:
        raise ParamsFileError(f"{params_file} defines {len(labs)} labs; use load_labs.")
    return labs[0]


def load_labs(params_file: Path) -> list[ParamsFile]:
    """Execute a params file and return its labs: one, or one per entry of `LABS`."""
    namespace = vars(_execute(params_file))
    if "LABS" not in namespace:
        return [_lab(namespace, str(params_file))]

    mixed = [name for name in SINGLE_LAB_NAMES if name in namespace]
    if mixed:
        raise ParamsFileError(f"{params_file} defines LABS, so it cannot also define {mixed}.")
    entries = namespace["LABS"]
    if not isinstance(entries, list | tuple):
        raise ParamsFileError(f"{params_file}: LABS must be a list of dicts.")
    if not entries:
        raise ParamsFileError(f"{params_file}: LABS is empty.")

    labs = []
    for i, entry in enumerate(entries):
        where = f"{params_file}, LABS[{i}]"
        if not isinstance(entry, Mapping):
            raise ParamsFileError(f"{where} must be a dict, got {type(entry).__name__}.")
        unknown = set(entry) - set(SINGLE_LAB_NAMES)
        if unknown:
            raise ParamsFileError(
                f"{where} has unknown keys {sorted(unknown)}; expected ROOT, LABORATORY, PARAMS."
            )
        labs.append(_lab(entry, where))

    roots: dict[Path, int] = {}
    for i, lab in enumerate(labs):
        first = roots.setdefault(lab.root.resolve(), i)
        if first != i:
            raise ParamsFileError(
                f"{params_file}: LABS[{first}] and LABS[{i}] have the same ROOT ({lab.root}), "
                "so they would overwrite each other's manifest. Give each lab its own ROOT."
            )
    return labs


def _execute(params_file: Path) -> Any:
    spec = importlib.util.spec_from_file_location("external_params", params_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {params_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["external_params"] = module
    spec.loader.exec_module(module)
    return module


def _lab(names: Mapping[str, Any], where: str) -> ParamsFile:
    ref = names.get("LABORATORY") or names.get("EXPERIMENT_TYPE")
    if ref is None:
        raise ParamsFileError(f"{where} must define LABORATORY (or legacy EXPERIMENT_TYPE).")
    missing = [name for name in ("ROOT", "PARAMS") if name not in names]
    if missing:
        raise ParamsFileError(f"{where} must define {' and '.join(missing)}.")
    lab_cls = ref if isinstance(ref, type) else resolve_laboratory(str(ref))
    return ParamsFile(root=Path(names["ROOT"]), laboratory=lab_cls, params=names["PARAMS"])
