"""The params file: an ordinary Python file that names a laboratory and its sweep.

It defines three names:

  * `ROOT`: directory that will hold the manifest and all results;
  * `LABORATORY`: the lab, as "package.module:ClassName", an entry-point name or the class
    itself (legacy files define `EXPERIMENT_TYPE` instead);
  * `PARAMS`: `{"seed": ..., "num_instances": ..., "<section>": <dict or sweep>, ...}`.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mechane.core.registry import resolve_laboratory

if TYPE_CHECKING:
    from mechane.core.laboratory import Laboratory


class ParamsFileError(ValueError):
    """The params file loaded, but does not say what a params file must say."""


@dataclass(frozen=True)
class ParamsFile:
    root: Path
    laboratory: type[Laboratory]
    params: dict[str, Any]

    def build_laboratory(self) -> Laboratory:
        """Expand the sweep into a `Laboratory` (nothing is written: call `.setup()` for that)."""
        return self.laboratory.from_configs(root=self.root, configs=self.params)


def load_params_file(params_file: Path) -> ParamsFile:
    """Execute a params file and return what it defines."""
    spec = importlib.util.spec_from_file_location("external_params", params_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {params_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["external_params"] = module
    spec.loader.exec_module(module)

    ref = getattr(module, "LABORATORY", None) or getattr(module, "EXPERIMENT_TYPE", None)
    if ref is None:
        raise ParamsFileError(f"{params_file} must define LABORATORY (or legacy EXPERIMENT_TYPE).")
    lab_cls = ref if isinstance(ref, type) else resolve_laboratory(str(ref))
    return ParamsFile(root=Path(module.ROOT), laboratory=lab_cls, params=module.PARAMS)
