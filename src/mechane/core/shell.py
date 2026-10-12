"""What an interactive session on a laboratory starts with."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mechane.core.laboratory import Laboratory


def shell_namespace(lab: Laboratory) -> dict[str, Any]:
    """The names available in `mechane shell`: the laboratory and ways to reach its parts."""
    return {
        "lab": lab,
        "experiments": lab.experiments,
        "experiment": lab.experiment,
        "build_instance": lab.build_instance,
    }
