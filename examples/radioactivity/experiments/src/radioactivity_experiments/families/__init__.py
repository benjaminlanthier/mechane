"""The experiment families of the radioactivity project: one module, and one mechane laboratory,
per family.

Each module is self-contained: it defines its own sections, so a change to one family never
affects another. Each lab is registered as an entry point in `experiments/pyproject.toml`.
"""

from radioactivity_experiments.families.decay import DecayLab
from radioactivity_experiments.families.half_life import HalfLifeLab

__all__ = ["DecayLab", "HalfLifeLab"]
