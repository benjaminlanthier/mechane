"""Lifetimes of radioactive atoms, and the quantities measured on them.

The lifetime of an atom that decays at `rate` is exponential with mean 1 / rate, so the fraction
still alive at time `t` is exp(-rate * t) and the half-life is ln 2 / rate.
"""

import math

import numpy as np


def sample_lifetimes(rng: np.random.Generator, rate: float, n_atoms: int) -> np.ndarray:
    """The lifetimes of `n_atoms` atoms that decay at `rate`."""
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    return rng.exponential(1.0 / rate, size=n_atoms)


def surviving_fraction(lifetimes: np.ndarray, t: float) -> float:
    """The fraction of the atoms still alive at time `t`."""
    return float((lifetimes > t).mean())


def half_life(lifetimes: np.ndarray) -> float:
    """The time by which half of the atoms have decayed: the median lifetime."""
    return float(np.median(lifetimes))


def exact_surviving_fraction(rate: float, t: float) -> float:
    return math.exp(-rate * t)


def exact_half_life(rate: float) -> float:
    return math.log(2) / rate
