"""The science package's own tests: they need neither mechane nor Snakemake."""

import math

import numpy as np
import pytest

from radioactivity import (
    exact_half_life,
    exact_surviving_fraction,
    half_life,
    sample_lifetimes,
    surviving_fraction,
)


def test_the_estimates_converge_to_the_exact_answers():
    lifetimes = sample_lifetimes(np.random.default_rng(0), rate=2.0, n_atoms=200_000)

    assert surviving_fraction(lifetimes, 1.0) == pytest.approx(math.exp(-2.0), abs=0.005)
    assert half_life(lifetimes) == pytest.approx(math.log(2) / 2.0, rel=0.01)


def test_the_exact_answers_agree_with_each_other():
    rate = 0.7
    assert exact_surviving_fraction(rate, exact_half_life(rate)) == pytest.approx(0.5)


def test_the_same_generator_gives_the_same_lifetimes():
    first = sample_lifetimes(np.random.default_rng(42), rate=1.0, n_atoms=10)
    second = sample_lifetimes(np.random.default_rng(42), rate=1.0, n_atoms=10)
    assert np.array_equal(first, second)


def test_a_rate_must_be_positive():
    with pytest.raises(ValueError, match="rate must be positive"):
        sample_lifetimes(np.random.default_rng(0), rate=0.0, n_atoms=10)
