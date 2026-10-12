"""Seed derivation: how an instance's RNG seed follows from what it is.

Pure functions of plain values, so the policy can be read and tested without building an
experiment. `Experiment.instance_seed` only chooses between the two schemes below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mechane.utils.serialization import stable_hash

SEED_BITS = 31
SEED_MASK = (1 << SEED_BITS) - 1  # seeds live in [0, 2**31): a non-negative C++ `int`


def mix31(x: int) -> int:
    """Bijection on [0, 2**31): distinct inputs give distinct outputs, yet neighbouring inputs
    give unrelated-looking outputs.

    Every step is invertible on 31 bits (xor-shift, multiplication by an odd constant mod 2**31),
    so the composition is too. This is the murmur3 finaliser adapted to 31 bits.
    """
    x &= SEED_MASK
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & SEED_MASK
    x ^= x >> 13
    x = (x * 0xC2B2AE35) & SEED_MASK
    x ^= x >> 16
    return x


def derived_base(seed: int, content_hash: str) -> int:
    """Starting point of an experiment's "derived" seed sequence.

    Depends on the user seed and the experiment's content hash only, so it survives sweep growth
    and `num_instances` changes.
    """
    payload = stable_hash({"seed": seed, "experiment": content_hash})
    return int(payload[:8], 16) & SEED_MASK


def derived_seed(base: int, instance_id: int) -> int:
    """The "derived" scheme: `mix31(base + instance_id)`.

    Because `mix31` is a bijection, two instances of the same experiment can never share a seed
    (for `instance_id < 2**31`). Seeds of different experiments are independent in practice, but
    cannot be guaranteed distinct: 31 bits are too few, and a seed must not depend on which other
    experiments are swept.
    """
    if not 0 <= instance_id < 1 << SEED_BITS:
        raise ValueError(f"instance_id must be in [0, 2**{SEED_BITS}), got {instance_id}")
    return mix31(base + instance_id)


def legacy_seed(config: Mapping[str, Any], instance_id: int) -> int:
    """The "legacy" scheme: the original formula, kept bit-for-bit so that result trees created
    by earlier versions stay reproducible. Guarantees nothing about uniqueness."""
    payload = stable_hash({"instance_id": instance_id, **config})
    return int(payload[:8], 16) & SEED_MASK
