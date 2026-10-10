"""Seed helpers."""

from __future__ import annotations

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
