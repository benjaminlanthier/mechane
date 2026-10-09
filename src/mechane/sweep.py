"""Explicit parameter sweeps.

Nothing is inferred from types: a list is a literal value unless it is passed as an
*axis* to `grid` / `zipped`. Specs compose:

    grid({"bc": "periodic"}, L=[8, 16], T=[1.0, 2.0])   # 4 points (Cartesian product)
    zipped(T=[1.0, 2.0], n_steps=[1000, 5000])          # 2 points (paired)
    a * b                                               # product of two sweeps
    a + b                                               # concatenation
    {"name": "x", "params": zipped(...)}                # sweeps may be nested in dicts
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from itertools import product
from typing import Any


class Sweep(ABC):
    @abstractmethod
    def points(self) -> list[dict[str, Any]]:
        """Fully expanded list of parameter dicts."""

    def __mul__(self, other: Sweep | Mapping) -> Sweep:
        return _Product(self, other)

    def __add__(self, other: Sweep | Mapping) -> Sweep:
        return _Chain(self, other)

    def __len__(self) -> int:
        return len(self.points())


def _has_sweep(value: Any) -> bool:
    if isinstance(value, Sweep):
        return True
    return isinstance(value, Mapping) and any(_has_sweep(v) for v in value.values())


def expand(spec: Sweep | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a section spec (a plain dict, possibly containing nested sweeps, or a Sweep)."""
    if isinstance(spec, Sweep):
        return spec.points()
    if isinstance(spec, Mapping):
        keys = list(spec)
        choices = [expand(spec[k]) if _has_sweep(spec[k]) else [copy.deepcopy(spec[k])] for k in keys]
        return [dict(zip(keys, combo, strict=True)) for combo in product(*choices)]
    raise TypeError(f"Expected a dict or Sweep, got {type(spec).__name__}")


class _Axes(Sweep):
    def __init__(self, mode: str, fixed: Mapping[str, Any] | None, axes: Mapping[str, Any]):
        fixed = dict(fixed or {})
        overlap = set(fixed) & set(axes)
        if overlap:
            raise ValueError(f"Keys given both as fixed values and as axes: {sorted(overlap)}")
        for key, values in axes.items():
            if isinstance(values, (str, bytes, Mapping)) or not hasattr(values, "__iter__"):
                raise TypeError(f"Axis {key!r} must be a list/tuple/array of values, got {values!r}")
        self.mode, self.fixed, self.axes = mode, fixed, {k: list(v) for k, v in axes.items()}

    def points(self) -> list[dict[str, Any]]:
        keys = list(self.axes)
        if not keys:
            combos: Any = [()]
        elif self.mode == "grid":
            combos = product(*self.axes.values())
        else:
            lengths = {k: len(v) for k, v in self.axes.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"zipped() axes must have equal lengths, got {lengths}")
            combos = zip(*self.axes.values(), strict=True)
        out: list[dict[str, Any]] = []
        for combo in combos:
            out.extend(expand({**self.fixed, **dict(zip(keys, combo, strict=True))}))
        return out


def grid(fixed: Mapping[str, Any] | None = None, /, **axes: Any) -> Sweep:
    """Cartesian product over `axes`; `fixed` values are copied into every point."""
    return _Axes("grid", fixed, axes)


def zipped(fixed: Mapping[str, Any] | None = None, /, **axes: Any) -> Sweep:
    """Pair up `axes` element-wise (all must have the same length)."""
    return _Axes("zip", fixed, axes)


class _Product(Sweep):
    def __init__(self, a: Sweep | Mapping, b: Sweep | Mapping):
        self.a, self.b = a, b

    def points(self) -> list[dict[str, Any]]:
        out = []
        for pa in expand(self.a):
            for pb in expand(self.b):
                overlap = set(pa) & set(pb)
                if overlap:
                    raise ValueError(f"Cannot multiply sweeps sharing keys: {sorted(overlap)}")
                out.append({**pa, **pb})
        return out


class _Chain(Sweep):
    def __init__(self, a: Sweep | Mapping, b: Sweep | Mapping):
        self.a, self.b = a, b

    def points(self) -> list[dict[str, Any]]:
        return expand(self.a) + expand(self.b)
