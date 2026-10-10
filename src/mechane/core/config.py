"""`Config`: the building block of an experiment's parameter space.

A *section* is a frozen dataclass inheriting from `Config`. Each section knows
  * how to serialize itself (`to_dict`),
  * which of its fields influence results (`hash_exclude` lists the ones that don't),
  * which directory components it contributes (`path_parts`, by default its hash).
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, ClassVar, Self, get_args, get_type_hints

from mechane.utils.serialization import stable_hash, to_jsonable


def _enum_type(tp: Any) -> type[Enum] | None:
    if isinstance(tp, type) and issubclass(tp, Enum):
        return tp
    for arg in get_args(tp):
        found = _enum_type(arg)
        if found is not None:
            return found
    return None


@cache
def _enum_fields(cls: type[Config]) -> dict[str, type[Enum]]:
    try:
        hints = get_type_hints(cls)
    except Exception:  # unresolved forward refs: skip coercion rather than fail
        return {}
    out = {}
    for f in dataclasses.fields(cls):
        enum_cls = _enum_type(hints.get(f.name))
        if enum_cls is not None:
            out[f.name] = enum_cls
    return out


class Config:
    """Mixin for frozen dataclasses describing one section of an experiment."""

    #: Fields that do NOT change the numerical result (backend, dtype, caching flags, ...).
    #: They are left out of the hash, so they never change the directory or the seed.
    hash_exclude: ClassVar[tuple[str, ...]] = ()

    if TYPE_CHECKING:
        # Subclasses are frozen dataclasses; declaring this makes `Config` satisfy the
        # `DataclassInstance` protocol so `dataclasses.fields(cls)` type-checks.
        __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

    def __post_init__(self) -> None:
        for name, enum_cls in _enum_fields(type(self)).items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, enum_cls):
                object.__setattr__(self, name, enum_cls(value))

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def hash_payload(self) -> dict[str, Any]:
        return {k: v for k, v in self.to_dict().items() if k not in self.hash_exclude}

    def hash(self) -> str:
        return stable_hash(self.hash_payload())

    def path_parts(self) -> tuple[str, ...]:
        """Directory components this section contributes to an experiment's path."""
        return (self.hash(),)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)  # type: ignore[call-arg]
