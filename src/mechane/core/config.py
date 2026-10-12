"""The configuration data model: what one experiment is *parameterised by*.

  * `Config`: one *section* of parameters, a frozen dataclass. A section knows
      - how to serialize itself (`to_dict`),
      - which of its fields influence results (`hash_exclude` lists the ones that don't),
      - which directory components it contributes (`path_parts`, by default its hash).
  * `Configs`: the ordered collection of sections. Its field order is the directory nesting order.
  * `ExperimentConfig`: the `Configs` of one experiment plus its id, seed and instance count.

This module knows nothing about directories, jobs or results.
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


RESERVED = frozenset({"experiment_id", "num_instances", "seed", "configs"})


@dataclasses.dataclass(frozen=True)
class Configs:
    """Base class for the user's collection of sections. Field order = nesting order."""

    def __post_init__(self) -> None:
        clash = RESERVED & set(self.names())
        if clash:
            raise ValueError(f"Config names {sorted(clash)} are reserved")

    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(self))

    def as_dict(self) -> dict[str, Config]:
        return {name: getattr(self, name) for name in self.names()}

    @classmethod
    def section_classes(cls) -> dict[str, type[Config]]:
        """Section name -> `Config` subclass, read from the field annotations (in field order)."""
        hints = get_type_hints(cls)
        out: dict[str, type[Config]] = {}
        for f in dataclasses.fields(cls):
            tp = hints[f.name]
            if not (isinstance(tp, type) and issubclass(tp, Config)):
                raise TypeError(
                    f"{cls.__name__}.{f.name} must be annotated with a Config subclass, got {tp!r}"
                )
            out[f.name] = tp
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**{n: tp.from_dict(data[n]) for n, tp in cls.section_classes().items()})


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: int
    num_instances: int
    seed: int
    configs: Configs

    def __getattr__(self, name: str) -> Config:
        configs = self.__dict__.get("configs")
        if configs is not None and name in configs.names():
            return getattr(configs, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "num_instances": self.num_instances,
            "seed": self.seed,
            **{name: cfg.to_dict() for name, cfg in self.configs.as_dict().items()},
        }

    def content_hash(self) -> str:
        """Hash of everything that influences results; independent of `experiment_id`,
        `num_instances` and every section's `hash_exclude` fields."""
        return stable_hash(
            {name: cfg.hash_payload() for name, cfg in self.configs.as_dict().items()}
        )

    @classmethod
    def configs_class(cls) -> type[Configs]:
        """The `Configs` subclass, read from this class's `configs` annotation."""
        return get_type_hints(cls)["configs"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            experiment_id=data["experiment_id"],
            num_instances=data["num_instances"],
            seed=data["seed"],
            configs=cls.configs_class().from_dict(data),
        )
