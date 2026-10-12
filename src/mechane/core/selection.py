"""Choosing which experiments a command applies to: by id, and/or by parameter value.

A filter is `KEY=VALUE`, where KEY is a dotted parameter name as it appears in the manifest
(`model.L`, `solver.sigma`, or the top-level `seed`) and VALUE is read as JSON when it parses
(`3`, `1e-3`, `true`, `[1, 2]`) and as plain text otherwise (`gauss`). Filters are ANDed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from mechane.utils.serialization import flatten

if TYPE_CHECKING:
    from mechane.core.experiment import Experiment
    from mechane.core.laboratory import Laboratory

_IDENTITY_KEYS = frozenset({"experiment_id", "num_instances", "seed"})


class SelectionError(ValueError):
    """A filter or an experiment id that does not make sense for this laboratory."""


def queryable(experiment: Experiment) -> dict[str, Any]:
    """Everything a filter can name: the identity keys plus every section parameter."""
    return flatten(experiment.config.to_dict())


def parameters_of(experiment: Experiment) -> dict[str, Any]:
    """The flat section parameters of an experiment, e.g. `{"model.L": 2, "solver.sigma": 1.0}`."""
    return {k: v for k, v in queryable(experiment).items() if k not in _IDENTITY_KEYS}


def _parse_clause(clause: str) -> tuple[str, Any]:
    key, separator, raw = clause.partition("=")
    key = key.strip()
    if not separator or not key:
        raise SelectionError(f"invalid filter {clause!r}: expected KEY=VALUE, e.g. model.L=3")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


def select_experiments(
    lab: Laboratory, experiment_ids: Sequence[int] = (), where: Iterable[str] = ()
) -> list[Experiment]:
    """The experiments with one of `experiment_ids` (all when empty) that satisfy every filter,
    in experiment-id order."""
    if experiment_ids:
        try:
            chosen = [lab.experiment(i) for i in dict.fromkeys(experiment_ids)]
        except KeyError as exc:
            raise SelectionError(str(exc.args[0])) from exc
        chosen.sort(key=lambda e: e.config.experiment_id)
    else:
        chosen = list(lab.experiments)

    known = queryable(lab.experiments[0])
    for key, value in map(_parse_clause, where):
        if key not in known:
            raise SelectionError(f"unknown parameter {key!r}; available: {sorted(known)}")
        chosen = [e for e in chosen if queryable(e)[key] == value]
    return chosen
