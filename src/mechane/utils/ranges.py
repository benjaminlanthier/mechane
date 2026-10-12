"""Compact notation for lists of instance ids: `0-2,5,7-8`."""

from __future__ import annotations

import re
from collections.abc import Iterable

_PART = re.compile(r"(\d+)(?:-(\d+))?")


def _span(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def compact_ranges(ids: Iterable[int]) -> str:
    """`[0, 1, 2, 5, 7, 8]` -> `"0-2,5,7-8"` (sorted, duplicates removed)."""
    spans: list[str] = []
    start: int | None = None
    previous = 0
    for value in sorted(set(ids)):
        if start is not None and value == previous + 1:
            previous = value
            continue
        if start is not None:
            spans.append(_span(start, previous))
        start = previous = value
    if start is not None:
        spans.append(_span(start, previous))
    return ",".join(spans)


def parse_ranges(text: str) -> list[int]:
    """`"0-2,5"` -> `[0, 1, 2, 5]` (first-seen order, duplicates removed).

    Raises `ValueError` for anything that is not a non-negative integer or an ascending range.
    """
    ids: dict[int, None] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = _PART.fullmatch(part)
        if match is None:
            raise ValueError(f"invalid id or range {part!r} (expected something like 0-9,15)")
        low = int(match[1])
        high = int(match[2]) if match[2] is not None else low
        if high < low:
            raise ValueError(f"empty range {part!r}")
        ids.update(dict.fromkeys(range(low, high + 1)))
    return list(ids)
