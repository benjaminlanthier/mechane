"""Compare the Monte Carlo estimates in the exported tables with the exact answers.

Usage: python summarize.py outputs/decay.csv outputs/half_life.csv
The name of each table says which lab it comes from. The exact answers come from the science
package, like the estimates.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from radioactivity import exact_half_life, exact_surviving_fraction

# lab: (the parameter columns, the output, the exact answer as a function of the parameters)
LABS = {
    "decay": (
        ("decay.rate", "decay.n_atoms", "observation.t"),
        "surviving_fraction",
        lambda rate, n_atoms, t: exact_surviving_fraction(rate, t),
    ),
    "half_life": (
        ("decay.rate", "decay.n_atoms"),
        "half_life",
        lambda rate, n_atoms: exact_half_life(rate),
    ),
}

for i, path in enumerate(sys.argv[1:]):
    lab = Path(path).stem
    columns, output, exact = LABS[lab]

    estimates: dict[tuple[float, ...], list[float]] = defaultdict(list)
    with open(path, newline="") as table:
        for row in csv.DictReader(table):
            key = tuple(float(row[column]) for column in columns)
            estimates[key].append(float(row[f"outputs.{output}"]))

    if i:
        print()
    print(f"{lab}: {output}")
    names = [column.split(".")[-1] for column in columns]
    print(" ".join(f"{name:>8}" for name in [*names, "estimate", "+/-", "exact"]))
    for key, values in sorted(estimates.items()):
        mean = statistics.mean(values)
        sem = statistics.stdev(values) / math.sqrt(len(values))
        print(
            " ".join(f"{value:>8g}" for value in key),
            f"{mean:>8.4f} {sem:>8.4f} {exact(*key):>8.4f}",
        )
