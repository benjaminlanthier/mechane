"""Compare the Monte Carlo estimates in the exported table with the exact exp(-rate * t)."""

import csv
import math
import statistics
import sys
from collections import defaultdict

estimates: dict[tuple[float, int, float], list[float]] = defaultdict(list)
with open(sys.argv[1], newline="") as table:
    for row in csv.DictReader(table):
        key = (float(row["decay.rate"]), int(row["decay.n_atoms"]), float(row["observation.t"]))
        estimates[key].append(float(row["outputs.surviving_fraction"]))

print(f"{'rate':>5} {'n_atoms':>8} {'estimate':>10} {'+/-':>8} {'exact':>8}")
for (rate, n_atoms, t), values in sorted(estimates.items()):
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    print(f"{rate:>5} {n_atoms:>8} {mean:>10.4f} {sem:>8.4f} {math.exp(-rate * t):>8.4f}")
