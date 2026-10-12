from pathlib import Path

from mechane import grid

# Absolute, so that every mechane command finds the results wherever it is run from.
ROOT = Path("results/decay").resolve()

LABORATORY = "decay:DecayLab"

PARAMS = {
    "seed": 1,
    "num_instances": 20,
    "decay": grid(rate=[0.5, 1.0, 2.0], n_atoms=[1_000, 10_000]),  # 6 experiments
    "observation": {"t": 1.0},
}
