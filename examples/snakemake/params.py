from pathlib import Path

from mechane import grid

# Absolute, so that every mechane command finds the results wherever it is run from.
OUTPUTS = Path("outputs").resolve()

# Two labs, each with its own ROOT (so its own manifest), seed, number of instances and sweep.
# The Snakefile's LABS lists the same names: the last part of each ROOT.
LABS = [
    {
        "ROOT": OUTPUTS / "decay",
        "LABORATORY": "decay:DecayLab",
        "PARAMS": {
            "seed": 1,
            "num_instances": 20,
            "decay": grid(rate=[0.5, 1.0, 2.0], n_atoms=[1_000, 10_000]),  # 6 experiments
            "observation": {"t": 1.0},
        },
    },
    {
        "ROOT": OUTPUTS / "half_life",
        "LABORATORY": "decay:HalfLifeLab",
        "PARAMS": {
            "seed": 2,
            "num_instances": 50,
            "decay": grid({"n_atoms": 2_000}, rate=[0.1, 1.0, 10.0]),  # 3 experiments
        },
    },
]
