# mechane

**M**anaging **E**xperiments, **C**onfigurations, **H**ashes **A**nd **N**umerical **E**xecutions.

`mechane` organizes numerical experiments: you describe a parameter sweep declaratively, and it
expands it into experiments, gives every result a content-addressed home on disk, seeds every
repetition reproducibly, and runs the work locally or as SLURM arrays through one small CLI.

It is domain-agnostic. It knows nothing about physics, machine learning or any particular solver:
you bring the parameters and a `run()` method.

> The name is the Greek *mēchanē* (μηχανή), the stage crane that lowered a god onto the scene
> to run the show, the origin of *deus ex machina*.

**Status:** alpha (`0.1.x`). The API may still change. Python ≥ 3.13. Dependencies: `click`, `numpy`.

## Contents

- [Why](#why)
- [Concepts](#concepts)
- [Install](#install)
- [Quickstart](#quickstart)
- [Defining sweeps](#defining-sweeps)
- [How results are organized](#how-results-are-organized)
- [Reproducibility and seeds](#reproducibility-and-seeds)
- [Result files and provenance](#result-files-and-provenance)
- [Aggregation](#aggregation)
- [Command-line reference](#command-line-reference)
- [Running on SLURM](#running-on-slurm)
- [Registering your laboratory](#registering-your-laboratory)
- [Python API at a glance](#python-api-at-a-glance)
- [Design notes and limitations](#design-notes-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Citing and license](#citing-and-license)

## Why

A typical research simulation grows the same scaffolding over and over: nested loops over
parameters, hand-named output folders, seeds that depend on loop order, bash scripts to slice
a job array, and a half-written results file after a pre-empted job. `mechane` is that
scaffolding, extracted and made generic:

- **Declarative sweeps.** `grid`, `zipped`, product (`*`) and concatenation (`+`), explicit
  rather than inferred from types.
- **Content-addressed result tree.** The directory of a result is derived from the hash of the
  parameters that affect it, so the same configuration always lands in the same place.
- **Reproducible seeds.** Each instance gets a seed derived from its identity, not from its
  position in a loop.
- **Manifest-driven jobs.** `mechane setup` writes a manifest once. Every later job rebuilds
  exactly the instance it needs from `(manifest, experiment_id, instance_id)`, so a requeue can
  never silently re-index your sweep.
- **Safe by default.** Atomic writes, skip-if-done, failure isolation inside batches,
  tracebacks saved next to the instance.
- **No lock-in.** Plain JSON on disk, plain directories, no database, no daemon.

## Concepts

| Concept | What it is | In code |
|---|---|---|
| **Section** | A named group of parameters (a frozen dataclass). | `Config` subclass |
| **Experiment** | One point of the sweep: an ordered set of sections, and the directory tree it owns. | `Experiment`, `ExperimentConfig` |
| **Instance** | One stochastic repetition of an experiment. Has its own seed and its own `results.json`. | `Instance` subclass, implements `run()` |
| **Laboratory** | Declares the sections and the instance class, expands the sweep, writes the manifest, aggregates. | `Laboratory` subclass |

```
Laboratory ── expands the sweep into ──▶ Experiment 0, 1, 2, …
Experiment ── repeated num_instances times ──▶ Instance 0, 1, 2, …
Instance   ── run() ──▶ results.json
```

## Install

From GitHub (pin a tag or commit for reproducibility, not `main`):

```bash
pip install "mechane @ git+https://github.com/benjaminlanthier/mechane@v0.1.0"
```

With [uv](https://docs.astral.sh/uv/), in your project's `pyproject.toml`:

```toml
[project]
dependencies = ["mechane"]

[tool.uv.sources]
mechane = { git = "https://github.com/benjaminlanthier/mechane", tag = "v0.1.0" }
```

From a local checkout (development):

```bash
git clone https://github.com/benjaminlanthier/mechane && cd mechane
pip install -e .
```

## Quickstart

A complete lab in two files: the mean squared displacement of random walks, swept over dimension
and walk length.

**`my_lab.py`**

```python
from dataclasses import dataclass

from mechane import Config, Instance, InstanceOutput, Laboratory
import numpy as np


@dataclass(frozen=True)
class Walk(Config):
    dim: int
    n_steps: int


@dataclass(frozen=True)
class Sampling(Config):
    n_walkers: int = 1000
    dtype: str = "int64"
    hash_exclude = ("dtype",)  # changes how we compute, not what we compute


class WalkInstance(Instance):
    def run(self):
        walk = self.experiment.config.walk
        sampling = self.experiment.config.sampling
        steps = self.rng.choice([-1, 1], size=(sampling.n_walkers, walk.n_steps, walk.dim))
        endpoints = steps.sum(axis=1).astype(sampling.dtype)
        return InstanceOutput(
            inputs={"first_endpoint": endpoints[0]},
            outputs={"msd": float((endpoints**2).sum(axis=1).mean())},
        )


class WalkLab(Laboratory):
    name = "walk"
    section_classes = {"walk": Walk, "sampling": Sampling}
    instance_class = WalkInstance
```

**`params.py`**

```python
from pathlib import Path

from mechane import grid

ROOT = Path("results/walks")
LABORATORY = "my_lab:WalkLab"
PARAMS = {
    "seed": 1,
    "num_instances": 20,
    "walk": grid(dim=[1, 2, 3], n_steps=[100, 400]),   # 6 experiments
    "sampling": {"n_walkers": 500},
}
```

**Run it**

```bash
export PYTHONPATH=$PWD        # the lab module must be importable (see Troubleshooting)

mechane setup --params-file params.py
# [Setup] 6 experiments x 20 instances. Manifest: results/walks/manifest.json

for e in $(seq 0 5); do
  mechane run-batch --manifest-path results/walks/manifest.json \
      --experiment-id $e --instance-ids $(seq -s, 0 19)
done

mechane status    --manifest-path results/walks/manifest.json
# experiment    0: 20 done, 0 failed, 0 missing
# ...
mechane aggregate --manifest-path results/walks/manifest.json
```

**Use the results**

```python
from mechane import load_laboratory

lab = load_laboratory("results/walks/manifest.json")
for exp in lab.experiments:
    walk = exp.config.walk
    msd = [r["outputs"]["msd"] for _, r in exp.iter_results()]
    print(f"dim={walk.dim} n_steps={walk.n_steps:>3}  <msd> = {sum(msd) / len(msd):7.2f}")
```

```
dim=1 n_steps=100  <msd> =   98.00
dim=1 n_steps=400  <msd> =  387.39
dim=2 n_steps=100  <msd> =  199.27
dim=2 n_steps=400  <msd> =  800.11
dim=3 n_steps=100  <msd> =  297.77
dim=3 n_steps=400  <msd> = 1196.95
```

(The theoretical value is `n_steps × dim`.)

## Defining sweeps

A section's entry in `PARAMS` is a plain dict (one point) or a sweep. **Nothing is inferred
from types: a list is a literal value unless you pass it as an axis.** That means vector-valued
parameters (per-qubit rates, coefficient lists) are never mistaken for sweeps.

| Expression | Meaning | Points |
|---|---|---|
| `{"a": 1, "b": [1, 2]}` | One point; `b` is the literal list `[1, 2]` | 1 |
| `grid(a=[1, 2], b=[3, 4, 5])` | Cartesian product of the axes | 6 |
| `zipped(a=[1, 2], b=[3, 4])` | Axes paired element-wise (equal lengths required) | 2 |
| `grid({"c": 0}, a=[1, 2])` | Fixed values first (positional dict), axes as keywords | 2 |
| `s1 * s2` | Product of two sweeps (keys must be disjoint) | `len(s1) · len(s2)` |
| `s1 + s2` | Concatenation | `len(s1) + len(s2)` |
| `{"name": "x", "params": zipped(n=[1, 2])}` | Sweeps may be nested inside dicts | 2 |

Example: a method and its options that must stay paired, crossed with an independent axis:

```python
"solver": zipped(method=["cg", "gmres"], method_opts=[{"tol": 1e-8}, {"restart": 30}])
          * grid(max_iter=[100, 1000]),
```

**Across sections** the sweep is the Cartesian product of every section's points. Experiments are
numbered in that order, with the **first** section in `sweep_order` varying slowest. By default
`sweep_order` is the section order; set it explicitly on your `Laboratory` if you need ids to stay
stable when porting an existing lab.

`Laboratory.from_params` rejects unknown top-level keys, so a typo such as `"modle"` fails loudly
instead of silently producing a one-point sweep.

## How results are organized

The directory of an experiment is built from its sections, in the order of `section_classes`:

```
<ROOT>/
├── manifest.json
├── aggregated_results.json            (after `mechane aggregate`)
└── <hash of Walk section>/
    └── <hash of Sampling section>/
        └── instances/
            ├── 00/
            │   ├── results.json
            │   └── job.err            (only if the instance failed)
            ├── 01/
            └── …
```

- **A section's hash** is the SHA-256 of its canonical JSON (first 24 hex characters),
  computed from its `to_dict()` minus the fields listed in `hash_exclude`. Fields that don't
  change the result (backend, dtype, caching flags, logging) belong in `hash_exclude`.
- **Every prefix of the path is a stage directory**, available as
  `experiment.stage_dir("<section name>")`. Experiments that share a prefix share that
  directory, which is the natural place for a cached artifact (a compiled object, a
  partitioning, a trained surrogate). **Put the sections whose results you want to share
  across the most experiments first.**
- **A section can shape its own path.** Override `path_parts()` to add readable components,
  for example `(self.family.value, self.hash())` gives `<ROOT>/surface-code/<hash>/…`.
- **Instance directories are zero-padded** to the width of `num_instances - 1` (`00`…`19`).
  If you later increase `num_instances` past a power of ten, `mechane setup` renames the
  existing directories accordingly.
- **Collisions are detected.** If two experiments resolve to the same directory (they differ only
  in `hash_exclude` fields, or the sweep contains a duplicate point) `setup()` raises instead of
  letting them overwrite each other.
- **Results key on configuration, not code.** If a bug fix changes results, add a version field
  (for example `code_version: int = 1`) to a section so the corrected run gets a new directory.

`Config` also accepts enums by value or by member (`"gauss"` or `Kind.GAUSS` are stored as the
member), and serializes enums, dataclasses, numpy scalars/arrays and paths to JSON for you.

## Reproducibility and seeds

Every instance exposes `self.seed` and a ready-made `self.rng` (`numpy.random.Generator`),
both deterministic, so rerunning an instance reproduces it exactly.

| `seed_scheme` | The seed is a hash of… | Use when |
|---|---|---|
| `"derived"` (default) | the lab seed, the experiment's content hash, and the instance id | New work. Adding sweep points or changing `num_instances` does **not** change existing seeds. |
| `"legacy"` | the instance id plus the full experiment config, including `experiment_id` and `num_instances` | You must keep reproducing result trees created by an earlier scheme. |

Set it on your `Experiment` subclass:

```python
class MyExperiment(Experiment):
    seed_scheme = "legacy"
```

Seeds are clamped to the signed 32-bit range, since some native libraries bind to a C `int`.

## Result files and provenance

Each `results.json` holds:

```jsonc
{
  "instance_id": 0,
  "experiment_id": 5,
  "seed": 1114423612,
  "inputs":  { ... },     // what was sampled or generated
  "outputs": { ... },     // what was measured
  "meta": {
    "hostname": "...", "platform": "...", "python": "3.12.3",
    "packages": {"mechane": "0.1.0", "numpy": "..."},   // see Laboratory.track_packages
    "slurm": {"SLURM_JOB_ID": "...", "SLURM_ARRAY_TASK_ID": "..."},
    "git": {"commit": "...", "dirty": false},           // null if not in a git repo
    "started_at": "2026-...", "wall_time_s": 0.018
  }
}
```

- `run()` returns an `InstanceOutput(inputs=…, outputs=…, meta=…)`, or a plain mapping, which is
  treated as `outputs`. Anything you put in `meta` is merged into the block above.
- Only an allow-list of environment variables is recorded (SLURM identifiers). The full
  environment is deliberately **not** saved: it bloats files and can leak credentials.
- Git information is read from the directory containing your `Laboratory` class, so the commit
  that is recorded is the commit of *your* code.
- Files are written atomically (temp file + rename), so a pre-empted job never leaves a truncated
  `results.json` that would later be mistaken for a finished instance.
- JSON is meant for scalars, short lists and metadata. For large arrays, write your own files
  (`.npy`, HDF5, …) into `self.dir` (the instance directory) and put the file name in `outputs`.

## Aggregation

`mechane aggregate` writes `aggregated_results.json` next to the manifest. By default each
experiment contributes:

```json
{
  "num_expected": 20, "num_done": 20,
  "failed": [], "missing": [],
  "instances": { "0": { ... }, "1": { ... } }
}
```

The only experiment-specific part is `Experiment.summarize(result)`, which reduces one result
record to whatever you want to keep (the default keeps `outputs`):

```python
class WalkExperiment(Experiment):
    def summarize(self, result):
        return {"msd": result["outputs"]["msd"], "seed": result["seed"]}
```

To use it, set `experiment_class = WalkExperiment` on your `Laboratory`. To change the layout of
the aggregate itself, override `aggregate_instance_results()`. `Experiment.iter_results()` yields
`(instance_id, record)` for the finished instances, and `Experiment.status()` returns
`{"done": […], "failed": […], "missing": […]}`.

## Command-line reference

`mechane <command>`, or the same commands under their own names if your project exposes them
(for example as `setup_lab`, `simulate`, …, via `[project.scripts]`).

| Command | Purpose | Options |
|---|---|---|
| `setup` | Expand the sweep, write the manifest, create all instance directories | `--params-file` |
| `run` | Run one instance | `--manifest-path`, `--experiment-id`, `--instance-id`, `--overwrite/--no-overwrite` |
| `run-batch` | Run several instances of one experiment in one process | `--manifest-path`, `--experiment-id`, `--instance-ids 0,1,2`, `--overwrite/--no-overwrite`, `--continue-on-error/--fail-fast` |
| `aggregate` | Collect results into `aggregated_results.json` | `--manifest-path` |
| `status` | Done / failed / missing counts per experiment | `--manifest-path` |

Behaviour worth knowing:

- Existing results are **skipped** unless `--overwrite` is given, which makes requeueing safe.
- In `run-batch`, one failing instance does not stop the others (`--continue-on-error`, the
  default). Each failure writes its traceback to that instance's `job.err`, and the command exits
  with status 1 once the batch is finished. A later successful run removes the stale `job.err`.
- Instance ids outside `0 … num_instances-1` are rejected with a clear error.
- `setup` is where the sweep is expanded. **Jobs never re-expand it**: they read the manifest.
  If you change `PARAMS`, run `setup` again.

### The params file

An ordinary Python file defining three names:

| Name | Meaning |
|---|---|
| `ROOT` | Directory that will hold the manifest and all results |
| `LABORATORY` | Your lab, as `"package.module:ClassName"`, an entry-point name, or the class itself |
| `PARAMS` | `{"seed": …, "num_instances": …, "<section>": <dict or sweep>, …}` (defaults: `seed=0`, `num_instances=1`) |

## Running on SLURM

`mechane` does not submit jobs for you; it gives your batch script a stable address for every
unit of work: `(manifest, experiment_id, instance_ids)`. A template that maps one flat
array index to a batch of instances of one experiment (adapt resources and paths):

```bash
#!/bin/bash
#SBATCH --job-name=walks
#SBATCH --array=0-11                 # = n_experiments * batches_per_experiment
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out

MANIFEST=results/walks/manifest.json
N_INSTANCES=20                       # num_instances in params.py
BATCH_SIZE=10
BATCHES_PER_EXP=$(( (N_INSTANCES + BATCH_SIZE - 1) / BATCH_SIZE ))

EXPERIMENT_ID=$(( SLURM_ARRAY_TASK_ID / BATCHES_PER_EXP ))
BATCH=$(( SLURM_ARRAY_TASK_ID % BATCHES_PER_EXP ))
START=$(( BATCH * BATCH_SIZE ))
END=$(( START + BATCH_SIZE - 1 ))
(( END >= N_INSTANCES )) && END=$(( N_INSTANCES - 1 ))   # last, partial batch

mechane run-batch --manifest-path "$MANIFEST" \
    --experiment-id "$EXPERIMENT_ID" \
    --instance-ids "$(seq -s, $START $END)"
```

For 6 experiments × 2 batches that is `--array=0-11`. Batching keeps the array under your
cluster's `MaxArraySize` and amortizes startup time, since the manifest is loaded once per batch.
The index arithmetic above was checked locally, but the script itself needs your site's modules,
environment activation and partitions.

Practical notes:

- Run `mechane setup` **before** submitting, so every directory already exists and array tasks
  never race on `mkdir`.
- Make sure the job's environment can import your lab module (activate the same virtualenv,
  or install your lab as a package).
- `mechane status` and `mechane aggregate` are cheap to run on the login node.
- Because finished instances are skipped, resubmitting the same array after a failure or a
  time-out only does the missing work.
- Any other scheduler or workflow engine can drive `mechane run` the same way, since each instance
  is addressed by the manifest path and two integers.

## Registering your laboratory

A manifest records the lab as `"module:Class"`, so any machine that can import your package can
resume a sweep. For short names (and for old manifests that only store a type name), register an
entry point in your project's `pyproject.toml`:

```toml
[project.entry-points."mechane.laboratories"]
walk = "my_lab:WalkLab"
```

then use `LABORATORY = "walk"` in your params file. A lab defined in `__main__` or inside a
function cannot be found by a job, and `setup` stops with an explanation instead of writing a
manifest nobody can load.

## Python API at a glance

```python
from mechane import (
    Config, Experiment, ExperimentConfig, Instance, InstanceOutput, Laboratory,
    grid, zipped, expand, load_laboratory, stable_hash, to_jsonable,
)
```

| Object | Things you set or override |
|---|---|
| `Config` | `hash_exclude`, `path_parts()`, `from_dict()` (inherits enum coercion; if you define `__post_init__`, call `super().__post_init__()`) |
| `Experiment` | `seed_scheme`, `summarize()`, `aggregate_instance_results()`; provides `stage_dir()`, `instances_dir`, `instance_dir(i)`, `instance_seed(i)`, `iter_results()`, `status()` |
| `Instance` | `run()` (required); provides `self.experiment`, `self.instance_id`, `self.seed`, `self.rng`, `self.dir` |
| `Laboratory` | `name`, `section_classes`, `instance_class` (required); `experiment_class`, `sweep_order`, `track_packages`, `default_manifest_path()`, `from_params()` (optional) |
| `load_laboratory(path)` | Rebuilds the right `Laboratory` subclass from a manifest alone |
| `stable_hash`, `to_jsonable` | The hashing and JSON helpers used throughout, exposed for your own artifacts |

`experiment.config.<section>` returns the section object (`experiment.config.walk.n_steps`).

## Design notes and limitations

**Guarantees**

- The same configuration always maps to the same directory and the same seeds.
- Sweeps are explicit. No parameter is turned into an axis because of its type.
- Writes are atomic, and finished work is never redone unless asked.
- No import side effects, global registries or hidden state: labs are found by an explicit
  reference.

**Not goals**

- Not a workflow/DAG engine: instances are independent. If your jobs have stage dependencies you
  can still drive `mechane run` from a workflow tool.
- No database, server or remote execution.
- Does not submit SLURM jobs itself.

**Known limitations**

- No built-in hook for building a stage artifact once and sharing it safely. You can use
  `stage_dir()` for the location, but if several instances may create the same artifact
  concurrently, write it to a temporary name and rename it (as `mechane.io.atomic_write_json` does
  for JSON).
- Results are JSON; large numerical arrays should live in separate files you manage.
- Every CLI invocation loads all experiment configs from the manifest. That is cheap for
  hundreds of experiments but adds startup cost for very large sweeps.
- Configuration hashes cover parameters, not code. Version your sections when logic changes.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: my_lab` when running `mechane …` | The console script does not put the current directory on `sys.path`. Install your lab as a package, or `export PYTHONPATH=$PWD` (and do the same inside your sbatch script). |
| `… is not importable by reference` at `setup` | The lab is defined in `__main__` or a local scope. Move it to an importable module. |
| `Experiments N and M map to the same directory` | They differ only in `hash_exclude` fields, or the sweep repeats a point. Remove the duplicate or make the differing field part of the hash. |
| `Unknown params keys […]` | A top-level key in `PARAMS` is not `seed`, `num_instances` or a section name. Check for a typo. |
| `zipped() axes must have equal lengths` | Paired axes need the same length; use `grid` (or `*`) for independent axes. |
| `instance_id … is out of range` | The id is outside `0 … num_instances-1`. In a batched sbatch script, cap the last batch (see the template). |
| An instance keeps being skipped after a code fix | Finished instances are skipped by design. Use `--overwrite`, or version your config so the result gets a new directory. |
| `Cannot serialize … to JSON` | A parameter or output is a type `to_jsonable` doesn't know. Convert it to a plain type before returning it. |

## Development

```bash
git clone https://github.com/benjaminlanthier/mechane && cd mechane
pip install -e .
pip install pytest
pytest
```

```
src/mechane/
  config.py        Config: hashing, enum coercion, path parts
  sweep.py         grid / zipped / product / chain
  experiment.py    ExperimentConfig, Experiment: paths, seeds, results, aggregation
  instance.py      Instance, InstanceOutput
  laboratory.py    Laboratory, manifest, load_laboratory
  registry.py      "module:Class" and entry-point resolution
  runner.py        run one instance (skip, save, traceback)
  provenance.py    allow-listed run metadata
  serialization.py to_jsonable, stable_hash
  io.py            atomic JSON writes
  cli.py           click commands
tests/
  toy.py           a minimal non-trivial lab, a template for new ones
  test_mechane.py
```

The test suite covers hashing, sweep semantics, seed stability, collision detection, padding
migration, and a full `setup → run → status → aggregate` workflow through the CLI, including
failure isolation.

## Citing and license

If `mechane` contributes to a publication, please cite it. A `CITATION.cff` and a Zenodo DOI can be
added once the first release is tagged.

License: *to be chosen.* Add a `LICENSE` file before sharing the code.