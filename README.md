# mechane

**M**anaging **E**xperiments, **C**onfigurations, **H**ashes **A**nd **N**umerical **E**xecutions.

`mechane` organizes numerical experiments: you describe a parameter sweep declaratively, and it
expands it into experiments, gives every result a content-addressed home on disk, seeds every
repetition reproducibly, and runs the work locally or under any scheduler or workflow tool
through one small CLI. The same CLI splits the work into tasks, and lets you inspect, verify and
clean up a sweep afterwards. A project that uses it together with Snakemake is included as an
example.

It is domain-agnostic. It knows nothing about physics, machine learning or any particular solver:
you bring the parameters and a `run()` method.

> The name is the Greek *mēchanē* (μηχανή), the stage crane that lowered a god onto the scene
> to run the show, the origin of *deus ex machina*.

**Status:** alpha (`0.1.x`). The API may still change. Python ≥ 3.13. Dependencies: `click`, `numpy`.
Known rough edges are listed under [Design notes and limitations](#design-notes-and-limitations).

## Contents

- [Why](#why)
- [Concepts](#concepts)
- [Install](#install)
- [Quickstart](#quickstart)
- [Defining a laboratory](#defining-a-laboratory)
- [Defining sweeps](#defining-sweeps)
- [How results are organized](#how-results-are-organized)
- [Reproducibility and seeds](#reproducibility-and-seeds)
- [Result files and provenance](#result-files-and-provenance)
- [Aggregation](#aggregation)
- [Command-line reference](#command-line-reference)
- [Using mechane with Snakemake](#using-mechane-with-snakemake)
- [Inspecting and maintaining a sweep](#inspecting-and-maintaining-a-sweep)
- [Registering your laboratory](#registering-your-laboratory)
- [Python API at a glance](#python-api-at-a-glance)
- [Design notes and limitations](#design-notes-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Architecture](#architecture)
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
  position in a loop, and no two instances of an experiment can ever share one.
- **Manifest-driven jobs.** `mechane setup` writes a manifest once. Every later job rebuilds
  exactly the instance it needs from `(manifest, experiment_id, instance_id)`, so a job never
  re-expands your sweep. (Running `setup` again after editing the sweep *does* renumber the
  experiments: see [limitations](#design-notes-and-limitations).)
- **Safe by default.** Atomic writes, skip-if-done, failure isolation inside batches,
  tracebacks saved next to the instance.
- **Inspectable.** Commands to see what each experiment is, what failed and why, how long
  instances take, whether the results on disk are consistent, and whether a stored result still
  reproduces.
- **Typed.** Your own classes are linked through ordinary type annotations, so editors and type
  checkers see your parameters (`config.configs.walk.n_steps`).
- **No lock-in.** Plain JSON on disk, plain directories, no database, no daemon.

## Concepts

| Concept | What it is | In code |
|---|---|---|
| **Section** | A named group of parameters (a frozen dataclass). | `Config` subclass |
| **Configs** | The ordered collection of sections. Its field order is the directory nesting order. | `Configs` subclass |
| **Experiment** | One point of the sweep: its configs, and the directory tree it owns. | `ExperimentConfig` and `Experiment` subclasses |
| **Instance** | One stochastic repetition of an experiment. Has its own seed and its own `results.json`. | `Instance` subclass, implements `run()` |
| **Laboratory** | Points at your experiment and instance classes, expands the sweep, writes the manifest, aggregates. | `Laboratory` subclass |

```
Laboratory ── expands the sweep into ──▶ Experiment 0, 1, 2, …
Experiment ── repeated num_instances times ──▶ Instance 0, 1, 2, …
Instance   ── run() ──▶ results.json
```

Two small JSON files tie the steps together. The **manifest** (`manifest.json`) is written once by
`mechane setup`: it is the expanded sweep, and every later command rebuilds its laboratory from
it. A **plan** is written by `mechane plan`: it says which instances each task runs, for a
scheduler or a workflow tool to run one task at a time.

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
uv sync
```

Check the installation with `mechane --help`, which lists every command.

## Quickstart

A complete lab in two files: the mean squared displacement of random walks, swept over dimension
and walk length.

**`my_lab.py`**

```python
from dataclasses import dataclass

import numpy as np

from mechane import (
    Config,
    Configs,
    Experiment,
    ExperimentConfig,
    Instance,
    InstanceOutput,
    Laboratory,
)


@dataclass(frozen=True)
class Walk(Config):
    dim: int
    n_steps: int


@dataclass(frozen=True)
class Sampling(Config):
    n_walkers: int = 1000
    dtype: str = "int64"
    hash_exclude = ("dtype",)  # changes how we compute, not what we compute


@dataclass(frozen=True)
class WalkConfigs(Configs):  # field order = directory nesting order
    walk: Walk
    sampling: Sampling


@dataclass(frozen=True)
class WalkExperimentConfig(ExperimentConfig):
    configs: WalkConfigs


class WalkExperiment(Experiment):
    config: WalkExperimentConfig


class WalkInstance(Instance):
    experiment: WalkExperiment

    def run(self):
        configs = self.experiment.config.configs
        walk, sampling = configs.walk, configs.sampling
        steps = self.rng.choice([-1, 1], size=(sampling.n_walkers, walk.n_steps, walk.dim))
        endpoints = steps.sum(axis=1).astype(sampling.dtype)
        return InstanceOutput(
            inputs={"first_endpoint": endpoints[0]},
            outputs={"msd": float((endpoints**2).sum(axis=1).mean())},
        )


class WalkLab(Laboratory):
    name = "walk"
    experiment_class = WalkExperiment
    instance_class = WalkInstance
```

**`params.py`**

```python
from pathlib import Path

from mechane import grid

ROOT = Path("results/walks").resolve()  # absolute: every command finds the results from anywhere
LABORATORY = "my_lab:WalkLab"
PARAMS = {
    "seed": 1,
    "num_instances": 20,
    "walk": grid(dim=[1, 2, 3], n_steps=[100, 400]),  # 6 experiments
    "sampling": {"n_walkers": 500},
}
```

**Run it**

```bash
export PYTHONPATH=$PWD        # the lab module must be importable (see Troubleshooting)

mechane setup --params-file params.py
# [Setup] 6 experiments x 20 instances. Manifest: /path/to/project/results/walks/manifest.json

for e in $(seq 0 5); do
  mechane run-batch --manifest-path results/walks/manifest.json \
      --experiment-id $e --instance-ids $(seq -s, 0 19)
done

mechane status    --manifest-path results/walks/manifest.json
# experiment    0: 20 done, 0 failed, 0 missing
# ...
mechane aggregate --manifest-path results/walks/manifest.json

mechane list      --manifest-path results/walks/manifest.json   # what each experiment id means
# id  walk.dim  walk.n_steps
# 0   1         100
# 1   1         400
# ...
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
dim=1 n_steps=100  <msd> =  100.59
dim=1 n_steps=400  <msd> =  406.32
dim=2 n_steps=100  <msd> =  203.24
dim=2 n_steps=400  <msd> =  800.84
dim=3 n_steps=100  <msd> =  300.00
dim=3 n_steps=400  <msd> = 1214.44
```

(The theoretical value is `n_steps × dim`.)

To start a lab of your own, `mechane init lab NAME` writes a runnable skeleton like this one
(`NAME.py` and `params.py`) for you to edit.

## Defining a laboratory

A lab is five small classes. You never repeat a list of sections: each class points at the next
one through a type annotation, and `mechane` reads them. (`mechane init lab` writes all five.)

| You write | It points at the next class through |
|---|---|
| `WalkLab(Laboratory)` | `experiment_class = WalkExperiment` and `instance_class = WalkInstance` |
| `WalkExperiment(Experiment)` | the annotation `config: WalkExperimentConfig` |
| `WalkExperimentConfig(ExperimentConfig)` | the annotation `configs: WalkConfigs` |
| `WalkConfigs(Configs)` | one annotated field per section, e.g. `walk: Walk` (field order = directory nesting order) |
| `Walk(Config)`, `Sampling(Config)` | their own dataclass fields |

Rules:

- Define the classes at module level. The lab must be importable by reference anyway so that jobs
  can find it.
- Every field of your `Configs` subclass must be annotated with a `Config` subclass.
- Section names cannot be `experiment_id`, `num_instances`, `seed`, `zip` or `configs`: the first
  four are top-level keys of the parameter dict (and, for the first three, of the manifest), and
  `configs` is the attribute that holds the sections.
- Annotate `experiment: WalkExperiment` in your `Instance` subclass so that
  `self.experiment.config` is fully typed.

**Typed access.** Inside `run()`, read your parameters through
`self.experiment.config.configs.walk`: every step of that path is a class you wrote, so editors
and type checkers know the fields. The shorter `self.experiment.config.walk` also works at runtime
(and `exp.config.walk` outside instances), but type checkers only see a plain `Config`, so use it
for quick scripts rather than library code.

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
| `"zip": [("a", "b")]` | Sections `a` and `b` are paired, not crossed (see below) | `len` of either |

Example: a method and its options that must stay paired, crossed with an independent axis:

```python
"solver": zipped(method=["cg", "gmres"], method_opts=[{"tol": 1e-8}, {"restart": 30}])
          * grid(max_iter=[100, 1000]),
```

**Across configs** the sweep is the Cartesian product of every section's points. Experiments are
numbered in that order, with the **first** section in `sweep_order` varying slowest. By default
`sweep_order` is the section order; set it explicitly on your `Laboratory` if you need ids to stay
stable when porting an existing lab.

`Laboratory.from_configs` rejects unknown top-level keys, so a typo such as `"modle"` fails loudly
instead of silently producing a one-point sweep.

**Pairing whole sections.** Sections are combined as a Cartesian product, so two parameters in
different sections cannot be paired with `zipped`. To pair two sections' sweeps instead of
crossing them, list them in `zip`. Each section keeps its own sweep, and their points are matched
one to one:

```python
PARAMS = {
    "zip": [("a", "b")],  # paired, not crossed
    "a": grid(param_a=[3, 5, 7, 9]),
    "b": grid(param_b=[3, 5, 7, 9]),
}
```

This is 4 experiments (without `zip` it would be 16): point `i` of one section goes with point `i`
of the other, whole points at a time, so a section may sweep several fields. The sections of a
group must expand to the same number of points. A group counts as one dimension of the product,
placed where its first member is in `sweep_order`, so it can still be crossed with the other
sections or with another group (`"zip": [("a", "b"), ("c", "d")]`); a single group may be written
flat (`"zip": ("a", "b")`).

## How results are organized

The directory of an experiment is built from its configs, in the field order of your `Configs`
class:

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
  partitioning, a trained surrogate). **Put the configs whose results you want to share
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
member), and serializes enums, dataclasses, numpy scalars/arrays and paths to JSON for you. Enum
coercion applies to fields annotated as an `Enum` or as a union containing one (`Kind | None`);
inside a generic such as `list[Kind]` it does not work (the whole list is handed to the enum,
which raises `ValueError`), so keep such values as plain strings.

## Reproducibility and seeds

Every instance exposes `self.seed` and a ready-made `self.rng` (`numpy.random.default_rng(self.seed)`),
both deterministic, so rerunning an instance reproduces it exactly. The seed is also stored in the
instance's `results.json`. `mechane replay` uses exactly that to re-run an instance and check that
it reproduces its stored result.

| `seed_scheme` | How the seed is built | Use when |
|---|---|---|
| `"derived"` (default) | `mix31(base + instance_id)`, where `base` is a hash of the lab seed and the experiment's content hash | New work. |
| `"legacy"` | A hash of the instance id plus the full experiment config, including `experiment_id` and `num_instances` | You must keep reproducing result trees created by an earlier scheme. |

Properties of the `"derived"` scheme:

- **Unique within an experiment, by construction.** `mix31` is a bijection on 31 bits (a murmur3-style
  finalizer), so two instances of the same experiment can never share a seed. It also scrambles
  consecutive ids, so neighbouring seeds look unrelated, which matters for natively seeded code.
- **Stable.** The seed depends on the experiment's content and the lab seed only. Adding sweep
  points, reordering them, or increasing `num_instances` does **not** change existing seeds.
- **Native-friendly.** Seeds lie in `[0, 2**31)`, a non-negative C `int`.
- **Across experiments: independent, but not guaranteed distinct.** 31 bits are too few for that,
  and a seed must not depend on which other experiments you sweep. For a sweep of 100 experiments
  × 1000 instances, a plain 31-bit hash per instance would contain at least one repeated seed about
  89% of the time (simulated); this scheme does about 0.5% of the time, and never inside an experiment.

The `"legacy"` scheme keeps the original formula bit-for-bit. It guarantees nothing about
uniqueness, and because it hashes `experiment_id`, its seeds change if the experiments are
renumbered.

Set the scheme on your `Experiment` subclass:

```python
class WalkExperiment(Experiment):
    config: WalkExperimentConfig
    seed_scheme = "legacy"
```

Changing the scheme changes the seeds of any instance that has not run yet, so do not mix schemes
inside one result tree.

## Result files and provenance

Each `results.json` holds:

```jsonc
{
  "instance_id": 0,
  "experiment_id": 5,
  "seed": 1531082473,
  "inputs":  { ... },     // what was sampled or generated
  "outputs": { ... },     // what was measured
  "meta": {
    "hostname": "...", "platform": "...", "python": "3.13.8",
    "packages": {"mechane": "0.1.0", "numpy": "..."},   // see Laboratory.track_packages
    "slurm": {"SLURM_JOB_ID": "...", "SLURM_ARRAY_TASK_ID": "..."},
    "git": {"commit": "...", "dirty": false},           // null if not in a git repo; dirty = tracked
                                                        // files modified (untracked files are ignored)
    "started_at": "2026-...", "wall_time_s": 0.018
  }
}
```

- `run()` returns an `InstanceOutput(inputs=…, outputs=…, meta=…)`, or a plain mapping, which is
  treated as `outputs`. Anything you put in `meta` is merged into the block above.
- Only an allow-list of environment variables is recorded (SLURM identifiers). The full
  environment is deliberately **not** saved: it bloats files and can leak credentials.
- Git information is read from the directory containing your `Laboratory` class, so the commit
  that is recorded is the commit of *your* code. If that source is not inside a git repository (for
  example a lab installed as a package in a container image), `git` is `null`; `packages` still
  records the versions.
- Files are written atomically (temp file + rename), so a pre-empted job never leaves a truncated
  `results.json` that would later be mistaken for a finished instance.
- JSON is meant for scalars, short lists and metadata. For large arrays, write your own files
  (`.npy`, HDF5, …) into `self.dir` (the instance directory) and put the file name in `outputs`.
- The `experiment_id` stored in a result is the experiment's number *when the instance ran*. The
  results directory, not that number, is what identifies the experiment.

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

Running it again refreshes the entry of every experiment in the existing file (entries for
experiment ids the current manifest does not have are left as they are).

The file is keyed by experiment id and holds no parameters: use `mechane list` (or the manifest)
to see which parameters an id stands for, and `mechane export` for a table that already carries
them. Because ids are positional, an aggregate made before you edited the sweep and re-ran `setup`
no longer matches the numbering.

The only experiment-specific part is `Experiment.summarize(result)`, which reduces one result
record to whatever you want to keep (the default keeps `outputs`):

```python
class WalkExperiment(Experiment):
    config: WalkExperimentConfig

    def summarize(self, result):
        return {"msd": result["outputs"]["msd"], "seed": result["seed"]}
```

`WalkLab.experiment_class` already points at this class, so nothing else is needed. To change the
layout of the aggregate itself, override `aggregate_instance_results()`.
`Experiment.iter_results()` yields `(instance_id, record)` for the finished instances, and
`Experiment.status()` returns `{"done": […], "failed": […], "missing": […]}`.

## Command-line reference

`mechane <command>` (or `python -m mechane.cli <command>`). `mechane --help` lists the commands and
`mechane <command> --help` describes each one.

| Command | Purpose | Options |
|---|---|---|
| `setup` | Expand the sweep, write the manifest, create all instance directories | `--params-file` |
| `run` | Run one instance | `--manifest-path`, `--experiment-id`, `--instance-id`, `--overwrite/--no-overwrite` |
| `run-batch` | Run several instances of one experiment in one process | `--manifest-path`, `--experiment-id`, `--instance-ids 0,1,2`, `--overwrite/--no-overwrite`, `--continue-on-error/--fail-fast` |
| `aggregate` | Collect results into `aggregated_results.json` | `--manifest-path` |
| `status` | Done / failed / missing counts per experiment; `--check` exits 1 unless everything is done | `--manifest-path`, `--check` |
| `plan` | Split the work into tasks of a few instances each | `--manifest-path`, `--batch-size`, `--unfinished`, `--output PLAN`, `--format table\|json`, selection |
| `run-plan` | Run one task of a saved plan | `--plan-file`, `--task-id`, `--overwrite/--no-overwrite`, `--continue-on-error/--fail-fast` |
| `list` | Experiments with their parameters | `--manifest-path`, selection, `--status`, `--paths`, `--all`, `--json` |
| `errors` | Failed instances grouped by their error | `--manifest-path`, selection, `--examples`, `--traceback` |
| `doctor` | Consistency checks on the results on disk | `--manifest-path`, selection, `--strict` |
| `stats` | Wall-time statistics, optionally disk usage and a suggested time limit | `--manifest-path`, selection, `--disk`, `--batch-size`, `--margin` |
| `replay` | Re-run one instance and compare with its stored result | `--manifest-path`, `--experiment-id`, `--instance-id`, `--rtol`, `--scratch-dir`, `--pdb` |
| `shell` | A Python prompt with the laboratory loaded as `lab` | `--manifest-path` |
| `export` | Finished instances as CSV or JSON lines | `--manifest-path`, selection, `--format csv\|jsonl`, `--output`, `--inputs`, `--meta` |
| `clean` | Remove `job.err` files, results or stray directories | `--manifest-path`, selection, `--what errors\|results\|stray`, `--instance-ids`, `--all-experiments`, `--yes` |
| `init lab NAME` | Write a runnable laboratory module and `params.py` | `--section` (repeatable), `--directory`, `--force` |

**Selection** (`plan`, `list`, `errors`, `doctor`, `stats`, `export`, `clean`) narrows a command to
some experiments: `--experiment-id N` (repeatable) and `--where KEY=VALUE` (repeatable, ANDed),
where KEY is a dotted parameter as in the manifest (`walk.dim`, `solver.sigma`, or the top-level
`seed`; `num_instances` and `experiment_id` work too). VALUE is read as JSON when it parses (`3`,
`1e-3`, `true`) and as text otherwise (`gauss`). `mechane list --all` shows every parameter key.

**Exit status.**

- `0`: success.
- `1`: `run-batch` or `run-plan` when any instance failed (once the batch has finished; with
  `--fail-fast`, the first exception is re-raised); `status --check` when any instance is not
  done; `doctor` when it finds errors (warnings too with `--strict`); `replay` when the replay
  differs from the stored result; and an exception raised by an instance in `run` or `replay`.
- `2`: usage errors (a bad option, an unknown `--where` key, a task id outside the plan, a params
  file that names no laboratory).
- `status` without `--check`, `aggregate`, `list`, `errors`, `stats` and `export` exit `0` whenever
  they run, even if work is missing or failed. Use `status --check` (is everything done?) and
  `doctor` (is what is there consistent?) to turn that into a check, for instance as a gate in a
  workflow.

Behaviour worth knowing:

- Existing results are **skipped** unless `--overwrite` is given, which makes requeueing safe.
- In `run-batch` and `run-plan`, one failing instance does not stop the others
  (`--continue-on-error`, the default). Each failure writes its traceback to that instance's
  `job.err`, and the command exits with status 1 once the batch is finished. A later successful run
  removes the stale `job.err`.
- **`run` and `run-batch` do not check that an instance id is below `num_instances`.** An id past
  the end runs and writes a result that `status`, `aggregate` and `export` never look at.
  `plan` and `run-plan` only use valid ids, and `replay` rejects invalid ones. A hand-written
  batching script must cap its last batch (see [Other schedulers and plain scripts](#other-schedulers-and-plain-scripts)); `doctor`
  reports the leftovers of one that did not, and `clean --what stray` removes them.
- An unknown experiment id fails with the list of valid ids: as a `KeyError` in `run` and
  `run-batch`, as a usage error in `replay` and in the commands that take a selection.
- `setup` is where the sweep is expanded. **Jobs never re-expand it**: they read the manifest.
  If you change `PARAMS`, run `setup` again, which renumbers the experiments (see
  [limitations](#design-notes-and-limitations)).
- `doctor` reads every results file and reports, without changing anything: records whose seed or
  instance id does not match the configuration, unreadable results, a stale `job.err` next to a
  result, instance directories past `num_instances` (which `status` and `aggregate` never look
  at), results produced by several code versions (by git commit when recorded, otherwise by
  package versions), a relative root, and directories the manifest does not use. It exits with
  status 1 on errors (and, with `--strict`, on warnings).
- `replay` re-runs one instance with its original seed in a scratch directory and compares
  `seed`, `inputs` and `outputs` with the stored result (exactly, or within `--rtol`). The record
  goes to standard output, the verdict to standard error, and the exit status is 1 if they differ.
  Nothing is written into the results tree, files the instance writes included.
- `clean` only lists what it would remove unless `--yes` is given. `--what errors` turns failed
  instances into missing ones, `--what results` makes finished instances run again (and needs a
  narrowing selection, or `--all-experiments`), and `--what stray` removes instance directories
  past `num_instances`.
- `stats --disk` walks every instance directory, which is slow on large trees and on parallel
  filesystems.

### The params file

An ordinary Python file defining three names:

| Name | Meaning |
|---|---|
| `ROOT` | Directory that will hold the manifest and all results |
| `LABORATORY` | Your lab, as `"package.module:ClassName"`, an entry-point name, or the class itself |
| `PARAMS` | `{"seed": …, "num_instances": …, "<section>": <dict or sweep>, …}` (defaults: `seed=0`, `num_instances=1`) |

`ROOT` is stored in the manifest exactly as you write it. A relative `ROOT` is read relative to
the current directory of every later command, so `status` run from another directory finds nothing
and `aggregate` reports no finished instances. Write `Path("results").resolve()`, as in the
Quickstart; `doctor` warns about a relative root.

## Using mechane with Snakemake

`mechane` does not submit jobs and knows nothing about schedulers. It gives schedulers and
workflow tools stable commands to call: `plan` and `run-plan` split the work into tasks and run one
task at a time, `status --check` and `doctor` say whether the results are complete and consistent,
and every command reports success or failure through its exit status. Snakemake is a good partner
when your project is a *pipeline* (simulate, check, export, analyse, plot): it orders the stages,
runs independent jobs in parallel and, through its profiles, talks to a cluster scheduler. The
settings that depend on the machine (account, partition, memory, run time, containers) then live
in your project's Snakemake profile, not in mechane.

| mechane | Snakemake |
|---|---|
| What the experiments are, where each result goes, which seed each instance uses | The order of the stages and which jobs can run in parallel |
| Which instances are done: finished ones are skipped, failures leave a traceback | Resources, retries, and submitting jobs to a cluster |
| `status --check`, `doctor`, `errors`, `replay`: is it complete, consistent, reproducible? | Re-running only what is out of date |

A complete, runnable project is in [`examples/snakemake/`](examples/snakemake/): a small Monte
Carlo lab, its sweep, a Snakefile and an analysis script. Abridged, its core is four rules:

```python
rule setup:                       # expand the sweep, write the manifest
    input: "params.py", "decay.py"
    output: MANIFEST
    shell: "mechane setup --params-file {input[0]}"

checkpoint make_plan:             # split the work into tasks
    input: MANIFEST
    output: PLAN
    shell: "mechane plan --manifest-path {input} --batch-size {BATCH_SIZE} --output {output} > /dev/null"

rule run_task:                    # one Snakemake job per mechane task
    input: PLAN
    output: touch("done/task_{t}")
    shell: "mechane run-plan --plan-file {input} --task-id {wildcards.t}"

rule complete:                    # a gate: fails unless every instance is done
    input: all_tasks
    output: touch("done/complete")
    shell: "mechane status --manifest-path {MANIFEST} --check"
```

The example then exports the results as a table (`mechane export`) and runs the analysis on it.
Things to know:

- **Give Snakemake tasks, not instances.** One Snakemake job should run a mechane *task* of
  several instances. On one machine, 12 instances run as 12 separate `mechane run` processes took
  1.95 s, against 0.18 s in a single `run-batch`: about 160 ms of Python start-up per process,
  which matters when instances are short. Snakemake built a 5,000-job DAG in about 4 s in a dry
  run, so the number of jobs is not what hurts: the cost per job is.
- **mechane is the source of truth for "done".** Snakemake's `done/task_N` files only mean "this
  task ran without failures". If you delete results or run `mechane clean` behind Snakemake's
  back, it does not notice, and `snakemake` says "Nothing to be done". Tell it with
  `snakemake --forcerun run_task`: that is cheap, because mechane skips finished instances. The
  `complete` gate then confirms that everything really is done.
- **Failures stop the pipeline.** `run-plan` exits 1 when any instance of its task failed, so the
  job fails and nothing downstream runs. Look with `mechane errors` and `mechane doctor`, fix the
  code, and run `snakemake` again: the failed tasks run again.
- **Changing the sweep.** Editing `params.py` re-runs `setup` and `make_plan`. Experiment ids are
  positional, so adding a point in the middle renumbers them (see
  [limitations](#design-notes-and-limitations)): let Snakemake regenerate the plan, and do not
  reuse a plan file that was saved by hand.
- **Choosing the batch size and the time limit.** `mechane stats --batch-size N` suggests a time
  limit for tasks of N instances, as a number of minutes (the unit of Snakemake's standard
  `runtime` resource).
- **Clusters.** The example was run with Snakemake's local executor. Running it on a cluster is
  done with Snakemake's executor plugins and a profile for your cluster, which this repository
  does not provide and has not tested.

### Other schedulers and plain scripts

Without a workflow tool, `mechane plan --output plan.json` writes the tasks. Task ids run from 0
to the number of tasks minus one (the first line of the table gives the count), so a shell loop or
any scheduler's array facility can run them:

```
$ mechane plan --manifest-path results/walks/manifest.json --batch-size 10 --output plan.json
120 instance(s) in 12 task(s) of up to 10; run task N (0 to 11) with `mechane run-plan`

task  experiment  count  instances
0     0           10     0-9
1     0           10     10-19
2     1           10     0-9
...
```

```bash
for t in $(seq 0 11); do mechane run-plan --plan-file plan.json --task-id $t; done
```

In an array job the array index is the task id (for example
`mechane run-plan --plan-file plan.json --task-id "$SLURM_ARRAY_TASK_ID"`). The plan file is the
contract between planning and running time, so task `N` always means the same instances, even
while other tasks finish.

- `mechane plan --unfinished` plans only the instances that are missing or failed, so a new plan
  covers exactly what is left. `--experiment-id` and `--where` plan some experiments only.
- A plan groups its tasks by experiment and writes each task's instances as a range, one batch
  per line, so its size follows the number of tasks and not of instances: a million instances in
  tasks of 1,000 is about 25 KB. Task `N` is the Nth batch in file order. A plan for two
  experiments of 20 instances, in tasks of 10:

  ```json
  {
    "version": 3,
    "manifest_path": "/path/to/results/walks/manifest.json",
    "batch_size": 10,
    "num_tasks": 4,
    "tasks": [
      {"experiment_id": 0, "batches": [
        "0-9",
        "10-19"
      ]},
      {"experiment_id": 1, "batches": [
        "0-9",
        "10-19"
      ]}
    ]
  }
  ```

  `num_tasks` is the total, which is what a script needs to count the tasks: `tasks` has one entry
  per experiment, so `len(plan["tasks"])` is not the number of tasks. A Snakefile written for an
  earlier version that counts tasks that way must read `plan["num_tasks"]` instead. `run-plan`
  reads only its own task, and plan files from earlier versions (every instance id listed, or one
  range per task) are still read.
- A plan belongs to its manifest. It records experiment ids, and `run-plan` does not check that
  the manifest still numbers its experiments the same way. After editing `PARAMS` and re-running
  `setup`, make a new plan.
- If you write the batching yourself with `run-batch`, cap the last batch at `num_instances - 1`:
  `run-batch` does not check ids, so a partial last batch would otherwise run instances past the
  end.
- Run `mechane setup` **before** starting tasks, so every directory already exists and parallel
  tasks never race on `mkdir`.
- Make sure the job's environment can import your lab module (activate the same virtualenv,
  install your lab as a package, or `export PYTHONPATH=…` in the job).
- `mechane status`, `mechane aggregate` and `mechane list` are cheap to run on a login node (see
  the scale note under [limitations](#design-notes-and-limitations)).
- From Python, `mechane.core.runner.run_batch` does what `mechane run-batch` does, without the
  command line (see [Python API at a glance](#python-api-at-a-glance)).

## Inspecting and maintaining a sweep

The examples below use the Quickstart lab, except that its `run()` raises `NotImplementedError`
for `dim=3` so that there is something to look at. They share `MANIFEST=results/walks/manifest.json`,
and timings will differ on your machine.

**What is each experiment, and how far along?** `list` shows the parameters that tell experiments
apart (`--all` shows every parameter, `--paths` the directories, `--json` machine-readable output):

```
$ mechane list --manifest-path $MANIFEST --status
id  walk.dim  walk.n_steps  done  failed  missing
0   1         100           20    0       0
1   1         400           20    0       0
2   2         100           20    0       0
3   2         400           20    0       0
4   3         100           0     20      0
5   3         400           0     20      0
```

**Why did instances fail?** `errors` groups failures by their error, the most frequent first
(`--examples N` lists more instances per error, `--traceback` prints the full traceback):

```
$ mechane errors --manifest-path $MANIFEST --examples 1
40 failed instance(s), 1 distinct error(s)

   40x  NotImplementedError: dim=3 is not implemented yet
        experiment 4 instance 0: /path/to/project/results/walks/b286…/5901…/instances/00/job.err
        ... and 39 more
```

After fixing the code, `mechane plan --manifest-path $MANIFEST --batch-size 10 --unfinished` plans
the failed instances again, and nothing else (`40 instance(s) in 4 task(s) of up to 10`).

**How long do instances take?** `stats` reports wall times from the results themselves, and with
`--batch-size` suggests a time limit in minutes (p95 of the slowest experiment × batch size ×
`--margin`, a heuristic that cannot know about a slower case that has not run yet):

```
$ mechane stats --manifest-path $MANIFEST --batch-size 10
experiment  done   median  p95   max   total
0           20/20  1ms     5ms   5ms   21ms
1           20/20  2ms     7ms   8ms   50ms
2           20/20  2ms     7ms   7ms   46ms
3           20/20  6ms     12ms  15ms  143ms
4           0/20   -       -     -     0ms
5           0/20   -       -     -     0ms

Total instance time: 261ms
Suggested time limit for 10 instance(s) per task: 1 min (slowest p95 x 10 x 1.25; a heuristic)
```

**Is anything inconsistent?** `doctor` checks the results on disk and changes nothing. Here a
batching script ran instance 20 of an experiment that has 20 (`0..19`):

```
$ mechane doctor --manifest-path $MANIFEST
WARNING experiment 0: instance directories 20 are outside 0..19; status and aggregate ignore them

0 error(s), 1 warning(s)
```

`clean` removes such debris. It lists what it would delete unless you add `--yes`:

```
$ mechane clean --manifest-path $MANIFEST --what stray
  experiment 0 instance 20: /path/to/project/results/walks/516c…/5901…/instances/20
Would remove 1 item(s). Re-run with --yes to delete them.
```

**Does a stored result reproduce?** `replay` re-runs one instance from its seed in a scratch
directory and compares it with what is stored. The replayed record goes to standard output, the
verdict to standard error, and the exit status is 1 if they differ (use `--rtol` to allow numerical
noise, `--pdb` to debug an instance that raises):

```
$ mechane replay --manifest-path $MANIFEST --experiment-id 0 --instance-id 3
{
  "seed": 860930082,
  "inputs": {
    "first_endpoint": [
      -6
    ]
  },
  "outputs": {
    "msd": 94.92
  }
}
Reproduced: the replay matches the stored result.
```

**Getting results out.** `export` writes one row per finished instance, with the experiment's
parameters and the outputs as columns (`--format jsonl`, `--output FILE`, `--inputs`, `--meta`):

```
$ mechane export --manifest-path $MANIFEST --where walk.dim=1 --where walk.n_steps=100
experiment_id,instance_id,seed,walk.dim,walk.n_steps,sampling.n_walkers,sampling.dtype,outputs.msd
0,0,1450271879,1,100,500,int64,95.88
0,1,1442899379,1,100,500,int64,102.576
...
```

`mechane shell --manifest-path $MANIFEST` opens a Python prompt with the laboratory loaded as `lab`
(plus `experiments`, `experiment` and `build_instance`).

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
    Config,
    Configs,
    Experiment,
    ExperimentConfig,
    Instance,
    InstanceOutput,
    Laboratory,
    Sweep,
    grid,
    zipped,
    expand,
    load_laboratory,
    stable_hash,
    to_jsonable,
)
```

| Object | Things you set or override |
|---|---|
| `Config` | `hash_exclude`, `path_parts()`, `from_dict()` (inherits enum coercion; if you define `__post_init__`, call `super().__post_init__()`) |
| `Configs` | One field per section, annotated with its `Config` class. Provides `names()`, `as_dict()`, `section_classes()` |
| `ExperimentConfig` | Annotate `configs` with your `Configs` class. Holds `experiment_id`, `num_instances`, `seed`; provides `to_dict()`, `content_hash()` |
| `Experiment` | Annotate `config` with your `ExperimentConfig` class. Set `seed_scheme`, override `summarize()`, `aggregate_instance_results()`; provides `stage_dir()`, `instances_dir`, `instance_dir(i)`, `instance_seed(i)`, `present_instance_ids()`, `iter_results()`, `status()` |
| `Instance` | `run()` (required); provides `self.experiment`, `self.instance_id`, `self.seed`, `self.rng`, `self.dir` |
| `Laboratory` | `name`, `experiment_class`, `instance_class` (required); `sweep_order`, `track_packages`, `default_manifest_path()` (optional); provides `from_configs()`, `from_manifest()`, `setup()`, `aggregate_results()`, `section_classes()`, `experiment(i)`, `build_instance(e, i)` |
| `load_laboratory(path)` | Rebuilds the right `Laboratory` subclass from a manifest alone |
| `stable_hash`, `to_jsonable` | The hashing and JSON helpers used throughout, exposed for your own artifacts |

**Scripting with the internals.** Everything the commands do is also available as plain functions
in `mechane.core`, which return data and never print. They are importable but not part of the
stable top-level API (only the names in `mechane.__all__` are), so they may move:

```python
from mechane import load_laboratory
from mechane.core.doctor import diagnose
from mechane.core.runner import run_batch
from mechane.core.selection import select_experiments

lab = load_laboratory("results/walks/manifest.json")
dim2 = select_experiments(lab, where=["walk.dim=2"])  # experiments 2 and 3
result = run_batch(lab, 0, range(20))  # BatchResult(instance_ids=..., failed=())
for finding in diagnose(lab, dim2):  # the checks behind `mechane doctor`
    print(finding.severity, finding.message)
```

`run_batch` and `run_instance` report progress through an optional `listener` callback (see
[Architecture](#architecture)); the other modules are listed in the tree under
[Development](#development). Internal helpers such as `mechane.utils.io.atomic_write_json` are
likewise importable but not part of the top-level API.

## Design notes and limitations

**Guarantees**

- The same configuration always maps to the same directory and the same seeds.
- No two instances of an experiment share a seed.
- Sweeps are explicit. No parameter is turned into an axis because of its type.
- Writes are atomic, and finished work is never redone unless asked.
- No import side effects, global registries or hidden state: labs are found by an explicit
  reference.
- `replay` and `clean` are conservative: `replay` never writes into the results tree, and `clean`
  deletes nothing without `--yes`.

**Not goals**

- Not a workflow/DAG engine: instances are independent. If your jobs have stage dependencies you
  can still drive `mechane run` or `mechane run-plan` from a workflow tool.
- No database, server or remote execution.
- Does not submit jobs or talk to a scheduler. It gives schedulers and workflow tools stable
  commands to call; see [Using mechane with Snakemake](#using-mechane-with-snakemake).

**Known limitations**

- **Experiment ids are positional.** Running `setup` again after changing `PARAMS` renumbers the
  experiments (a new point in the middle of a sweep shifts every id after it), and `setup` prints
  no warning when it replaces a manifest. Results on disk are unaffected, because directories are
  content-addressed, and `"derived"` seeds do not change. But anything that stores an id does: jobs
  or scripts that were given `--experiment-id`, saved plan files (`run-plan` would run whichever
  experiment now has that id), and `aggregated_results.json`. After changing the sweep, make a new
  plan, aggregate again, and check the numbering with `mechane list`.
- **A relative `ROOT` is stored as written.** See [The params file](#the-params-file).
- **`run` and `run-batch` accept instance ids past `num_instances`.** See
  [Command-line reference](#command-line-reference); `doctor` and `clean --what stray` deal with
  the leftovers.
- **Input validation is uneven.** A non-integer in `--instance-ids` (`1,a`) and a params file
  without `ROOT` or `PARAMS` end in a Python traceback (`ValueError`, `AttributeError`) instead of a
  usage message. A missing `LABORATORY` is reported properly.
- **Enum fields inside generics are not coerced** (see
  [How results are organized](#how-results-are-organized)).
- **No `--version` option.** `pip show mechane` reports the installed version, and every result
  records it under `meta.packages`.
- No built-in hook for building a stage artifact once and sharing it safely. You can use
  `stage_dir()` for the location, but if several instances may create the same artifact
  concurrently, write it to a temporary name and rename it (as `mechane.utils.io.atomic_write_json`
  does for JSON).
- Seeds of *different* experiments can coincide with small probability (see
  [Reproducibility and seeds](#reproducibility-and-seeds)).
- Results are JSON; large numerical arrays should live in separate files you manage.
- Every CLI invocation loads all experiment configs from the manifest. That is cheap for
  hundreds of experiments but adds startup cost for very large sweeps. The commands that look at
  instances list each experiment's `instances/` directory once and only examine the instances that
  have a directory, so an experiment of a million instances that has barely started costs
  well under a second. Reading finished results costs one file each: with 20,000 results on a local
  disk, `status` took 0.3 s, `aggregate` and `export` 0.6 s, and `doctor` 1 s. None of this has
  been measured on a parallel filesystem, where it will be slower.
- **One instance is one directory and one `results.json`**, so a million instances is about two
  million files per experiment, which many clusters limit. If your instances are tiny, make each
  one do more work (draw many samples inside `run()`) instead of running millions of them.
- Configuration hashes cover parameters, not code. Version your configs when logic changes;
  `doctor` reports results that were produced by more than one recorded code version.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: my_lab` when running `mechane …` | The console script does not put the current directory on `sys.path`. Install your lab as a package, or `export PYTHONPATH=$PWD` (and do the same in your job script; the Snakemake example sets it with `shell.prefix`). |
| `… is not importable by reference` at `setup` | The lab is defined in `__main__` or a local scope. Move it to an importable module. |
| `Experiments N and M map to the same directory` | They differ only in `hash_exclude` fields, or the sweep repeats a point. Remove the duplicate or make the differing field part of the hash. |
| `Unknown configs keys […]` | A top-level key in `PARAMS` is not `seed`, `num_instances` or a section name. Check for a typo. |
| `zipped() axes must have equal lengths` | Paired axes need the same length; use `grid` (or `*`) for independent axes. |
| `ValueError: zipped sections must expand to the same number of points, got {…}` | The sections listed together in `zip` sweep different numbers of points. Their sweeps must have the same length. |
| `instance_id N is out of range for experiment E` | Raised by `replay`. `run` and `run-batch` do **not** check, so a batch that overshoots the end silently creates extra results: `mechane doctor` reports them and `mechane clean --what stray --yes` removes them. Cap the last batch in a hand-written script, or use `plan` and `run-plan`. |
| `KeyError: No experiment N; valid ids: 0..K` (or `Error: No experiment N …` from commands with a selection) | The experiment id is not in the manifest. After changing `PARAMS`, run `mechane setup` again, and see the renumbering note under limitations. |
| `status` shows everything missing, or `aggregate` reports no finished instances, although results exist | The manifest's `ROOT` is relative and you are in another directory. `cd` back to where you ran `setup`, or use an absolute `ROOT` and run `setup` again. `mechane doctor` warns about a relative root. |
| `TypeError: … declares no sections` | The lab still uses the base classes. Subclass `Configs`, annotate `ExperimentConfig.configs` with it, annotate `Experiment.config` with your `ExperimentConfig` subclass, and set `experiment_class` on the lab. |
| `TypeError: … must be annotated with a Config subclass` | A field of your `Configs` class has a type that is not a `Config` subclass. |
| `ValueError: Config names […] are reserved` | A section is named `experiment_id`, `num_instances`, `seed`, `zip` or `configs`. Rename it. |
| `ValueError: […] is not a valid Kind` (or your enum's name) when building a `Config` | An enum inside a generic (`list[Kind]`) is not coerced. Keep the values as plain strings. |
| `NameError: name '…' is not defined` when the lab is first used | Your classes are defined inside a function and the file uses `from __future__ import annotations`. Define them at module level. |
| A type checker reports `Object of type Config has no attribute …` | You used the `config.<section>` shortcut. Use the typed path `config.configs.<section>`. |
| An instance keeps being skipped after a code fix | Finished instances are skipped by design. Use `--overwrite`, remove the results with `mechane clean --what results --experiment-id N --yes`, or version your config so the result gets a new directory. |
| `ValueError: invalid literal for int()` from `run-batch` | `--instance-ids` takes comma-separated integers (`0,1,2`). `plan` and `run-plan` avoid writing ids by hand. |
| `AttributeError: module 'external_params' has no attribute 'ROOT'` | The params file must define `ROOT`, `LABORATORY` and `PARAMS`. |
| `Error: unknown parameter 'x'; available: […]` | A `--where` key must be a dotted parameter as in the manifest (or `seed`, `num_instances`, `experiment_id`). `mechane list --all` shows the parameters. |
| `Error: task id N is out of range: the plan has M task(s)` | The scheduler ran more tasks than the plan has. Task ids go from 0 to the number of tasks minus one, which the first line of `mechane plan` shows; use the plan file the job was given. |
| `Error: refusing to remove the results of every experiment` | `clean --what results` needs `--experiment-id` or `--where`, or `--all-experiments`. |
| `NOT reproduced: the replay differs from the stored result` | The code changed since the instance ran (`mechane doctor` shows mixed code versions), the computation is not deterministic, or the difference is numerical noise that `--rtol` would absorb. |
| Snakemake says `Nothing to be done` although results were deleted or cleaned | Its `done/` files do not know about that. Run `snakemake --forcerun run_task` (cheap: finished instances are skipped). |
| `Cannot serialize … to JSON` | A parameter or output is a type `to_jsonable` doesn't know. Convert it to a plain type before returning it. |

## Development

```bash
git clone https://github.com/benjaminlanthier/mechane && cd mechane
uv sync                              # creates .venv with the dev tools
uv run pytest
uv run pre-commit install            # lint, format and type-check on every commit; tests on every push
uv run pre-commit run --all-files    # the same checks, on demand
```

`pyproject.toml` configures pytest to run with `pytest-xdist` (one worker) and `pytest-cov` (line
and branch coverage of `src`), and to turn warnings into errors (except `UserWarning`s); the suite currently covers 100%
of lines and branches. Lint, format and type checks are `ruff` and `ty`.

CI runs the lint and type checks, the tests on Linux and macOS, and builds the wheel and
smoke-tests the `mechane` command in a clean environment.

Releases are automatic. After CI succeeds on `main`, the release workflow tags and publishes a
GitHub release: if the version in `pyproject.toml` has no tag yet it is released as it is,
otherwise the patch version is bumped (put `#minor` or `#major` in a commit message since the last
release for a bigger bump). A merge that changes nothing in `src/`, `pyproject.toml` or `uv.lock`
does not release, and nothing is uploaded to PyPI.

```
src/mechane/
  __init__.py          the public API (re-exports)
  core/                the domain model and the work it does (never imports click)
    config.py          Config, Configs, ExperimentConfig: parameters, hashing, enum coercion
    experiment.py      Experiment: directory layout, seeds, results, status, aggregation
    instance.py        Instance, InstanceOutput
    laboratory.py      Laboratory: sweep expansion, manifest content, setup, aggregation
    manifest.py        read_manifest, load_laboratory
    registry.py        "module:Class" and entry-point resolution of a Laboratory
    params.py          load_params_file: ROOT / LABORATORY / PARAMS
    runner.py          run_instance, run_batch: skip, execute, save, record failures
    selection.py       --experiment-id / --where: choosing experiments
    plan.py            plans: which instances each task runs (build, save, read)
    errors.py          failed instances grouped by error
    doctor.py          consistency checks on the results on disk
    stats.py           wall-time statistics, disk usage, suggested time limit
    listing.py         experiments described by their parameters
    replay.py          re-run one instance in a scratch directory and compare
    shell.py           the namespace of `mechane shell`
    export.py          results as a flat table
    clean.py           find and remove job.err files, results, stray directories
    scaffold.py        text of a new laboratory
  utils/               leaf helpers: standard library and numpy only
    sweep.py           grid / zipped / product / chain
    serialization.py   to_jsonable, stable_hash, flatten
    ranges.py          compact_ranges / parse_ranges: `0-2,5,7-8`
    seeding.py         mix31 and the derived / legacy seed schemes
    provenance.py      allow-listed run metadata
    io.py              atomic JSON and text writes
  cli/                 click commands: thin wrappers over core
    __init__.py        the `mechane` command group
    _options.py        click options shared by several commands, including the selection
    _output.py         the wording of what the commands print, and shared formatting
    commands/          one module per command (setup, run, run_batch, aggregate, status, plan,
                       run_plan, errors, doctor, stats, list_experiments, replay, shell, export,
                       clean, init)
examples/
  snakemake/           a project that uses mechane with Snakemake: lab, sweep, Snakefile, analysis
tests/
  toy.py               a minimal non-trivial lab, a template for new ones
  conftest.py          fixtures: `register_laboratory`, a set-up two-experiment `lab`, helpers
  test_mechane.py      hashing, sweeps, seeds, and the full CLI workflow
  test_core.py         config, experiment, instance, laboratory, registry, manifests
  test_runner.py       core.runner: running instances and batches
  test_params.py       core.params: reading a params file
  test_seeding.py      utils.seeding: the seed schemes as pure functions
  test_utils.py        serialization, sweeps, ranges, provenance
  test_cli.py          the command line: entry point, params files, run-batch, wording, formatting
  test_cli_commands.py the newer commands through the command line
  test_scan.py         finding instances on disk from one directory listing, as before but faster
  test_zip.py          `zip`: sections paired element by element instead of crossed
  test_selection.py, test_plan.py, test_errors.py, test_doctor.py, test_stats.py,
  test_listing.py, test_replay.py, test_export.py, test_clean.py, test_scaffold.py
                       the logic of each newer command, in `core`
  test_examples.py     the Snakemake example keeps working (lab, analysis, mechane calls)
  test_architecture.py the layering rules below, enforced on the imports
```

The test suite covers hashing, sweep semantics, seed uniqueness and stability (with pinned values),
collision detection, padding migration, running and batching instances (including failure
isolation), the logic of every command, every command through the CLI, and the layering rules.
Tests that touch the results tree use real files in temporary directories; `replay` is tested to
leave the tree untouched, and `clean` to delete nothing without `--yes`.

## Architecture

Dependencies point one way, and `tests/test_architecture.py` enforces it by reading the imports
of every module:

```
cli  ──▶  core  ──▶  utils
```

- **`utils`** holds leaf helpers that import nothing else from `mechane`.
- **`core`** is the domain model (`Config` … `Laboratory`) and the work it does (`runner`,
  `params`, `manifest`, `plan`, `doctor`, …). It knows nothing about terminals: `run_instance` and
  `run_batch` report progress as events (`InstanceStarted`, `InstanceSkipped`, `InstanceSaved`,
  `InstanceFailed`) to an optional listener instead of printing, and the other modules return data.
- **`cli`** parses options, calls `core`, and decides the wording (`cli/_output.py`). Commands hold
  no logic of their own.

To add a command: put the logic in `core` as a function that returns data (or reports through a
listener), add `cli/commands/<name>.py` with the click options and the formatting, and register it
in `cli/commands/__init__.py` and `cli/__init__.py`. Commands that act on some experiments take
their options from `cli/_options.py` (`selection_options`) and resolve them with `selected`. Only
the top-level names in `mechane.__all__` are the stable API; the modules above are importable but
may move.

## Citing and license

If `mechane` contributes to a publication, please cite it. A `CITATION.cff` and a Zenodo DOI can be
added once the first release is tagged.

License: *to be chosen.* Add a `LICENSE` file before sharing the code.
