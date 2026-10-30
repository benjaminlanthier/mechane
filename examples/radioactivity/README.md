# radioactivity: a research project that runs its simulations with mechane

This directory stands for your own research repository. It is a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) of two packages, installed
side by side in one virtual environment:

- the science, `radioactivity`, at the root, which knows nothing about mechane;
- the numerical experiments, `radioactivity-experiments`, in the `experiments` member: the
  mechane laboratories, built on the science package.

The workflow that runs the experiments (the Snakefile, the params file and the analysis) sits at
the root, next to both.

```
radioactivity/                     the workspace root
├── pyproject.toml                 package `radioactivity`; members = ["experiments"]; group `workflow`
├── uv.lock                        one lock for the whole workspace
├── src/radioactivity/decay.py     the science: sampling lifetimes, measuring them, the exact answers
├── tests/                         the science package's own tests
├── experiments/                   the workspace member, package `radioactivity-experiments`
│   ├── pyproject.toml             depends on radioactivity and mechane; registers the labs
│   └── src/radioactivity_experiments/families/
│       ├── decay.py               the `decay` family: DecayLab
│       └── half_life.py           the `half_life` family: HalfLifeLab
├── params.py                      the workflow: the two labs and their sweeps
├── Snakefile                      the workflow: the stages, below
└── summarize.py                   the workflow: the analysis
```

## Run it

From this directory:

```bash
uv run --group workflow snakemake -c4     # runs everything, 4 jobs at a time
cat outputs/summary.txt
```

The root's `workflow` group installs the `experiments` member (and mechane, its dependency) and
Snakemake into `.venv`, next to the science package. So the `mechane` command is on the PATH and
the labs are importable. Any single mechane command works the same way, for example
`uv run --group workflow mechane status --manifest-path outputs/decay/manifest.json`.

The science package is tested on its own, without mechane or the labs:

```bash
uv run pytest
```

## Who does what

| Part | Knows about | Its job |
|---|---|---|
| `radioactivity` | numpy | The science: `sample_lifetimes`, `surviving_fraction`, `half_life`, and the exact answers. Reusable and testable on its own. |
| `radioactivity_experiments.families` | `radioactivity`, mechane | One module per experiment family, each with one thin lab. A lab says what its parameters are and which function of `radioactivity` an instance calls with them. Each module defines its own sections, so a change to one family never affects the other. |
| `params.py` | the lab names | What to run: two labs, each with its own sweep, seed and number of instances. |
| `Snakefile` | mechane's commands | In which order, and how many jobs at once. |
| `summarize.py` | `radioactivity` | The analysis: compares the estimates with the exact answers of the science package. |

The two labs:

| Lab | Measures | Exact answer | Sweep | Seed | Instances |
|---|---|---|---|---|---|
| `decay` | the fraction of atoms still alive at time `t` | `exp(-rate * t)` | 3 rates × 2 `n_atoms` = 6 experiments | 1 | 20 |
| `half_life` | the half-life (the median lifetime) | `ln 2 / rate` | 3 rates = 3 experiments | 2 | 50 |

## How the pieces are wired

- **The workspace.** The root `pyproject.toml` lists `experiments` as a member, and the member
  depends on `radioactivity` with `{ workspace = true }`. The root's `workflow` dependency group
  depends on the member the same way, which is what installs the labs when the workflow runs from
  the root. A group is not part of the package, so `radioactivity` itself still depends only on
  numpy. Both packages are installed in editable mode, so a change to the science or to the labs
  is used by the next run without reinstalling.
- **mechane.** This example uses the mechane of this repository (`path = "../../.."`), so that it
  always runs the code next to it. A real project pins a release instead:

  ```toml
  [tool.uv.sources]
  mechane = { git = "https://github.com/benjaminlanthier/mechane", tag = "v0.2.0" }
  ```

- **The labs are entry points.** `experiments/pyproject.toml` registers them:

  ```toml
  [project.entry-points."mechane.laboratories"]
  decay = "radioactivity_experiments.families.decay:DecayLab"
  half_life = "radioactivity_experiments.families.half_life:HalfLifeLab"
  ```

  so `params.py` says `LABORATORY = "decay"`. The manifest still records the full
  `"module:Class"` path, so a job finds the lab on any machine where the workspace is installed.
  Because the labs are installed, the Snakefile needs no `PYTHONPATH`.
- **One params file, two labs.** `params.py` defines `LABS`, one entry per lab, each with its own
  `ROOT` under `outputs/`. `mechane setup` sets up both.

## The stages

| Rule | Runs | Why a separate stage |
|---|---|---|
| `setup` | `mechane setup` | Expands both sweeps and writes one manifest per lab, `outputs/<lab>/manifest.json`. Re-runs when `params.py` or a module of `families/` changes. |
| `make_plan` | `mechane plan` | Splits a lab's work into tasks, `plans/<lab>.json`. It is a *checkpoint*: how many tasks there are is only known once the manifest exists. |
| `run_task` | `mechane run-plan` | One Snakemake job per mechane task. Finished instances are skipped. |
| `complete` | `mechane status --check` | A gate: fails unless every instance of every experiment of the lab is done. |
| `export` | `mechane export` | A lab's results as a table, `outputs/<lab>.csv`, one row per instance. |
| `summarize` | your script | Your analysis, of both tables. |

Every rule after `setup` has a `{lab}` wildcard, so the two labs go through their stages
independently, and the tasks of both labs run side by side. Snakemake stops starting new jobs
after a failure; with `snakemake -c4 --keep-going` the other lab still finishes its own stages.

## Adding a lab

1. Write the science it needs in `radioactivity`, and the lab in a new module of
   `experiments/src/radioactivity_experiments/families/` (export it from `families/__init__.py`,
   and list the module as an input of the Snakefile's `setup` rule).
2. Register the lab under a short name in `experiments/pyproject.toml`, then run
   `uv sync --group workflow` so the new entry point is installed.
3. Add the lab to `LABS` in `params.py`, with its own `ROOT` under `outputs/`.
4. Add the last part of that `ROOT` to `LABS` in the Snakefile: the per-lab rules find a lab's
   manifest at `outputs/<lab>/manifest.json`.
5. Tell `summarize.py` how to analyse its table (the `LABS` dict at the top).

## Things to know

- **Changing the science does not change where results go.** mechane hashes parameters, not
  code, so a fix in `radioactivity` leaves finished instances in place, and they are skipped.
  Remove them with `mechane clean --what results … --yes` and run again, or version your configs (see "Design notes and
  limitations" in the main README). `mechane doctor` reports results produced by more than one
  recorded code version.
- **Granularity.** One Snakemake job runs a *task* of several instances (`BATCH_SIZE` in the
  Snakefile), not one instance, because every process starts Python and loads the manifest again.
  `mechane stats --batch-size N` helps you choose `BATCH_SIZE` and the `runtime` resource.
- **mechane is the source of truth for "done".** Snakemake's `done/<lab>/task_N` files only mean
  "this task ran without failures". If you delete or `mechane clean` results behind Snakemake's
  back, it will not notice. Tell it with `--forcerun run_task`, which is cheap because mechane
  skips finished instances.
- **Failures.** `mechane run-plan` exits with status 1 when any instance of its task failed, so
  the job fails, and the lab's `complete`, `export` and `summarize` do not run. Look with
  `mechane errors` and `mechane doctor`, fix the code, and run `snakemake` again: the failed tasks
  run again.
- **Changing the sweep.** Editing `params.py` re-runs `setup` and `make_plan` for every lab.
  Experiment ids are positional, so adding a point in the middle renumbers them; the plans are
  regenerated for you, but do not reuse a plan file you saved by hand.
- **Clusters.** Snakemake runs jobs on a cluster through its executor plugins and profiles. The
  account, partition, memory, run time and container settings belong in a Snakemake profile for
  your cluster, not in mechane. The `runtime=10` resource in the Snakefile (minutes) is only a
  placeholder, and local runs ignore it. This example was run with Snakemake's local executor; it
  has not been run on a cluster here.
