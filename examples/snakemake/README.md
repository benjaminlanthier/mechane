# mechane + Snakemake

A small project that uses both tools. `mechane` decides *what* runs and where the results go
(seeds, directories, which instances are done); Snakemake decides the *order* of the stages and,
on a cluster, how jobs are submitted. mechane itself needs no scheduler settings.

The params file defines **two labs**, each with its own sweep, seed and number of instances:
`decay` estimates the fraction of atoms that survive to time `t`, and `half_life` estimates the
half-life. `mechane setup` sets up both; every later stage runs once per lab.

```bash
pip install mechane snakemake
cd examples/snakemake
snakemake -c4          # runs everything, 4 jobs at a time
cat outputs/summary.txt
```

| File | What it is |
|---|---|
| `decay.py` | The two laboratories, `DecayLab` and `HalfLifeLab`: Monte Carlo radioactive decay. Replace them with your own. |
| `params.py` | The two labs and their sweeps (`LABS`: a `ROOT`, `LABORATORY` and `PARAMS` per lab). |
| `Snakefile` | The stages, below. |
| `summarize.py` | The analysis: compares the estimates with the exact answers, `exp(-rate * t)` and `ln 2 / rate`. |

## The stages

| Rule | Runs | Why a separate stage |
|---|---|---|
| `setup` | `mechane setup` | Expands both sweeps and writes one manifest per lab, `outputs/<lab>/manifest.json`. Re-runs when `params.py` or `decay.py` change. |
| `make_plan` | `mechane plan` | Splits a lab's work into tasks, `plans/<lab>.json`. It is a *checkpoint*: how many tasks there are is only known once the manifest exists. |
| `run_task` | `mechane run-plan` | One Snakemake job per mechane task. Finished instances are skipped. |
| `complete` | `mechane status --check` | A gate: fails unless every instance of every experiment of the lab is done. |
| `export` | `mechane export` | A lab's results as a table, `outputs/<lab>.csv`, one row per instance. |
| `summarize` | your script | Your analysis, of both tables. |

Every rule after `setup` has a `{lab}` wildcard, so the two labs go through their stages
independently, and the tasks of both labs run side by side. Snakemake stops starting new jobs
after a failure; with `snakemake -c4 --keep-going` the other lab still finishes its own stages.

## Adding a lab

1. Add the lab to `LABS` in `params.py`, with its own `ROOT` under `outputs/`.
2. Add the last part of that `ROOT` to `LABS` in the Snakefile: the per-lab rules find a lab's
   manifest at `outputs/<lab>/manifest.json`.
3. Tell `summarize.py` how to analyse its table (the `LABS` dict at the top).

## Things to know

- **Granularity.** One Snakemake job runs a *task* of several instances (`BATCH_SIZE` in the
  Snakefile), not one instance. On a small test, running 12 instances as 12 separate
  `mechane run` processes took about ten times longer than one `mechane run-batch`, because every
  process starts Python and loads the manifest again. That matters when instances are short;
  `mechane stats --batch-size N` helps you choose `BATCH_SIZE` and the `runtime` resource.
- **mechane is the source of truth for "done".** Snakemake's `done/task_N` files only mean
  "this task ran without failures". If you delete or `mechane clean` results behind Snakemake's
  back, it will not notice: `snakemake -c4` says "Nothing to be done". Tell it with
  `snakemake -c4 --forcerun run_task`, which is cheap because mechane skips finished instances.
- **Failures.** `mechane run-plan` exits with status 1 when any instance of its task failed, so
  the job fails, and `complete`, `export` and `summarize` do not run. Look with `mechane errors`
  and `mechane doctor`, fix the code, and run `snakemake` again: the failed tasks run again.
- **Changing the sweep.** Editing `params.py` re-runs `setup` and `make_plan` for every lab
  (finished instances are still skipped). Experiment ids are
  positional, so adding a point in the middle renumbers them; the plan is regenerated for you, but
  do not reuse a plan file you saved by hand (see "Known limitations" in the main README).
- **Clusters.** Snakemake runs jobs on a cluster through its executor plugins and profiles. The
  account, partition, memory, run time and container settings belong in a Snakemake profile for
  your cluster, not in mechane. The `runtime=10` resource in the Snakefile (minutes) is only a
  placeholder, and local runs ignore it. This example was run with Snakemake's local executor; it
  has not been run on a cluster here.
