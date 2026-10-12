# mechane + Snakemake

A small project that uses both tools. `mechane` decides *what* runs and where the results go
(seeds, directories, which instances are done); Snakemake decides the *order* of the stages and,
on a cluster, how jobs are submitted. mechane itself needs no scheduler settings.

```bash
pip install mechane snakemake
cd examples/snakemake
snakemake -c4          # runs everything, 4 jobs at a time
cat results/summary.txt
```

| File | What it is |
|---|---|
| `decay.py` | The laboratory: Monte Carlo radioactive decay. Replace it with your own. |
| `params.py` | The sweep (`ROOT`, `LABORATORY`, `PARAMS`). |
| `Snakefile` | The stages, below. |
| `summarize.py` | The analysis: compares the estimates with the exact answer `exp(-rate * t)`. |

## The stages

| Rule | Runs | Why a separate stage |
|---|---|---|
| `setup` | `mechane setup` | Expands the sweep and writes the manifest. Re-runs when `params.py` or `decay.py` change. |
| `make_plan` | `mechane plan` | Splits the work into tasks. It is a *checkpoint*: how many tasks there are is only known once the manifest exists. |
| `run_task` | `mechane run-plan` | One Snakemake job per mechane task. Finished instances are skipped. |
| `complete` | `mechane status --check` | A gate: fails unless every instance of every experiment is done. |
| `export` | `mechane export` | The results as a table, one row per instance. |
| `summarize` | your script | Your analysis. |

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
- **Changing the sweep.** Editing `params.py` re-runs `setup` and `make_plan`. Experiment ids are
  positional, so adding a point in the middle renumbers them; the plan is regenerated for you, but
  do not reuse a plan file you saved by hand (see "Known limitations" in the main README).
- **Clusters.** Snakemake runs jobs on a cluster through its executor plugins and profiles. The
  account, partition, memory, run time and container settings belong in a Snakemake profile for
  your cluster, not in mechane. The `runtime=10` resource in the Snakefile (minutes) is only a
  placeholder, and local runs ignore it. This example was run with Snakemake's local executor; it
  has not been run on a cluster here.
