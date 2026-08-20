# Replicating a 2026 Run

Every result in [`RESULTS.md`](RESULTS.md) was produced by the same five steps. This document is
the runbook; if you follow it verbatim for one of the arms you should land on the numbers in that
table.

**Prerequisite:** a working environment — see [`../Setup/README.md`](../Setup/README.md).

---

## How a run is laid out

The model lives once, in [`shared/`](shared/). Each run is a folder under [`arms/`](arms/) or
[`extensions/`](extensions/) holding only what makes *that* run different: a `config.py`, a
dataset builder, and a `transformer_job.sl`.

```
2026/shared/Transformer.py           <- the model; every arm runs this same file
2026/arms/reduced/length16/
    config.py                        <- the only thing that differs between sweep points
    transformer_job.sl               <- submits the job
    data.csv                         <- you generate this (gitignored)
    logs/                            <- you create this
    workspace/_scratch/model.pth     <- the run writes this
```

The job script puts its own folder on `PYTHONPATH` before invoking the shared model:

```bash
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$PWD:$PYTHONPATH"
python ../../../shared/Transformer.py
```

That one line is what makes `from config import *` inside the model resolve to *this run's*
config, so the checkpoint and curves are written here rather than into `shared/`. It works
because Python puts the script's own directory first on `sys.path` and `shared/` deliberately
contains no `config.py`. Nothing else about the model is run-specific.

### Which model does a run use?

| Config declares | Model | Used by |
|---|---|---|
| `DATA_CSV` + `TRAINING_SPLIT` | `shared/Transformer.py` | reduced arm, random arm — one CSV, shuffled and split at load time with `DATA_SEED` |
| `TRAIN_CSV` + `TEST_CSV` | `shared/Transformer_presplit.py` | normal-form arm, minimal_length, affine_a3 — a fixed partition baked into two files |
| Cross-entropy targets | `shared/Transformer_classification.py` | element_classification, nf_generation — single label per position, not a multi-label bitmask |

The job script in each folder already points at the right one. If you add a run, copy the
nearest existing folder rather than writing the script from scratch.

---

## The five steps

### 1. Activate the environment

```bash
interactive -G 1
module use /m31/modulefiles/static
module load miniconda
conda activate /projects/expmmllab/CoxeterEnv
```

### 2. Pick a run and generate its dataset

```bash
cd 2026/arms/reduced/length16
python ../build_descent_dataset.py
```

Each builder has its knobs in a block at the top of the file — for the reduced arm those are
`COXETER_MATRIX`, `NUM_WORDS`, `MIN_LEN` / `MAX_LEN`, `FIXED_LENGTH`, and `SEED`. They are
per-experiment settings, not defaults to be preserved.

Builders for the exhaustive datasets (normal form, minimal_length, affine_a3) enumerate by BFS
and take no size parameter — the dataset is whatever the group has up to the length bound. Those
builders run their own smoke tests on every build (shell sizes, prefix-closure, brute-force
cross-checks at small lengths), so a silent failure there is unlikely.

> ⚠️ **`SEQUENCE_LENGTH` in `config.py` must equal `FIXED_LENGTH` in the builder.** Nothing
> enforces this. If they disagree you get a shape error at best and a quietly truncated dataset
> at worst. Check both before submitting. `python check_repo.py --configs` verifies it for every
> run in the repo.

### 3. Check the job script

Open `transformer_job.sl` and confirm:

- `--job-name` is something you will recognize in `squeue`
- `--time` is enough. The 50k-epoch runs take longer than the 10–15k ones; the committed scripts
  use `04:00:00` and `02:00:00` respectively. If a log shows the job hitting the wall, cancel,
  raise `--time`, resubmit.
- the dataset-build line is still commented out (you built it in step 2)

### 4. Submit

```bash
mkdir -p logs          # SLURM will NOT create this, and the job fails silently without it
sbatch transformer_job.sl
squeue -u $USER
```

### 5. Read the results

Training prints one line per 100 epochs to `logs/<jobname>_<jobid>.out`:

```bash
grep '^Epoch' logs/*.out | tail -1
```

The checkpoint lands at `workspace/_scratch/model.pth` and contains the config, final weights,
optimizer and scheduler state, a weight snapshot every `CHECKPOINT_STEP` epochs, and the full
per-epoch loss/accuracy history.

For plots and interpretability, open [`shared/Analysis.ipynb`](shared/Analysis.ipynb) (or
`Analysis_presplit.ipynb` / `Analysis_classification.ipynb` to match your model), set `ARM_DIR`
in the first code cell to your run's folder, and run top to bottom.

---

## Things that will bite you

- **Missing `logs/`.** Covered above, and it is the single most common way to lose an hour: the
  job disappears without producing an error anywhere.
- **`SEQUENCE_LENGTH` ≠ `FIXED_LENGTH`.** Also covered above. Unenforced by design.
- **Login-node jobs.** Data generation on a login node is both slow and against cluster policy.
  `interactive -G 1` first, and hold only one allocation at a time.
- **Checkpoint "corruption" that is really a version mismatch.** `transformer_lens` pickles do
  not travel across versions. Check `python --version` and `transformer_lens.__version__` against
  what produced the file before concluding it is damaged.
- **Train metrics look one step stale.** They are — logged metrics are computed before that
  epoch's optimizer step. At the end of a converged run a logged `1.0000` can correspond to
  weights with a few errors in them.
- **A100 requests pending forever.** For checkpoint *analysis* you do not need a GPU at all; drop
  the `--gres` line rather than waiting in the queue.

---

## Verifying the repo itself

```bash
python 2026/check_repo.py
```

Checks that documentation paths resolve, notebooks parse, Python files compile, no duplicate
files have crept back in, each config agrees with its builder, `RESULTS.md` matches the logs it
cites, and `Setup/requirements.txt` matches the live environment. Run it before opening a PR.
