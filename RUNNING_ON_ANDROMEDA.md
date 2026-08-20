# Running Jobs on Andromeda

Cluster mechanics for Boston College's Andromeda cluster (`andromeda.bc.edu`). For *what* to run
and why, see [`2026/REPLICATION.md`](2026/REPLICATION.md). For environment setup, see
[`Setup/README.md`](Setup/README.md).

> **Rough job-time estimate:** `words (in thousands) × epochs (in thousands) × 7.2` seconds.
>
> - **Under ~1 hour** → run interactively (faster turnaround).
> - **Longer** → submit a batch job with `sbatch`.

---

## Interactive (jobs under ~1 hour, and all short work)

SSH into your `andromeda.bc.edu` account, then in a **bash** terminal:

```bash
interactive -G 1
module use /m31/modulefiles/static
module load miniconda
conda activate /projects/expmmllab/CoxeterEnv
```

Then `cd` into the run folder you want — e.g. `2026/arms/reduced/length16/`.

Data generation, quick scripts, and notebook work all belong here rather than on a login node.
Hold **at most one** `interactive` allocation at a time; check with `squeue -u $USER` before
requesting another.

### Generate a dataset

```bash
python ../build_descent_dataset.py      # path varies by arm — see that folder's README
```

Set the knobs at the top of the builder first, and make `SEQUENCE_LENGTH` in `config.py` match
the builder's `FIXED_LENGTH`. Nothing enforces this.

### Train

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
python ../../../shared/Transformer.py
```

The `PYTHONPATH` line is what makes the shared model pick up *this* folder's `config.py`. The
`../../../` depth depends on how deep the run folder sits; the `transformer_job.sl` in each
folder already has it right, so copy from there if in doubt.

The trained model is saved to `workspace/_scratch/model.pth` in the run folder.

---

## Batch (longer jobs)

```bash
cd 2026/arms/reduced/length16
mkdir -p logs
sbatch transformer_job.sl
squeue -u $USER
```

**`mkdir logs` is not optional.** SLURM does not create the directory named in `--output`, and a
job whose log directory is missing fails without writing an error anywhere — it looks like the
job never ran.

Every `transformer_job.sl` activates the environment itself, so you do not need an interactive
session to submit one. Job output goes to `logs/<jobname>_<jobid>.out`, errors to `.err`.

### Adjusting a job script

| Line | When to change it |
|---|---|
| `--job-name` | Always — so you can find it in `squeue`. |
| `--time` | The committed values are `04:00:00` for 50k-epoch runs and `02:00:00` for 10–15k. If a log shows the job hitting the wall, cancel it, raise this, resubmit. |
| `--gres=gpu:a100:1` | Keep for training (CUDA 12.8 needs A100 or newer — V100s fail at runtime). **Drop it entirely for checkpoint analysis**, which needs no GPU and otherwise waits in `PD`. |
| `--mail-user` | Uncomment and set to your BC address for start/end/fail email. |
| the build line | Commented out by default. Uncomment only if you want the job to regenerate the dataset before training. |

---

## Analysis

After training, open the analysis notebook matching your model variant, in `2026/shared/`:

| Notebook | For runs whose config uses |
|---|---|
| `Analysis.ipynb` | `DATA_CSV` + `TRAINING_SPLIT` |
| `Analysis_presplit.ipynb` | `TRAIN_CSV` + `TEST_CSV` |
| `Analysis_classification.ipynb` | cross-entropy targets |

Set `ARM_DIR` in the first code cell to your run's folder, then run the cells top to bottom. The
notebook reproduces the exact train/test split used during training by calling the same loader
the training script used.

> If loading the checkpoint fails on an unpickling error, it is almost certainly a
> `transformer_lens` version mismatch rather than a damaged file — see
> [`Setup/README.md`](Setup/README.md).
