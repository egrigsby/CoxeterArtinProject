# Random Arm — unconstrained random words

Words drawn uniformly at random over the generators of Ã₂ with **no constraints at all** —
repeats allowed, so `aab` and `aaa` are valid training words. This is the least structured of the
three arms and the baseline the other two are compared against: whatever the model achieves here
is what it can do without any help from the data's shape.

## Status

**No training log for this arm is in the repo.** The configs are here and the generator runs, but
the runs themselves happened on the arm owner's machine. If you have those logs, add a row to
[`../../RESULTS.md`](../../RESULTS.md) — the table already has a placeholder marked
*no run log*.

## Files

| File | Purpose |
|---|---|
| `Set_Generation(Right_Descent).py` | Generates `data.csv`. Computes per-prefix right descent sets from a **minimal-root reflection table** over the alphabet `{a, b, c}`, then maps to token IDs and bitmasks. A different implementation from `shared/descents.py`, which uses the geometric (Tits) representation — the two agree, and the older string-based ancestors of this file are in [`../../reference/legacy_generators/`](../../reference/legacy_generators/). |
| `config.py` | 1-layer baseline — the frozen shared configuration at `SEQUENCE_LENGTH = 22`. |
| `config_2layer.py` | Hyperparameter exploration: `LAYERS = 2`, `DIM_MODEL = 64`, `DIM_HEADS = 16`, `DIM_MLP = 128`, `NORMALIZATION = "LNPre"`, `WEIGHT_DECAY = 0.05`, `NUM_EPOCHS = 100000`. **Departs from the frozen shared config** — results from it are not comparable to the other two arms. |
| `transformer_job.sl` | Submits the run. |

> `config_2layer.py` exists because the arm was exploring capacity, but note what it costs: it
> changes normalization, width, depth, and weight decay simultaneously, so a difference against
> the other arms cannot be attributed to the data. Use `config.py` for anything that feeds the
> three-arm comparison.

## Running

```bash
# INSTANCES (word count) is at the top of the generator; word length comes from
# config.py's SEQUENCE_LENGTH, which the generator imports directly.
python "Set_Generation(Right_Descent).py"
mkdir -p logs
sbatch transformer_job.sl
```

To run the 2-layer configuration instead, either swap the files or copy this folder and rename
`config_2layer.py` to `config.py` there — the shared model always imports the module named
`config`.
