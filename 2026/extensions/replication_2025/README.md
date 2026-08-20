# 2025 Replication

The bridge from [`../../../2025/`](../../../2025/) to the 2026 work: a replication of last year's
task on this year's stack.

The 2025 question was **binary** — is a given word trivial (equal to the identity) or not? — over
129,300 words of Ã₂ at length 22, with **bidirectional** attention. The 2026 work replaced it
with the per-prefix multi-label descent-set task and causal attention, so this folder is the last
point where the two are comparable.

| File | Purpose |
|---|---|
| `Transformer.py` | The replication training script. Configuration lives **inside** the script rather than in a separate `config.py`. |
| `Transformer.ipynb` | Notebook version. |
| `train.csv` / `test.csv` | The 2025 dataset (30/70 split), also present under `2025/transformer/`. |
| `transformer_job.sl` | Submits the run. |

**No training log is in the repo.**

> ⚠️ Known defect, left as found: the configuration block at the top of `Transformer.py` has
> trailing commas on every line (`SEQUENCE_LENGTH=22,`), which makes each value a **1-tuple**
> rather than a scalar. Anyone reviving this run should fix that first. It is not fixed here
> because this folder is a historical record, not an active experiment.
