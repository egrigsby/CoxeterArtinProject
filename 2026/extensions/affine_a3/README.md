# Affine Ã₃ — does the result transfer to a bigger group?

The normal-form arm solves Ã₂ (3 generators) completely. This asks whether that survives a
larger group: **Ã₃**, 4 generators, `TOKEN_TYPES = 5`, `DIM_OUTPUT = 4`.

Exhaustive ShortLex normal-form words of length ≤ 18 — **4,254 words**, 30/70 split, same frozen
model, 10k epochs.

| Run | Labels | Test exact | Test bit | Source log |
|---|---|---|---|---|
| [`left/`](left/) | per-prefix left descent set | **0.9971** | 0.9993 | `LeftNFTransformerA3L18_2761963.out` |
| [`right/`](right/) | per-prefix right descent set | **0.9860** | 0.9962 | `RightNFTransformerA3L18_2762013.out` |

Both train to 1.0000. The result largely transfers — but not perfectly, and the same left-easier-
than-right gap appears here as everywhere else.

## Building

`right/` inherits `left/`'s words and partition, so build in order:

```bash
python build_left_descent_nf_dataset.py    # BFS enumeration -> left/{train,test}.csv
python build_right_descent_nf_dataset.py   # reads left/, relabels -> right/{train,test}.csv
cd left && mkdir -p logs && sbatch transformer_job.sl
```

`resplit_train_test.py` repartitions an existing pair of CSVs in place at a different
`TRAIN_FRAC`; point its `TRAIN_CSV` / `TEST_CSV` at whichever run folder you want resplit.
