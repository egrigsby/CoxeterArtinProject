# Reduced Arm — locally reduced words

Words drawn uniformly over the generators of Ã₂ **avoiding immediate repeats**: no letter is ever
followed by itself. This makes each word *locally* reduced. Most are **not** globally reduced —
`aba` and `bab` are the same element, and the model gets no help from that. The descent-path
logic in [`../../shared/descents.py`](../../shared/descents.py) handles globally non-reduced
elements correctly, so the labels are right regardless.

This arm is the length sweep: the same model at three word lengths, looking for a length short
enough for a 1-layer transformer to learn but long enough to force a general solution rather than
memorization.

## Results

| Run | Words | Length | Split | Test exact | Test bit |
|---|---|---|---|---|---|
| [`length10/`](length10/) | 1,536 | 10 | 40/60 | **0.9348** | 0.9752 |
| [`length16/`](length16/) | 2,000 | 16 | 40/60 | **0.7801** | 0.9024 |
| [`length22/`](length22/) | 2,000 | 22 | 40/60 | **0.6202** | 0.8117 |

Train accuracy is ≥ 0.988 in all three. The falling test accuracy against pinned train accuracy
is memorization, and it gets worse the longer the words are. Full numbers and source logs in
[`../../RESULTS.md`](../../RESULTS.md).

> **Reading these numbers honestly.** With no-adjacent-repeat data, short prefixes are
> exhaustively covered by the training set — at length ≤ 11 there simply are not enough distinct
> prefixes for a held-out one to be genuinely unseen. Only **prefix lengths ≥ 12** carry real
> generalization signal. A high aggregate test accuracy at length 10 partly reflects that
> coverage rather than learning.

## Files

| File | Purpose |
|---|---|
| `build_descent_dataset.py` | Generates `data.csv` for a sweep point. Imports `reflection_matrices` and `right_descent_path` from `../../shared/descents.py`. |
| `length{10,16,22}/config.py` | The sweep points. `SEQUENCE_LENGTH` is the only thing that differs. |
| `length{10,16,22}/transformer_job.sl` | Submits that sweep point. |

## Running a sweep point

```bash
cd length16
# edit the knobs at the top of ../build_descent_dataset.py:
#   COXETER_MATRIX, NUM_WORDS, MIN_LEN, MAX_LEN, FIXED_LENGTH, SEED, OUTPUT_CSV
python ../build_descent_dataset.py
mkdir -p logs
sbatch transformer_job.sl
```

`FIXED_LENGTH` in the builder **must** equal `SEQUENCE_LENGTH` in `config.py`. Nothing enforces
it; `python ../../check_repo.py --configs` will tell you if they disagree.

Adding a sweep point means copying one of the `length*/` folders, changing `SEQUENCE_LENGTH`, and
setting the builder's `FIXED_LENGTH` to match before generating.

## Next steps

The sweep is what the 2-layer experiment builds on: pick the length where a 1-layer model clearly
fails to generalize, then change `LAYERS = 2` and nothing else.
