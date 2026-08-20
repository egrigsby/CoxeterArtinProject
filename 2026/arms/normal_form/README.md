# Normal-Form Arm — inverse-ShortLex normal-form words

Words in **ShortLex normal form**: the lexicographically-least reduced word for each group
element (generator order 1 < 2 < 3). Exactly one word per element, and the dataset is
**exhaustive** — every normal-form word of length 1..36 in Ã₂, which is 3ℓ elements per length ℓ,
**1,998 words** total. Variable length, padded to 36.

Two structural properties make this arm different from the other two:

1. **Prefix-closure.** The ShortLex language is prefix-closed: every prefix of a normal-form word
   is itself a normal form. So every one of the model's per-prefix predictions is a prediction
   about a genuine dataset element, not an arbitrary intermediate.
2. **Global reducedness.** Every word is a reduced expression, so word length equals the group
   element's length.

## Results

All four variants are solved **completely** — 1.0000 train *and* test exact-match, on a held-out
70%.

| Run | Labels | Epochs | First reaches 1.0000 test | Source log |
|---|---|---|---|---|
| [`left_descent/`](left_descent/) | per-prefix **left** descent set | 10k | epoch 700 | `LeftNFTransformer_2717592.out` |
| [`right_descent/`](right_descent/) | per-prefix **right** descent set | 15k | epoch 11,300 | `RightNFTransformer_2717954.out` |
| [`left_ascent/`](left_ascent/) | per-prefix **left** ascent set | 10k | epoch 800 | `LeftNFAscent_2732941.out` |
| [`right_ascent/`](right_ascent/) | per-prefix **right** ascent set | 15k | epoch 4,700 | `RightNFAscent_2732942.out` |

Two things to take from this:

- **The normal-form arm solves what the reduced arm cannot.** At length 22 the reduced arm scores
  0.62 test exact-match; this arm reaches 1.0000 at length 36 on a *smaller* training set
  (599 words). The difference is the structure of the data.
- **Left is roughly an order of magnitude faster than right.** Both end at 1.0000, but left
  descents are available directly from the prefix a causal model has already read, while right
  descents are not. The same gap appears in `extensions/minimal_length/` on a completely
  different dataset.

Ascent labels are the complement of descent labels (in any Coxeter group every generator is
exactly one of ascent or descent), so the ascent runs are a consistency check on the label
pipeline rather than an independent result. The builders verify this explicitly, position by
position, on every row.

## How the four runs share one dataset

Only `left_descent/` enumerates words from scratch. The other three **inherit its words and its
exact train/test partition** and rewrite only the `descents` column, so any performance
difference is attributable to the labels alone and nothing else:

```
left_descent/   build_left_descent_nf_dataset.py     BFS enumeration -> train.csv, test.csv
right_descent/  build_right_descent_dataset.py       reads ../left_descent/, relabels
left_ascent/    build_left_ascent_nf_dataset.py      reads ../left_descent/, relabels
right_ascent/   build_right_ascent_dataset.py        reads ../right_descent/, relabels
```

Build them in that order. Each builder runs its own smoke tests (shell sizes, prefix-closure,
brute-force cross-check at small lengths, left-vs-reversed-right agreement) and fails loudly
rather than writing a bad CSV.

`left_descent/` also ships `resplit_train_test.py`, which repartitions the existing CSVs in place
at a different `TRAIN_FRAC` — that is how the 30/70 split was produced from the original 80/20.

## Running

```bash
cd left_descent
python build_left_descent_nf_dataset.py
mkdir -p logs
sbatch transformer_job.sl
```

All four use `shared/Transformer_presplit.py`, since the partition is fixed by the two CSV files
rather than recomputed at load time.

## `right_descent/pranav_nfdata/`

A separate, unfinished pipeline for the same task from a different direction: a hand-produced
`NFdata.csv` of normal-form words with their final descent sets, plus a converter to the
two-column training format. It is **not** the run that produced the results above, and its
converter still expects an input file (`NFdata (1).csv`) that is not in the repo. Kept because
the `NFdata` CSVs are original data, not regenerable from a script here.
