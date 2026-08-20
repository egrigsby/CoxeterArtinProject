# Minimal Length (all reduced words, Left vs Right descents)

Two parallel tests of the frozen shared model on the **exhaustive set of
minimal-length words**: every reduced expression of every element of Ã₂ up to
length 16 (5,259 words — a word w·s is reduced iff s is not a right descent of
w, so the language is prefix-closed and enumerable by BFS). Unlike the
normal-form arm (one ShortLex normal form per element) this dataset
contains *all* reduced words, and unlike the `Reduced` sweep (locally reduced
only) every word is globally reduced.

- **`left_descent/`**, **`right_descent/`** — per-prefix left / right descent sets.
- **`left_ascent/`**, **`right_ascent/`** — the ascent counterparts, built by relabelling the two above.

Both tests use the **identical words and identical 80/20 train/test partition**
(one shuffle, seed 0); only the labels differ, so any performance gap is
attributable to the left-vs-right labeling alone.

## Files

- `build_minimal_length_datasets.py` — shared dataset builder (self-contained:
  embeds the Tits-representation machinery from `../../shared/descents.py`
  plus the left-descent variant). Runs smoke tests on every build (3ℓ elements
  per shell, brute-force reduced-word check to length 8, prefix-closure,
  left = reversed-right crosscheck), then writes `train.csv`/`test.csv` into
  `left_descent/` and `right_descent/`. Run it from this folder.
- Each subfolder is a standard self-contained run: `config.py`
  (`SEQUENCE_LENGTH = 16`, `NUM_EPOCHS = 50000`, otherwise the frozen shared
  config), the frozen `../../shared/Transformer_presplit.py`,
  `Analysis.ipynb`, `transformer_job.sl`, pre-split CSVs, and `logs/`.

## Running

From inside `left_descent/` or `right_descent/`: `mkdir -p logs && sbatch transformer_job.sl`.
Checkpoints land at `workspace/_scratch/model.pth` as usual.

First launch 2026-07-19: jobs 2731473 (Left) and 2731474 (Right).
