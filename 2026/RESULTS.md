# 2026 Results

Every number below was extracted from the training run's own SLURM `.out` log — the final
`Epoch …` line and the `Loaded dataset` / `Train size` header lines. `check_repo.py --results`
re-extracts them and fails if this table has drifted. Where a configuration is present in the
repo but no run log exists, the row says so rather than being omitted.

## The frozen model

Every run in every table below uses the **same architecture and optimizer**. Only the dataset,
`SEQUENCE_LENGTH`, `TOKEN_TYPES`, `DIM_OUTPUT`, and `NUM_EPOCHS` change — that is what makes the
three-arm comparison a controlled one.

| | |
|---|---|
| Layers / heads / `d_head` | 1 / 4 / 64 |
| `d_model` / `d_mlp` | 256 / 1024 |
| Activation | `relu` |
| Normalization | `None` |
| Positional embedding | `standard` (learned absolute) |
| Attention | `causal` |
| Optimizer | AdamW, lr `1e-4`, weight decay `0.5`, betas `(0.9, 0.98)` |
| Batching | full-batch |
| Biases | frozen (`requires_grad=False`) |
| Seeds | `DATA_SEED = 598`, `LENS_SEED = 999` |

Two deliberate quirks: `scheduler.step()` is commented out, so `ReduceLROnPlateau` is constructed
but the learning rate never actually moves; and training is full-batch on purpose, for clean loss
curves. Neither is a bug — see [`shared/README.md`](shared/README.md).

**Metric definitions.** *Exact* = per-prefix exact-set-match (a position counts only if all
`DIM_OUTPUT` bits are right). *Bit* = per-generator bit accuracy, i.e. partial credit.
*Seq* = per-sequence accuracy (a whole word counts only if every one of its prefixes is right).
Descent/ascent runs report exact + bit; classification and sequence-output runs report exact + seq.

> The logged train metrics are computed *before* the optimizer step for that epoch, so they lag
> the saved weights by one step. This matters only at the very end of a converged run, where a
> logged `1.0000` can correspond to weights with a handful of errors.

---

## The three arms

The controlled comparison: identical model, identical objective (per-prefix right descent set of
Ã₂), datasets differing only in how the words are drawn.

| Arm | Words | Data | Seq len | Split | Epochs | Train exact | Test exact | Train bit | Test bit | Source log |
|---|---|---|---|---|---|---|---|---|---|---|
| **reduced** | locally reduced (no adjacent repeats) | 1,536 random | 10 | 40/60 | 50k | 1.0000 | **0.9348** | 1.0000 | 0.9752 | `Test4_L12_2714631.out` |
| **reduced** | locally reduced | 2,000 random | 16 | 40/60 | 50k | 0.9984 | **0.7801** | 0.9995 | 0.9024 | `Test2_L16_2700208.out` |
| **reduced** | locally reduced | 2,000 random | 22 | 40/60 | 50k | 0.9884 | **0.6202** | 0.9961 | 0.8117 | `Test7_L22_2717530.out` |
| **normal form** | inverse-ShortLex NF, right descents | 1,998 exhaustive (len 1–36) | 36 | 30/70 | 15k | 1.0000 | **1.0000** | 1.0000 | 1.0000 | `RightNFTransformer_2717954.out` |
| **normal form** | inverse-ShortLex NF, left descents | 1,998 exhaustive | 36 | 30/70 | 10k | 1.0000 | **1.0000** | 1.0000 | 1.0000 | `LeftNFTransformer_2717592.out` |
| **normal form** | NF, right *ascents* | 1,998 exhaustive | 36 | 30/70 | 15k | 1.0000 | **1.0000** | 1.0000 | 1.0000 | `RightNFAscent_2732942.out` |
| **normal form** | NF, left *ascents* | 1,998 exhaustive | 36 | 30/70 | 10k | 1.0000 | **1.0000** | 1.0000 | 1.0000 | `LeftNFAscent_2732941.out` |
| **random** | repeats allowed | — | 22 | 40/60 | — | — | — | — | — | *no run log — config only* |

**What the reduced sweep shows.** Test exact-match falls monotonically with word length —
0.93 → 0.78 → 0.62 — while train accuracy stays pinned near 1.0. The gap is memorization: a
1-layer model fits the training words at every length but generalizes worse the longer the words
get. Read this together with `repeats.txt` in the working tree: with no-adjacent-repeat data,
short prefixes are exhaustively covered by the training set, so only prefix lengths ≥12 carry
honest generalization signal.

**What the normal-form arm shows.** Every NF variant is solved completely — 1.0000 train *and*
test, on a held-out 70% — where the reduced arm at a comparable length is at 0.62. Ascent labels
behave identically to descent labels, as they must (in Ã₂ the ascent set is the complement of the
descent set), which is a useful sanity check on the label pipeline rather than a new result.

---

## Extensions

Experiments outside the three-arm comparison. Same frozen model unless the row says otherwise.

| Experiment | Task | Data | Seq len | Split | Epochs | Train | Test | Source log |
|---|---|---|---|---|---|---|---|---|
| **minimal_length** — left descent | per-prefix left descent set | 5,259 exhaustive *globally* reduced words, Ã₂ len 1–16 | 16 | 80/20 | 50k | 1.0000 exact | **1.0000 exact** | `MinLenLeft_2731473.out` |
| **minimal_length** — right descent | per-prefix right descent set | same 5,259 words, same split | 16 | 80/20 | 50k | 1.0000 exact | **1.0000 exact** | `MinLenRight_2731474.out` |
| **minimal_length** — left ascent | per-prefix left ascent set | same 5,259 words | 16 | 80/20 | 50k | 1.0000 exact | **1.0000 exact** | `MinLenLeftAscent_2732939.out` |
| **minimal_length** — right ascent | per-prefix right ascent set | same 5,259 words | 16 | 80/20 | 50k | 1.0000 exact | **1.0000 exact** | `MinLenRightAscent_2732940.out` |
| **affine_a3** — left descent | per-prefix left descent set, **Ã₃** | 4,254 NF words len ≤ 18 | 18 | 30/70 | 10k | 1.0000 exact | **0.9971 exact** / 0.9993 bit | `LeftNFTransformerA3L18_2761963.out` |
| **affine_a3** — right descent | per-prefix right descent set, **Ã₃** | same 4,254 words | 18 | 30/70 | 10k | 1.0000 exact | **0.9860 exact** / 0.9962 bit | `RightNFTransformerA3L18_2762013.out` |
| **element_classification** | which of the 6 elements of finite A₂ (= S₃) the prefix equals — cross-entropy, not multi-label | 4,000 words | 18 | 30/70 | 15k | 1.0000 exact | **0.9981 exact** / 0.9818 seq | `FiniteA2Class_2769002.out` |
| **nf_generation** | next-token LM over NF words (generate the normal form) | 1,998 NF words | 36 | 30/70 | 10k | 0.9091 exact | **0.8933 exact** / 0.0000 seq | `NFGenA2_2762307.out` |
| **nf_multiplication** — right | NF(w) × s → NF of the product | 5,994 pairs | 74 | 80/20 | 10k | 0.9737 exact / 0.4654 seq | **0.9609 exact** / 0.2692 seq | `NFMulA2R_2769003.out` |
| **nf_multiplication** — left | s × NF(w) → NF of the product | 5,994 pairs | 74 | 80/20 | 10k | 0.9678 exact / 0.4226 seq | **0.9517 exact** / 0.2600 seq | `NFMulA2L_2769004.out` |
| **shorter_curriculum** | curriculum len 10 → 14 → 18 → 22 on random-arm data | 4 exact-length CSVs | 10–22 | — | — | — | — | *no run log — code + data only* |
| **grokking** | grokking-dynamics demo notebook | — | — | — | — | — | — | *notebook, never run as a batch job* |
| **replication_2025** | 2025 binary trivial/non-trivial task, bidirectional | 129,300 words | 22 | 30/70 | — | — | — | *no run log — bridge from the 2025 arc* |

**Reading the extensions.** Three findings are worth pulling out.

1. *Globally reduced data is what makes the task easy.* `minimal_length` uses **every** reduced
   expression of every element up to length 16 and is solved perfectly — train and test, all four
   label variants. The reduced arm at length 16 scores 0.78 on the same model. The difference is
   global vs. merely local reducedness, not model capacity.
2. *Left converges roughly an order of magnitude faster than right.* All four left/right pairs
   end at 1.0000, but the epoch at which test exact-match **first** hits 1.0000 differs sharply:

   | Dataset | Left | Right | Ratio |
   |---|---|---|---|
   | minimal_length, descents | 1,100 | 12,100 | 11.0× |
   | minimal_length, ascents | 900 | 15,900 | 17.7× |
   | normal form, descents | 700 | 11,300 | 16.1× |
   | normal form, ascents | 800 | 4,700 | 5.9× |

   Left descents of a left-to-right causal read are available from the prefix directly; right
   descents are not, and the model needs far more optimization to find them. This holds across
   two different datasets and both label polarities, so it is a property of the task rather than
   of one dataset.
3. *Sequence-level tasks hit a 1-layer ceiling.* `nf_generation` reaches 0.89 per-token but
   **0.0000** whole-sequence under teacher forcing — not one held-out word is predicted correctly
   at every position. (That is a stricter bar than free-running generation, which is evaluated
   separately in the visualizer, not here.) `nf_multiplication` reaches 0.96 per-token but 0.27
   per-sequence, and its *train* sequence accuracy stalls at ~0.45 — a capacity ceiling rather
   than an optimization failure. Both are the natural place to try 2 layers.

---

## Configuration present, no results here

These have configs in the repo but no training log, because the run happened on a teammate's
machine or never happened:

- **`arms/random/`** — the random-word arm (repeats allowed). `config.py` (1 layer) and
  `config_2layer.py` (2 layers, `LNPre`, 100k epochs) are the hyperparameter-exploration configs.
  Results live with that arm's owner.
- **`arms/normal_form/right_descent/`** — Pranav's NF right-descent setup, including
  `NFdata.csv`. `config_exhaustive.py` is the exhaustive-BFS variant that *did* run (row above).
- **`extensions/shorter_curriculum/`**, **`extensions/grokking/`**,
  **`extensions/replication_2025/`** — as marked in the table.

The gap is deliberate and visible: this file records what can be sourced, not what is believed.

---

## Reproducing a number

```bash
cd 2026/arms/reduced/length16          # or any arm/extension folder
mkdir -p logs
sbatch transformer_job.sl
grep '^Epoch' logs/*.out | tail -1
```

Full instructions, including generating the dataset first, are in
[`REPLICATION.md`](REPLICATION.md).
