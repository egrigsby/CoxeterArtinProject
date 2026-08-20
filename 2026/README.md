# 2026 — Descent Sets in Coxeter Groups

The 2026 work asks a single question in many forms: **can a small transformer learn the right
descent set of every prefix of a group word, and if so, what algorithm does it implement?**

At each prefix `s₁…sᵢ` of a word, the model predicts which generators are descents of the group
element `s₁⋯sᵢ`. Attention is **causal**, so the prediction at position `i` sees exactly the
prefix ending at `i` — no label shifting is needed. A prefix can have several descents at once,
so this is **multi-label**: one independent sigmoid per generator, trained with masked binary
cross-entropy.

- **[`RESULTS.md`](RESULTS.md)** — every run: configuration, final metrics, and the log each
  number came from.
- **[`REPLICATION.md`](REPLICATION.md)** — how to reproduce any of them.
- **[`../Setup/README.md`](../Setup/README.md)** — environment.

---

## The three-arm study

The core experiment is a controlled comparison. **The model is frozen** — identical architecture,
optimizer, and seeds across all three arms — so the only thing that varies is how the training
words are drawn. Any difference in performance is attributable to the data.

| Arm | Words | Owner |
|---|---|---|
| [`arms/random/`](arms/random/) | uniformly random over the generators, repeats allowed | random arm |
| [`arms/reduced/`](arms/reduced/) | no *adjacent* repeats — locally reduced; most are not globally reduced | reduced arm |
| [`arms/normal_form/`](arms/normal_form/) | inverse-ShortLex normal form — one canonical word per group element | normal-form arm |

Do not change `shared/Transformer.py` to make one arm work. If a change seems necessary, it has
to happen for all three or the comparison is confounded — raise it with the team first.

The headline: at comparable word lengths the reduced arm tops out around **0.62** test
exact-match while the normal-form arm reaches **1.0000**. Global structure in the data, not model
capacity, is what makes the task learnable. See [`RESULTS.md`](RESULTS.md).

---

## Layout

```
2026/
├── RESULTS.md            every run, with sourced numbers
├── REPLICATION.md        the runbook
├── check_repo.py         the checks that keep both honest
│
├── shared/               the model — ONE copy, used by every run
│   ├── Transformer.py                  split-on-load (DATA_CSV + TRAINING_SPLIT)
│   ├── Transformer_presplit.py         fixed partition (TRAIN_CSV + TEST_CSV)
│   ├── Transformer_classification.py   cross-entropy variant
│   ├── Analysis*.ipynb                 one notebook per model variant
│   ├── descents.py                     descent sets via the geometric (Tits) representation
│   ├── transformer_job.sl              SLURM template
│   └── README.md                       full model + config reference
│
├── arms/                 the three-arm study (see table above)
├── extensions/           experiments outside the controlled comparison
└── reference/            supporting material, not part of the training pipeline
```

Each run folder holds only what makes that run different: a `config.py`, its dataset builder, and
a `transformer_job.sl`. Datasets and checkpoints are gitignored — regenerate them with the
builder.

---

## Extensions

| Folder | What it asks |
|---|---|
| [`extensions/minimal_length/`](extensions/minimal_length/) | Does *globally* reduced data make the task easy? (Yes — 1.0000 on all four label variants.) |
| [`extensions/affine_a3/`](extensions/affine_a3/) | Does it transfer to a bigger group? (Ã₃, 4 generators — 0.997 / 0.986.) |
| [`extensions/element_classification/`](extensions/element_classification/) | Can the model track *which* group element a prefix is, not just its descents? (Finite A₂ = S₃, 0.998.) |
| [`extensions/nf_generation/`](extensions/nf_generation/) | Can it *generate* a normal form rather than label one? Includes a browser visualizer that runs the trained model live. |
| [`extensions/nf_multiplication/`](extensions/nf_multiplication/) | Can it multiply — NF(w) × s → NF of the product? |
| [`extensions/shorter_curriculum/`](extensions/shorter_curriculum/) | Does a length curriculum (10 → 14 → 18 → 22) help? |
| [`extensions/grokking/`](extensions/grokking/) | Grokking dynamics, explored separately from the main loop. |
| [`extensions/replication_2025/`](extensions/replication_2025/) | The 2025 binary trivial/non-trivial task — the bridge from [`../2025/`](../2025/). |

---

## Reference

[`reference/`](reference/) holds material that supports the work without being part of the
training pipeline:

- `reference/descent_set_notebooks/` — hand-computed left/right descent and ascent sets for Ã₂ normal-form
  words, plus a C++ implementation. These predate and validate the closed-form logic in
  `shared/descents.py`.
- `reference/legacy_generators/` — the original string-based (`"abc"`) descent generators built on a
  minimal-root reflection table. Superseded by `descents.py`, kept because they are what the
  earliest datasets came from.
- `minroots.cpp` — minimal-roots computation from Fokko du Cloux's *Coxeter 3.0*, for reference.
- `reference/hyperbolic/` — left-descent calculation for a hyperbolic (non-affine) Coxeter group.

---

## Deliberate quirks — do not "fix" these

These look like bugs and are not. They are part of the frozen shared setup, and changing any of
them silently invalidates the three-arm comparison.

- **`scheduler.step()` is commented out.** `ReduceLROnPlateau` is constructed but the learning
  rate never moves. Intentional.
- **Training is full-batch.** For clean loss curves; interpretability depends on them. Do not
  reintroduce mini-batching.
- **Biases are frozen** (`requires_grad=False` on `b_*`), for interpretability.
- **`NORMALIZATION = None`** and **`TYPE = "relu"`**, to keep the computation graph simple but
  nonlinear.
- **Metrics are limited to loss, exact-match accuracy, and per-bit accuracy** by team agreement.
  No F1, no by-length breakdowns in the training loop — those belong in the analysis notebook.
