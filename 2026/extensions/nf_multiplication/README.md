# Normal-Form Multiplication

The hardest task in the repo, and the one that most directly targets research question (1):
given the normal form of an element and a generator, produce the **normal form of the product**.

- [`right/`](right/) — NF(w) × s → NF(ws)
- [`left/`](left/) — s × NF(w) → NF(sw)

Both are sequence-output tasks with cross-entropy over token IDs, `SEQUENCE_LENGTH = 74`
(input NF, separator, output NF), `TOKEN_TYPES = 7`. They use the local `Transformer.py` here,
not one of the `shared/` variants.

| Run | Data | Epochs | Train per-token | Test per-token | Train whole-seq | Test whole-seq |
|---|---|---|---|---|---|---|
| `right/` | 5,994 pairs, 80/20 | 10k | 0.9737 | **0.9609** | 0.4654 | **0.2692** |
| `left/` | same | 10k | 0.9678 | **0.9517** | 0.4226 | **0.2600** |

**The interesting number is the train one.** Whole-sequence accuracy stalls at ~0.45 on data the
model has seen — that is a capacity ceiling, not an optimization failure or a generalization gap.
A 1-layer model learns the length and descent rules governing the product but not the full
product. This is the clearest candidate in the repo for trying `LAYERS = 2`.

```bash
python build_multiplication_datasets.py   # writes left/ and right/ together
cd right && mkdir -p logs && sbatch transformer_job.sl
```
