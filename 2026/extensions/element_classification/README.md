# Finite A₂ Element Classification

A different question from the rest of the repo: instead of predicting a prefix's *descent set*,
predict **which group element the prefix actually is**.

Finite A₂ = S₃ has exactly 6 elements, so at every prefix of a random word over its 2 generators
the model does a 6-way classification (ShortLex order: `e`, `1`, `2`, `12`, `21`, `121`). That
makes it **softmax cross-entropy over element IDs**, not the multi-label sigmoid used elsewhere —
hence [`../../shared/Transformer_classification.py`](../../shared/Transformer_classification.py).

| Data | Seq len | Split | Epochs | Test per-position | Test whole-word |
|---|---|---|---|---|---|
| 4,000 random words | 18 | 30/70 | 15k | **0.9981** | 0.9818 |

Source log: `FiniteA2Class_2769002.out`. Train accuracy is 1.0000.

This is the most direct test of the automaton hypothesis in the repo: to score 0.998 the model
*must* be tracking the current group element in its residual stream. The analysis notebook
([`../../shared/Analysis_classification.ipynb`](../../shared/Analysis_classification.ipynb))
includes a 6×6 confusion matrix and a PCA of `resid_post` colored by true element — if the model
implements the 6-state automaton, those states should separate into 6 clusters.

```bash
python build_element_dataset.py
mkdir -p logs && sbatch transformer_job.sl
```
