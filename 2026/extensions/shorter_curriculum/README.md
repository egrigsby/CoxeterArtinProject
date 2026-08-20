# Shorter-Words Curriculum

Does training on short words first and lengthening help? This trains on exact-length datasets in
a curriculum — **10 → 14 → 18 → 22** — rather than a single fixed length.

| File | Purpose |
|---|---|
| `Set_Generation_Shorter.py` | Builds the exact-length datasets. |
| `curriculum_data/exact_len_{10,14,18,22}.csv` | One dataset per curriculum stage. |
| `curriculum_data/all_exact_curriculum_data.csv` | All stages concatenated. |
| `Transformer_Shorter.py` | Curriculum training loop — a **modified** copy of the shared model, not the frozen one. |
| `Transformer_Shorter_len10_len14.py` | Two-stage (10 → 14) variant. |
| `config_shorter.py`, `config_shorter_len10_len14.py` | Their configs. |
| `Analysis_Shorter.ipynb`, `Analysis_Shorter_len10_len14.ipynb` | Their analysis notebooks. |

**No training log is in the repo**, so [`../../RESULTS.md`](../../RESULTS.md) lists this as
code-and-data only. Because the model here is modified rather than frozen, results from it are
not directly comparable to the three arms.

This is data from the **random** arm's style (repeats allowed), not the reduced arm's.
