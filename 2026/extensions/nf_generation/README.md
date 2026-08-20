# Normal-Form Generation

Instead of labelling a normal form, **generate** one: a next-token language model over ShortLex
normal-form words of Ã₂, trained on the same 1,998 exhaustive words as the normal-form arm
(30/70 split, `SEQUENCE_LENGTH = 36`). Cross-entropy over next-token IDs, so it uses
[`../../shared/Transformer_classification.py`](../../shared/Transformer_classification.py).

| Epochs | Train per-token | Test per-token | Train whole-seq | Test whole-seq |
|---|---|---|---|---|
| 10k | 0.9091 | **0.8933** | 0.0050 | **0.0000** |

Source log: `NFGenA2_2762307.out`.

Per-token accuracy is respectable; whole-sequence accuracy under teacher forcing is **zero** on
held-out words — not one is reproduced correctly at every position. Note that this is a stricter
bar than free-running generation, which is what the visualizer below actually exercises.

## The visualizer

`generation_viz.html` is a **single self-contained page** that runs the trained model live in the
browser: fp16 weight export, a hand-written JS forward pass, and exact integer alcove geometry
for the Ã₂ Cayley graph. Open it in any browser — no server, no dependencies.

`viz_src/` holds the sources it is built from (`geometry.js`, `model.js`, `ui.js`, `style.css`,
`template.html`, plus `verify.mjs` / `smoke.mjs`); `build_viz.py` turns a checkpoint into the
payload and assembles the page. The generated payload itself is not committed — rebuild it from a
checkpoint. See [`viz_src/README.md`](viz_src/README.md).

```bash
python build_nf_lm_dataset.py     # inherits words + split from arms/normal_form/left_descent/
mkdir -p logs && sbatch transformer_job.sl
```
