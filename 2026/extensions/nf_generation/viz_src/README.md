# generation_viz — the trained model as a shareable web page

`../generation_viz.html` is a single self-contained file that runs **this
experiment's trained model in the browser**. Open it by double-clicking, mail it,
serve it anywhere — it makes no network requests. It is also published as a
Claude artifact at
<https://claude.ai/code/artifact/7cb2dc93-b3bf-40db-8b7d-fae0f531e920>
(private until shared from the page's share menu).

The page shows a word being generated two ways at once: the model's next-letter
probabilities with the legal continuations marked, and the walk the word traces
across the Ã₂ alcove tiling.

## Rebuilding

The weights and the normal-form language have to be exported from the checkpoint
once; after that, editing the HTML/CSS/JS only needs the assemble step, which
runs anywhere Python does.

```bash
# 1. export — needs the checkpoint and the conda env, so run it on a compute node
srun --partition=short --time=00:30:00 --mem=24g --cpus-per-task=4 bash -lc '
  module purge; module use /m31/modulefiles/static; module load miniconda
  conda activate /projects/expmmllab/CoxeterEnv
  python build_viz.py --export-only'

# 2. verify the JavaScript really is the checkpoint  (~90 s, login node is fine)
node viz/verify.mjs

# 3. assemble the page  (no torch, no GPU)
python3 build_viz.py --assemble-only

# 4. behavioural smoke test of the page itself
node viz/smoke.mjs
```

Steps 3–4 are the whole loop when only the interface changed.

## Files

| file | role |
|---|---|
| `../build_viz.py` | checkpoint → `payload.json` + `nf_words.txt` + `reference.json`, then assembles the page |
| `model.js` | the forward pass in JavaScript — loads in both the browser and node |
| `geometry.js` | the alcove walk, in exact integer lattice arithmetic |
| `ui.js`, `style.css`, `template.html` | the page |
| `verify.mjs` | four gates against PyTorch; writes `verified.json`, which the page quotes |
| `smoke.mjs` | drives `ui.js` against a DOM stub — catches broken handlers, not bad layout |
| `payload.json`, `nf_words.txt`, `reference.json`, `verified.json` | build products (~2 MB total) |

## Why the JavaScript can be trusted

`config.py` sets `NORMALIZATION = None`, so there are no LayerNorms and the whole
forward pass is embeddings → attention → relu MLP → unembed. Two shortcuts in
`model.js` are exact, not approximations:

- it runs on the raw prefix with no padding, because attention is causal and
  every pad key sits after the query position;
- only the last position gets an MLP and unembed, because in a 1-layer model the
  keys and values come from the embeddings alone.

Weights ship at **float16** (800,004 values, 1.6 MB). `verify.mjs` gates that
choice: identical top-1 on all 381 sampled test prefixes, largest probability
deviation 0.0011, and all 40 reference greedy rollouts reproduced exactly. If a
future checkpoint fails those gates, rebuild with `--fp32` rather than relaxing
the tolerance.

## Things worth knowing

- The LM has **no start token**, so generation needs a seed of at least one letter.
- Greedy generation ends with `stop` on its own for 356 of the 513 half-word
  seeds and hits the 36-letter cap for the other 157, writing words of 28–36
  letters — all 513 of them normal forms (`verify.mjs` gate 4).
- Which wall of the triangle belongs to which generator is a labelling
  convention; all three generators of Ã₂ are interchangeable.
- `payload.json` and `generation_viz.html` are ~2 MB build products. `Runs/` is a
  git repo and this folder is currently untracked — add an ignore rule before
  committing it if you don't want them in history.
