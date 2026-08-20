"""
Build the standalone generation visualizer: checkpoint -> self-contained HTML page.

The trained model is tiny (~800k parameters) and has no LayerNorms
(`NORMALIZATION = None`), so its forward pass is re-implemented in JavaScript
(`viz/model.js`) and the weights ride along inside the page. The result runs the
real model on arbitrary user input with no server, no cluster and no torch.

Usage (on a compute node, in /projects/expmmllab/CoxeterEnv):

  python build_viz.py                 # export weights + assemble generation_viz.html
  python build_viz.py --export-only   # just the payload + fixtures (for node verify.mjs)
  python build_viz.py --fp32          # ship float32 weights (bigger page, exact match)
  python build_viz.py --assemble-only # rebuild the page from an existing payload,
                                      # no checkpoint and no torch — the usual loop
                                      # when only the HTML/CSS/JS changed

Outputs
  viz/payload.json    packed weights, base64, + manifest      (build product)
  viz/nf_words.txt    the 1998 ShortLex normal-form words     (build product)
  viz/reference.json  PyTorch fixtures for viz/verify.mjs     (build product)
  generation_viz.html the shareable page                      (build product)

Nothing in the experiment folder is modified: the checkpoint, config.py,
Transformer.py, generate.py and the CSVs are read-only inputs.
"""

import argparse
import ast
import base64
import json
import math
import random
import re
import sys
from pathlib import Path

# Only config.py at import time (it is pure stdlib). torch, numpy, pandas and the
# project modules are imported inside the functions that need them, so
# --assemble-only runs anywhere python does — no env, no GPU node.
from config import *


def _torch():
    """Import torch and friends, with the Analysis.ipynb unpickling shim."""
    # Checkpoints were saved by a transformer_lens that pickled the config under
    # a module path newer installs renamed. We only want cached["model"], but
    # torch.load unpickles the whole dict.
    try:
        import transformer_lens.config.hooked_transformer_config as _htc
        sys.modules.setdefault("transformer_lens.HookedTransformerConfig", _htc)
    except ImportError:
        pass
    import torch  # noqa: F401
    return torch

VIZ_DIR = Path(__file__).resolve().parent / "viz"
OUT_HTML = Path(__file__).resolve().parent / "generation_viz.html"

# Every tensor the JS forward pass needs, in packing order.
# (export name, state-dict key, expected shape)
TENSORS = [
    ("W_E",   "embed.W_E",         (TOKEN_TYPES, DIM_MODEL)),
    ("W_pos", "pos_embed.W_pos",   (SEQUENCE_LENGTH, DIM_MODEL)),
    ("W_Q",   "blocks.0.attn.W_Q", (HEADS, DIM_MODEL, DIM_HEADS)),
    ("b_Q",   "blocks.0.attn.b_Q", (HEADS, DIM_HEADS)),
    ("W_K",   "blocks.0.attn.W_K", (HEADS, DIM_MODEL, DIM_HEADS)),
    ("b_K",   "blocks.0.attn.b_K", (HEADS, DIM_HEADS)),
    ("W_V",   "blocks.0.attn.W_V", (HEADS, DIM_MODEL, DIM_HEADS)),
    ("b_V",   "blocks.0.attn.b_V", (HEADS, DIM_HEADS)),
    ("W_O",   "blocks.0.attn.W_O", (HEADS, DIM_HEADS, DIM_MODEL)),
    ("b_O",   "blocks.0.attn.b_O", (DIM_MODEL,)),
    ("W_in",  "blocks.0.mlp.W_in", (DIM_MODEL, DIM_MLP)),
    ("b_in",  "blocks.0.mlp.b_in", (DIM_MLP,)),
    ("W_out", "blocks.0.mlp.W_out", (DIM_MLP, DIM_MODEL)),
    ("b_out", "blocks.0.mlp.b_out", (DIM_MODEL,)),
    ("W_U",   "unembed.W_U",       (DIM_MODEL, DIM_OUTPUT)),
    ("b_U",   "unembed.b_U",       (DIM_OUTPUT,)),
]

N_PREFIX_FIXTURES = 400     # sampled test prefixes checked logit-for-logit
N_ROLLOUT_FIXTURES = 40     # seeds whose greedy rollout must reproduce exactly


# ---------------------------------------------------------------------------
# Weight packing
# ---------------------------------------------------------------------------

def pack_weights(state_dict, dtype):
    """Concatenate the forward-pass tensors into one flat buffer + manifest."""
    import numpy as np
    manifest, chunks, offset = {}, [], 0
    for name, key, shape in TENSORS:
        if key not in state_dict:
            raise KeyError(f"{key} missing from checkpoint state dict "
                           f"(have: {sorted(state_dict)[:8]}...)")
        arr = state_dict[key].detach().cpu().float().numpy()
        if arr.shape != shape:
            raise ValueError(f"{key}: expected shape {shape}, got {arr.shape}")
        flat = np.ascontiguousarray(arr, dtype=np.float32).ravel().astype(dtype)
        manifest[name] = {"offset": offset, "shape": list(shape)}
        chunks.append(flat)
        offset += flat.size
    return manifest, np.concatenate(chunks)


def report_biases(state_dict):
    """Biases are frozen in build_model, so they should still be exactly zero."""
    nonzero = [key for name, key, _ in TENSORS
               if name.startswith("b_") and float(state_dict[key].abs().max()) != 0.0]
    if nonzero:
        print(f"  note: {len(nonzero)} bias tensor(s) are nonzero and are being "
              f"shipped as-is: {nonzero}")
    else:
        print("  biases all exactly zero (frozen, as expected) — shipped anyway")


# ---------------------------------------------------------------------------
# PyTorch reference fixtures
# ---------------------------------------------------------------------------

def torch_logits(model, prefixes, device1):
    """Logits at the last letter of each prefix, via the padded training idiom."""
    import torch
    from Transformer import create_attention_mask, register_pad_mask_hook
    n = len(prefixes)
    tokens = torch.zeros((n, SEQUENCE_LENGTH), dtype=torch.long)
    for r, p in enumerate(prefixes):
        tokens[r, :len(p)] = torch.tensor(p, dtype=torch.long)
    if device1 is not None:
        tokens = tokens.to(device1)
    model.reset_hooks()
    register_pad_mask_hook(model, create_attention_mask(tokens))
    with torch.no_grad():
        out = model(tokens)
    idx = torch.tensor([len(p) - 1 for p in prefixes], device=out.device)
    return out[torch.arange(n, device=out.device), idx].float().cpu().numpy()


def read_words(csv_name):
    import pandas as pd
    df = pd.read_csv(DATA_PATH / csv_name)
    return [[x for x in (int(t) for t in ast.literal_eval(w)) if x != 0]
            for w in df["word"]]


def build_fixtures(model, device1):
    from generate import rollout
    words = read_words(TEST_CSV)
    rng = random.Random(0)

    # Prefixes spread over every length that occurs, not just short ones.
    by_len = {}
    for w in words:
        for i in range(1, len(w) + 1):
            by_len.setdefault(i, []).append(tuple(w[:i]))
    prefixes = []
    per_len = max(1, N_PREFIX_FIXTURES // len(by_len))
    for length in sorted(by_len):
        pool = sorted(set(by_len[length]))
        prefixes += [list(p) for p in rng.sample(pool, min(per_len, len(pool)))]
    print(f"  {len(prefixes)} prefix fixtures over lengths "
          f"{min(map(len, prefixes))}..{max(map(len, prefixes))}")

    logits = torch_logits(model, prefixes, device1)

    seeds = sorted({tuple(w[:math.ceil(len(w) / 2)]) for w in words})
    seeds = [list(s) for s in rng.sample(seeds, min(N_ROLLOUT_FIXTURES, len(seeds)))]
    rolled = rollout(model, seeds, device1)
    print(f"  {len(seeds)} rollout fixtures")

    return {
        "prefixes": prefixes,
        "logits": [[float(x) for x in row] for row in logits],
        "rollouts": [{"seed": s, "word": w} for s, w in zip(seeds, rolled)],
    }


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

def build_meta(payload, n_nf_words):
    """The facts the page states about itself."""
    n_params = 0
    for spec in payload["manifest"].values():
        size = 1
        for dim in spec["shape"]:
            size *= dim
        n_params += size
    meta = {
        "dtype": payload["dtype"],
        "n_params": n_params,
        "n_nf_words": n_nf_words,
        "max_len": payload["config"]["n_ctx"],
    }
    # The colophon quotes the JS-vs-PyTorch parity numbers, so it can only quote
    # them once `node viz/verify.mjs` has actually produced them for this payload.
    verified_path = VIZ_DIR / "verified.json"
    if verified_path.exists():
        verified = json.loads(verified_path.read_text(encoding="utf-8"))
        if verified.get("dtype") == payload["dtype"]:
            meta["verified"] = verified
        else:
            print("  note: viz/verified.json is for a different dtype — re-run "
                  "node viz/verify.mjs; the page will omit the parity claim")
    else:
        print("  note: no viz/verified.json — run `node viz/verify.mjs` first if "
              "you want the parity numbers in the page colophon")
    return meta


def assemble(payload_json, nf_words, meta):
    template = (VIZ_DIR / "template.html").read_text(encoding="utf-8")
    parts = {
        "CSS": (VIZ_DIR / "style.css").read_text(encoding="utf-8"),
        "MODEL_JS": (VIZ_DIR / "model.js").read_text(encoding="utf-8"),
        "GEOMETRY_JS": (VIZ_DIR / "geometry.js").read_text(encoding="utf-8"),
        "UI_JS": (VIZ_DIR / "ui.js").read_text(encoding="utf-8"),
        "PAYLOAD": payload_json,
        "NF_WORDS": json.dumps(nf_words),
        "META": json.dumps(meta),
    }
    for key, value in parts.items():
        token = "{{" + key + "}}"
        if token not in template:
            raise KeyError(f"template.html is missing the {token} placeholder")
        template = template.replace(token, value)
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", template)
    if leftover:
        raise ValueError(f"unsubstituted placeholders remain: {sorted(set(leftover))}")
    OUT_HTML.write_text(template, encoding="utf-8")
    print(f"Wrote {OUT_HTML.name} ({OUT_HTML.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true",
                    help="ship float32 weights instead of float16 (bigger page)")
    ap.add_argument("--export-only", action="store_true",
                    help="write payload/fixtures but skip HTML assembly")
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild the page from the existing payload (no torch)")
    args = ap.parse_args()

    VIZ_DIR.mkdir(exist_ok=True)

    if args.assemble_only:
        payload_json = (VIZ_DIR / "payload.json").read_text(encoding="utf-8")
        nf_words = (VIZ_DIR / "nf_words.txt").read_text(encoding="utf-8")
        payload = json.loads(payload_json)
        assemble(payload_json, nf_words,
                 build_meta(payload, len(nf_words.split("\n"))))
        return

    # Re-exporting needs the experiment folder; a downloaded snapshot only has
    # the payload, so say so plainly rather than dying on an import.
    if not (Path(__file__).resolve().parent / "generate.py").exists():
        sys.exit("Re-exporting the weights needs the experiment folder "
                 "(generate.py, Transformer.py and workspace/_scratch/model.pth).\n"
                 "In a downloaded snapshot, use:  python3 build_viz.py --assemble-only")

    _torch()   # install the unpickling shim before anything reads the checkpoint
    import numpy as np
    from generate import load_model, shortlex_language

    dtype = np.float32 if args.fp32 else np.float16

    print(f"Loading {PTH_LOCATION} ...")
    model, device1 = load_model()
    state = model.state_dict()
    report_biases(state)

    manifest, buf = pack_weights(state, dtype)
    print(f"  packed {buf.size:,} parameters as {np.dtype(dtype).name} "
          f"({buf.nbytes / 1e6:.2f} MB raw, {buf.nbytes * 4 / 3 / 1e6:.2f} MB base64)")

    payload = {
        "dtype": np.dtype(dtype).name,
        "config": {
            "n_ctx": SEQUENCE_LENGTH, "n_layers": LAYERS, "n_heads": HEADS,
            "d_head": DIM_HEADS, "d_model": DIM_MODEL, "d_mlp": DIM_MLP,
            "d_vocab": TOKEN_TYPES, "d_vocab_out": DIM_OUTPUT, "act_fn": TYPE,
        },
        "manifest": manifest,
        "data": base64.b64encode(buf.tobytes()).decode("ascii"),
    }
    payload_json = json.dumps(payload)
    (VIZ_DIR / "payload.json").write_text(payload_json, encoding="utf-8")
    print(f"  wrote viz/payload.json ({len(payload_json) / 1e6:.2f} MB)")

    language = shortlex_language(SEQUENCE_LENGTH)
    nf_words = "\n".join("".join(str(g) for g in w) for w in sorted(language))
    (VIZ_DIR / "nf_words.txt").write_text(nf_words, encoding="utf-8")
    print(f"  wrote viz/nf_words.txt ({len(language)} words, "
          f"{len(nf_words) / 1e3:.1f} KB)")

    print("Building PyTorch fixtures ...")
    fixtures = build_fixtures(model, device1)
    fixtures["dtype"] = payload["dtype"]
    (VIZ_DIR / "reference.json").write_text(json.dumps(fixtures), encoding="utf-8")
    print("  wrote viz/reference.json")

    if args.export_only:
        print("Export only — run `node viz/verify.mjs`, then re-run without "
              "--export-only to assemble the page.")
        return

    assemble(payload_json, nf_words, build_meta(payload, len(language)))


if __name__ == "__main__":
    main()
