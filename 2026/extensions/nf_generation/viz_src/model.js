/*
 * The trained model, in the browser.
 *
 * A faithful re-implementation of the forward pass of the 1-layer HookedTransformer
 * built by Transformer.py / config.py:
 *
 *   resid = W_E[tok_i] + W_pos[i]
 *   attn:  4 heads, d_head 64, causal, scores scaled by 1/sqrt(d_head)   (no ln1)
 *   resid += attn_out
 *   resid += W_out * relu(W_in * resid + b_in) + b_out                    (no ln2)
 *   logits = resid * W_U + b_U                                            (no ln_final)
 *
 * There are no LayerNorms because config.py sets NORMALIZATION = None, which is
 * what makes this transcription short enough to trust.
 *
 * Two shortcuts that are exact rather than approximate:
 *   - We run on the raw prefix with no padding. generate.rollout() pads to
 *     SEQUENCE_LENGTH and masks pad keys, but attention is causal and every pad
 *     sits after the query position, so it can never reach the last letter.
 *   - Only the last position gets an MLP and unembed. In a 1-layer model the
 *     keys/values come from resid_pre (embeddings only), so no other position's
 *     MLP output can influence the final logits.
 *
 * Loads in a browser (window.NFModel) and in node (module.exports), so
 * viz/verify.mjs checks the same code the page runs.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NFModel = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  // ---------------------------------------------------------------- decoding

  function b64ToBytes(b64) {
    if (typeof atob === "function") {
      const bin = atob(b64);
      const out = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(b64, "base64")); // node
  }

  // IEEE half -> double. (Float16Array is too new to depend on.)
  function halfToFloat(h) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x03ff;
    if (exp === 0) return sign * Math.pow(2, -14) * (frac / 1024);
    if (exp === 0x1f) return frac ? NaN : sign * Infinity;
    return sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }

  function decodeBuffer(payload) {
    const bytes = b64ToBytes(payload.data);
    const owned = new Uint8Array(bytes); // fresh buffer: guarantees alignment
    if (payload.dtype === "float32") return new Float32Array(owned.buffer);
    if (payload.dtype !== "float16") throw new Error("unknown dtype " + payload.dtype);
    const u16 = new Uint16Array(owned.buffer);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
    return out;
  }

  /** payload -> { cfg, T } where T.<name> is a flat Float32Array view. */
  function load(payload) {
    const flat = decodeBuffer(payload);
    const T = {};
    for (const name of Object.keys(payload.manifest)) {
      const { offset, shape } = payload.manifest[name];
      const size = shape.reduce((a, b) => a * b, 1);
      T[name] = flat.subarray(offset, offset + size);
    }
    return { cfg: payload.config, T, dtype: payload.dtype };
  }

  // ---------------------------------------------------------------- forward

  /**
   * Next-token logits after `tokens` (a non-empty array of generator ids 1..3).
   * Returns Float32Array(d_vocab_out) = scores for [STOP, 1, 2, 3].
   */
  function logits(model, tokens) {
    const { d_model: dm, d_head: dh, n_heads: nh, d_mlp: dmlp,
            d_vocab_out: dout, n_ctx } = model.cfg;
    const T = model.T;
    const L = tokens.length;
    if (L === 0) throw new Error("the model has no start token: seed with >= 1 letter");
    if (L > n_ctx) throw new Error("prefix longer than n_ctx=" + n_ctx);

    // resid_pre[j] = W_E[tok_j] + W_pos[j]
    const resid = new Float32Array(L * dm);
    for (let j = 0; j < L; j++) {
      const e = tokens[j] * dm, p = j * dm, o = j * dm;
      for (let d = 0; d < dm; d++) resid[o + d] = T.W_E[e + d] + T.W_pos[p + d];
    }

    const last = (L - 1) * dm;
    const attnOut = new Float32Array(dm);
    const q = new Float32Array(dh), k = new Float32Array(L * dh), v = new Float32Array(L * dh);
    const scores = new Float32Array(L);
    const scale = 1 / Math.sqrt(dh);

    for (let h = 0; h < nh; h++) {
      const wq = h * dm * dh, bq = h * dh; // W_*: [n_heads, d_model, d_head]

      // Query at the last position only.
      for (let a = 0; a < dh; a++) q[a] = T.b_Q[bq + a];
      for (let d = 0; d < dm; d++) {
        const r = resid[last + d], row = wq + d * dh;
        if (r !== 0) for (let a = 0; a < dh; a++) q[a] += r * T.W_Q[row + a];
      }

      // Keys and values at every position (causal: all j <= L-1 are visible).
      k.fill(0); v.fill(0);
      for (let j = 0; j < L; j++) {
        const jo = j * dh, ro = j * dm;
        for (let a = 0; a < dh; a++) { k[jo + a] = T.b_K[bq + a]; v[jo + a] = T.b_V[bq + a]; }
        for (let d = 0; d < dm; d++) {
          const r = resid[ro + d], row = wq + d * dh;
          if (r === 0) continue;
          for (let a = 0; a < dh; a++) {
            k[jo + a] += r * T.W_K[row + a];
            v[jo + a] += r * T.W_V[row + a];
          }
        }
      }

      // Attention pattern over the prefix.
      let max = -Infinity;
      for (let j = 0; j < L; j++) {
        let s = 0;
        const jo = j * dh;
        for (let a = 0; a < dh; a++) s += q[a] * k[jo + a];
        s *= scale;
        scores[j] = s;
        if (s > max) max = s;
      }
      let sum = 0;
      for (let j = 0; j < L; j++) { scores[j] = Math.exp(scores[j] - max); sum += scores[j]; }

      // z = pattern . v, then out += z . W_O[h]
      const z = new Float32Array(dh);
      for (let j = 0; j < L; j++) {
        const w = scores[j] / sum, jo = j * dh;
        for (let a = 0; a < dh; a++) z[a] += w * v[jo + a];
      }
      const wo = h * dh * dm; // W_O: [n_heads, d_head, d_model]
      for (let a = 0; a < dh; a++) {
        const za = z[a];
        if (za === 0) continue;
        const row = wo + a * dm;
        for (let d = 0; d < dm; d++) attnOut[d] += za * T.W_O[row + d];
      }
    }

    // resid_mid (last position only — nothing downstream reads the others).
    const x = new Float32Array(dm);
    for (let d = 0; d < dm; d++) x[d] = resid[last + d] + attnOut[d] + T.b_O[d];

    // MLP
    const pre = new Float32Array(dmlp);
    pre.set(T.b_in);
    for (let d = 0; d < dm; d++) {
      const xd = x[d];
      if (xd === 0) continue;
      const row = d * dmlp; // W_in: [d_model, d_mlp]
      for (let m = 0; m < dmlp; m++) pre[m] += xd * T.W_in[row + m];
    }
    for (let d = 0; d < dm; d++) x[d] += T.b_out[d];
    for (let m = 0; m < dmlp; m++) {
      const act = pre[m] > 0 ? pre[m] : 0; // relu (config.py: TYPE = "relu")
      if (act === 0) continue;
      const row = m * dm; // W_out: [d_mlp, d_model]
      for (let d = 0; d < dm; d++) x[d] += act * T.W_out[row + d];
    }

    // Unembed
    const out = new Float32Array(dout);
    for (let c = 0; c < dout; c++) out[c] = T.b_U[c];
    for (let d = 0; d < dm; d++) {
      const xd = x[d], row = d * dout; // W_U: [d_model, d_vocab_out]
      for (let c = 0; c < dout; c++) out[c] += xd * T.W_U[row + c];
    }
    return out;
  }

  // ---------------------------------------------------------------- sampling

  function softmax(scores, temperature) {
    const t = temperature === undefined ? 1 : temperature;
    const out = new Float32Array(scores.length);
    if (t <= 0) { // greedy: a point mass on the argmax
      let best = 0;
      for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
      out[best] = 1;
      return out;
    }
    let max = -Infinity;
    for (let i = 0; i < scores.length; i++) if (scores[i] / t > max) max = scores[i] / t;
    let sum = 0;
    for (let i = 0; i < scores.length; i++) { out[i] = Math.exp(scores[i] / t - max); sum += out[i]; }
    for (let i = 0; i < scores.length; i++) out[i] /= sum;
    return out;
  }

  function argmax(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return best;
  }

  /** Deterministic PRNG so a sampled word is reproducible from its seed number. */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function sampleFrom(probs, u) {
    let acc = 0;
    for (let i = 0; i < probs.length; i++) { acc += probs[i]; if (u < acc) return i; }
    return probs.length - 1;
  }

  /**
   * One generation step. Returns the model's own belief (`probs`, temperature 1),
   * the distribution actually sampled from (`sampled`), and the chosen token.
   */
  function step(model, tokens, temperature, rng) {
    const raw = logits(model, tokens);
    const probs = softmax(raw, 1);
    const t = temperature || 0;
    if (t <= 0) return { logits: raw, probs, sampled: null, choice: argmax(raw) };
    const sampled = softmax(raw, t);
    return { logits: raw, probs, sampled, choice: sampleFrom(sampled, rng()) };
  }

  /** Greedy rollout to STOP or the n_ctx cap — mirrors generate.rollout(). */
  function rolloutGreedy(model, seed) {
    const word = seed.slice();
    while (word.length < model.cfg.n_ctx) {
      const next = argmax(logits(model, word));
      if (next === 0) break; // STOP
      word.push(next);
    }
    return word;
  }

  // ------------------------------------------------------- the NF language

  /**
   * The ShortLex normal-form language as a membership test. It is prefix-closed,
   * so a plain set of the words doubles as the set of legal prefixes.
   */
  function buildLanguage(text) {
    const words = text.split("\n").filter(Boolean);
    const set = new Set(words);
    return {
      size: words.length,
      has: (s) => s === "" || set.has(s),
      /** Which of {1,2,3} keep a legal normal form (STOP is always legal). */
      legalNext: (s) => [1, 2, 3].map((g) => set.has(s + g)),
      words,
    };
  }

  return { load, logits, softmax, argmax, sampleFrom, step, rolloutGreedy,
           mulberry32, buildLanguage, halfToFloat };
});
