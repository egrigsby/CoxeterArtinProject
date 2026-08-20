/*
 * Proof that the JavaScript model is the same model as the checkpoint.
 *
 *   node viz/verify.mjs            (after `python build_viz.py --export-only`)
 *
 * Gate 1  logit parity   — JS logits vs PyTorch logits on sampled test prefixes,
 *                          with 100% argmax agreement required.
 * Gate 2  rollout parity — JS greedy rollout reproduces generate.rollout()'s word.
 * Gate 3  geometry       — the alcove walk is a real walk on the tiling, and the
 *                          alcove -> normal form index round-trips.
 *
 * Exits nonzero on any failure, so it can gate a rebuild.
 */
import { readFileSync, writeFileSync } from "node:fs";
import NFModel from "./model.js";
import NFGeom from "./geometry.js";

const here = (name) => new URL(name, import.meta.url);
const read = (name) => readFileSync(here(name), "utf8");

const payload = JSON.parse(read("payload.json"));
const reference = JSON.parse(read("reference.json"));
const lang = NFModel.buildLanguage(read("nf_words.txt"));

const FP16 = payload.dtype === "float16";
const LOGIT_TOL = FP16 ? 0.05 : 2e-3;
const PROB_TOL = FP16 ? 0.01 : 1e-4;

let failures = 0;
function check(ok, label, detail) {
  console.log(`${ok ? "  PASS" : "  FAIL"}  ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures++;
}

console.log(`payload: ${payload.dtype}, ${Object.keys(payload.manifest).length} tensors, ` +
            `${(payload.data.length / 1e6).toFixed(2)} MB base64`);
console.log(`language: ${lang.size} normal-form words\n`);

const t0 = Date.now();
const model = NFModel.load(payload);
console.log(`loaded weights in ${Date.now() - t0} ms\n`);

// ---------------------------------------------------------------- gate 1
console.log(`Gate 1 — logit parity over ${reference.prefixes.length} test prefixes`);
let maxLogitDiff = 0, maxProbDiff = 0, argmaxMismatch = 0, nonFinite = 0;
reference.prefixes.forEach((prefix, r) => {
  const mine = NFModel.logits(model, prefix);
  const theirs = reference.logits[r];
  for (let c = 0; c < theirs.length; c++) {
    if (!Number.isFinite(mine[c])) nonFinite++;
    maxLogitDiff = Math.max(maxLogitDiff, Math.abs(mine[c] - theirs[c]));
  }
  const p = NFModel.softmax(mine, 1);
  const q = NFModel.softmax(Float32Array.from(theirs), 1);
  for (let c = 0; c < q.length; c++) maxProbDiff = Math.max(maxProbDiff, Math.abs(p[c] - q[c]));
  if (NFModel.argmax(mine) !== NFModel.argmax(theirs)) argmaxMismatch++;
});
check(nonFinite === 0, "all logits finite", `${nonFinite} bad values`);
check(argmaxMismatch === 0, "argmax agreement",
      `${reference.prefixes.length - argmaxMismatch}/${reference.prefixes.length}`);
check(maxLogitDiff < LOGIT_TOL, "max |Δ logit|",
      `${maxLogitDiff.toExponential(2)} (tol ${LOGIT_TOL})`);
check(maxProbDiff < PROB_TOL, "max |Δ probability|",
      `${maxProbDiff.toExponential(2)} (tol ${PROB_TOL})`);

// ---------------------------------------------------------------- gate 2
console.log(`\nGate 2 — greedy rollout parity over ${reference.rollouts.length} seeds`);
let rolloutMismatch = 0, illegal = 0;
for (const { seed, word } of reference.rollouts) {
  const mine = NFModel.rolloutGreedy(model, seed);
  if (mine.join(",") !== word.join(",")) {
    rolloutMismatch++;
    if (rolloutMismatch <= 3) console.log(`    seed ${seed}\n      js    ${mine}\n      torch ${word}`);
  }
  if (!lang.has(mine.join(""))) illegal++;
}
check(rolloutMismatch === 0, "rollouts identical to generate.rollout()",
      `${reference.rollouts.length - rolloutMismatch}/${reference.rollouts.length}`);
check(illegal === 0, "rollouts are legal normal forms",
      `${reference.rollouts.length - illegal}/${reference.rollouts.length}`);

// ---------------------------------------------------------------- gate 3
console.log("\nGate 3 — alcove-walk geometry");
const index = NFGeom.buildIndex(lang.words);
const sharesEdge = (a, b) => {
  const bk = b.map((p) => p.join(","));
  return a.filter((p) => bk.includes(p.join(","))).length === 2;
};
let notAdjacent = 0, revisited = 0, badIndex = 0;
for (const w of lang.words) {
  const walk = NFGeom.path([...w].map(Number));
  for (let i = 1; i < walk.length; i++) if (!sharesEdge(walk[i - 1], walk[i])) notAdjacent++;
  const keys = new Set(walk.map(NFGeom.key));
  if (keys.size !== walk.length) revisited++;
  if (index.get(NFGeom.key(walk[walk.length - 1])) !== w) badIndex++;
}
check(notAdjacent === 0, "consecutive alcoves share a wall", `${notAdjacent} bad steps`);
check(revisited === 0, "reduced words never revisit an alcove", `${revisited} words revisit`);
check(badIndex === 0, "alcove -> normal form round-trips", `${badIndex} mismatches`);
check(index.size === lang.size + 1, "index covers every element + identity",
      `${index.size} vs ${lang.size + 1}`);

// A non-reduced word must double back — the property the page relies on to show
// off-language sampling honestly.
const back = NFGeom.path([1, 1]);
check(NFGeom.key(back[0]) === NFGeom.key(back[2]), "s*s returns to the identity alcove");

// ---------------------------------------------------------------- gate 4
// How generation actually behaves, over every half-word seed the language
// offers. The page describes this in words, so it must not be guessed at.
console.log("\nGate 4 — greedy generation over every half-word seed");
const seeds = new Set(lang.words.map((w) => w.slice(0, Math.ceil(w.length / 2))));
let stopped = 0, capped = 0, legalGen = 0, minLen = Infinity, maxLen = 0;
for (const s of seeds) {
  const w = NFModel.rolloutGreedy(model, [...s].map(Number));
  if (w.length < model.cfg.n_ctx) stopped++; else capped++;
  if (lang.has(w.join(""))) legalGen++;
  minLen = Math.min(minLen, w.length);
  maxLen = Math.max(maxLen, w.length);
}
check(legalGen === seeds.size, "every greedy generation is a normal form",
      `${legalGen}/${seeds.size}`);
console.log(`  INFO  ${stopped} end with stop, ${capped} reach the ${model.cfg.n_ctx}-letter ` +
            `cap; lengths ${minLen}–${maxLen}`);

// The page quotes these numbers in its colophon, so they are written out rather
// than transcribed by hand. A failed run leaves no claim behind.
if (failures === 0) {
  writeFileSync(here("verified.json"), JSON.stringify({
    dtype: payload.dtype,
    prefixes: reference.prefixes.length,
    argmax_agreement: 1,
    max_logit_diff: Number(maxLogitDiff.toPrecision(3)),
    max_prob_diff: Number(maxProbDiff.toPrecision(3)),
    rollouts: reference.rollouts.length,
    generation: { seeds: seeds.size, stopped, capped, min_len: minLen, max_len: maxLen },
  }, null, 2) + "\n");
  console.log("\nAll checks passed — wrote viz/verified.json");
} else {
  console.log(`\n${failures} CHECK(S) FAILED.`);
}
process.exit(failures === 0 ? 0 : 1);
