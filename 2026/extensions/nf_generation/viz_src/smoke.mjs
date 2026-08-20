/*
 * Drive the page headlessly.
 *
 *   node viz/smoke.mjs
 *
 * There is no browser on the cluster, so this runs model.js + geometry.js +
 * ui.js against a DOM stub built from the ids in template.html: enough to catch
 * a missing element, a typo'd handler or a thrown exception, and enough to check
 * the instrument actually produces the word the checkpoint produces.
 *
 * It is not a substitute for looking at the rendered page — it checks behaviour,
 * not layout.
 */
import { readFileSync } from "node:fs";
import vm from "node:vm";

const here = (n) => new URL(n, import.meta.url);
const read = (n) => readFileSync(here(n), "utf8");

let failures = 0;
const check = (ok, label, detail) => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${label}${detail ? " — " + detail : ""}`);
  if (!ok) failures++;
};

// ----------------------------------------------------------------- DOM stub

const template = read("template.html");
const ids = [...template.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
const drawn = [];

function makeEl(tag = "div") {
  const el = {
    tagName: tag,
    children: [],
    dataset: {},
    style: {},
    classes: new Set(),
    handlers: {},
    value: "",
    max: "",
    disabled: false,
    _text: "",
    _html: "",
    className: "",
    get textContent() { return this._text; },
    set textContent(v) { this._text = String(v); if (v === "") this.children = []; },
    get innerHTML() { return this._html; },
    set innerHTML(v) { this._html = String(v); },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(type, fn) { (this.handlers[type] ||= []).push(fn); },
    fire(type, ev) { for (const fn of this.handlers[type] || []) fn(ev || { target: this }); },
    classList: {
      add(...c) { c.forEach((x) => el.classes.add(x)); },
      remove(...c) { c.forEach((x) => el.classes.delete(x)); },
      contains(c) { return el.classes.has(c); },
      toggle(c, on) { (on === undefined ? !el.classes.has(c) : on) ? el.classes.add(c) : el.classes.delete(c); },
    },
    querySelector(sel) { return (el.kids ||= {})[sel] ||= makeEl(); },
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 700, height: 460 }),
  };
  return el;
}

const registry = Object.fromEntries(ids.map((id) => [id, makeEl()]));

// The canvas: record calls so we can assert something was actually drawn.
const canvas = registry["tiling"];
canvas.clientWidth = 700;
canvas.clientHeight = 460;
canvas.width = 700;
canvas.height = 460;
canvas.parentElement = makeEl();
canvas.parentElement.clientWidth = 700;
canvas.parentElement.clientHeight = 460;
canvas.getContext = () => new Proxy({}, {
  get: (t, k) => {
    if (k === "canvas") return canvas;
    if (!(k in t)) t[k] = (...a) => { drawn.push(k); return undefined; };
    return typeof t[k] === "function" ? t[k] : t[k];
  },
  set: () => true,
});

// The four probability rows and the quick-seed buttons.
const barRows = [["1", "1"], ["2", "2"], ["3", "3"], ["stop", "0"]].map(([g, cls]) => {
  const row = makeEl();
  row.dataset.g = g;
  row.dataset.cls = cls;
  return row;
});
const seedButtons = ["1", "2", "3", "121"].map((s) => {
  const b = makeEl("button");
  b.dataset.seed = s;
  return b;
});

const documentStub = {
  documentElement: makeEl(),
  getElementById: (id) => registry[id] || (registry[id] = makeEl()),
  createElement: (tag) => makeEl(tag),
  querySelectorAll: (sel) =>
    sel === ".bar-row" ? barRows : sel === "[data-seed]" ? seedButtons : [],
};

const rafQueue = [];
const sandbox = {
  console,
  document: documentStub,
  devicePixelRatio: 2,
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  getComputedStyle: () => ({ getPropertyValue: (n) => (n === "--trail-rgb" ? "19, 26, 27" : "#123456") }),
  requestAnimationFrame: (cb) => { rafQueue.push(cb); return rafQueue.length; },
  cancelAnimationFrame: () => {},
  ResizeObserver: class { observe() {} },
  MutationObserver: class { observe() {} },
  setTimeout: () => 0,           // suppress the autoplay kick-off
  setInterval: () => 0,
  clearInterval: () => {},
  Math, JSON, Number, String, Array, Object, Set, Map, Float32Array, Uint8Array,
  Uint16Array, Buffer, Infinity, NaN, isFinite, parseInt, parseFloat,
  PAYLOAD: JSON.parse(read("payload.json")),
  NF_WORDS: read("nf_words.txt"),
  VIZ_META: JSON.parse(read("../generation_viz.html").match(/var VIZ_META = (\{.*?\});/s)[1]),
};
// In a browser window === self === globalThis, which is how the UMD wrappers in
// model.js / geometry.js and the window.* reads in ui.js meet.
sandbox.window = sandbox;
sandbox.self = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

const drainRaf = () => {
  for (let i = 0; i < 500 && rafQueue.length; i++) rafQueue.shift()();
};

// ------------------------------------------------------------------ run it

console.log("Loading the page scripts against a DOM stub");
let loadError = null;
try {
  for (const f of ["model.js", "geometry.js", "ui.js"]) {
    vm.runInContext(read(f), sandbox, { filename: f });
  }
  drainRaf();
} catch (e) {
  loadError = e;
}
check(!loadError, "scripts run without throwing");
if (loadError) { console.error(loadError); process.exit(1); }

check(drawn.length > 0, "the canvas was drawn to", `${drawn.length} calls`);
check(registry["m-step"].textContent === "1/36", "starts on letter 1 of 36",
      registry["m-step"].textContent);
check(registry["colophon"].innerHTML.includes("float16"), "colophon states the export precision");
check(registry["prov-params"].textContent === "800,004", "parameter count shown",
      registry["prov-params"].textContent);

// Step to the end and read the word off the strip.
console.log("\nStepping through the whole rollout");
for (let i = 0; i < 40; i++) { registry["fwd"].fire("click"); drainRaf(); }
const word = registry["word-strip"].children.map((c) => c.textContent).join("");
const EXPECTED = "123121312131213121312131213121312131"; // logs/NFGenA2_2762307.out, seed [1]
check(word === EXPECTED, "greedy word matches the checkpoint's own rollout",
      word === EXPECTED ? word : `got ${word}`);
check(registry["m-step"].textContent === "36/36", "cursor stops at the cap",
      registry["m-step"].textContent);
check(registry["status-text"].textContent === "ShortLex normal form", "final word is legal",
      registry["status-text"].textContent);
check(registry["m-ell"].textContent === "36", "length ℓ equals the letter count",
      registry["m-ell"].textContent);
const values = barRows.map((r) => r.querySelector(".bar-value").textContent);
check(values.every((v) => /%$/.test(v)), "every bar shows a probability", values.join(" "));
const widths = barRows.map((r) => parseFloat(r.querySelector(".bar-fill").style.width));
check(Math.abs(widths.reduce((a, b) => a + b, 0) - 100) < 0.5, "bar widths sum to 100%",
      widths.map((w) => w.toFixed(1)).join(" + "));

// Seed handling.
console.log("\nControls");
registry["seed-input"].value = "1 2 1";
registry["seed-go"].fire("click");
drainRaf();
check(registry["word-strip"].children.length === 1, "new seed resets to the first letter");
check(!registry["seed-hint"].classes.has("error"), "valid seed accepted");
for (let i = 0; i < 40; i++) { registry["fwd"].fire("click"); drainRaf(); }
const w121 = registry["word-strip"].children.map((c) => c.textContent).join("");
check(w121.startsWith("121"), "seed letters are kept", w121.slice(0, 6));
// Seed 121 is one the model stops on by itself, at 35 letters (matching
// generate.py's own rollout in logs/NFGenA2_2762307.out).
const chips = registry["word-strip"].children.map((c) => c.textContent);
check(chips.length === 36 && chips[35] === "stop", "stops itself after 35 letters",
      `${chips.length} chips, last "${chips[chips.length - 1]}"`);
check(registry["source-caption"].textContent === "chosen by the model", "caption names the source",
      registry["source-caption"].textContent);

registry["seed-input"].value = "9";
registry["seed-go"].fire("click");
check(registry["seed-hint"].classes.has("error"), "invalid seed is refused");

// Sampling.
registry["seed-input"].value = "1";
registry["seed-go"].fire("click");
drainRaf();
registry["temp"].value = "10";
let tempError = null;
try { registry["temp"].fire("input", { target: registry["temp"] }); drainRaf(); }
catch (e) { tempError = e; }
check(!tempError, "temperature 1.0 rebuilds without throwing",
      tempError ? String(tempError.message) : registry["temp-val"].textContent);
check(registry["legend-note"].innerHTML.includes("sampled from"), "legend explains the tick marks");

console.log(failures === 0 ? "\nAll checks passed." : `\n${failures} CHECK(S) FAILED.`);
process.exit(failures === 0 ? 0 : 1);
