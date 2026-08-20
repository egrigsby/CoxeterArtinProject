/*
 * The instrument: builds a trajectory once, then draws it.
 *
 * A trajectory is the whole word plus, for every prefix, the model's next-letter
 * distribution and which letter was taken. Computing it up front (at most 36
 * forward passes, a few hundred milliseconds) makes playing, stepping and
 * scrubbing pure replay, so nothing is ever re-sampled behind the viewer's back.
 */
(function () {
  "use strict";

  const M = window.NFModel, G = window.NFGeom;
  const $ = (id) => document.getElementById(id);

  const model = M.load(PAYLOAD);
  const lang = M.buildLanguage(NF_WORDS);
  const index = G.buildIndex(lang.words);
  const MAXLEN = model.cfg.n_ctx;
  const MIN_SPAN = 7;           // never zoom in closer than this many lattice units
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const canvas = $("tiling"), ctx = canvas.getContext("2d");
  const hover = $("hover");

  // ------------------------------------------------------------------ theme

  let THEME = {};
  function readTheme() {
    const s = getComputedStyle(document.documentElement);
    const v = (n) => s.getPropertyValue(n).trim();
    THEME = {
      gen: [null, v("--gen-1"), v("--gen-2"), v("--gen-3")],
      ink: v("--ink"),
      ink2: v("--ink-2"),
      muted: v("--muted"),
      hairline: v("--hairline"),
      surface: v("--surface"),
      sunken: v("--sunken"),
      trail: (a) => `rgba(${v("--trail-rgb")}, ${a})`,
    };
  }
  readTheme();
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    readTheme(); requestDraw();
  });
  new MutationObserver(() => { readTheme(); requestDraw(); })
    .observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // ------------------------------------------------------------- trajectory

  let traj = null;      // { word, seedLen, frames, alcoves, stopped }
  let cursor = 1;       // how many letters are on screen (1-based)
  let temperature = 0;
  let rngSeed = 7;
  let playing = false;
  let timer = null;

  function build(seed, temp, seedNo) {
    const rng = M.mulberry32(seedNo);
    const word = seed.slice();
    const alcoves = [G.IDENTITY];
    const frames = [];
    let stopped = false;

    for (let i = 1; i <= MAXLEN; i++) {
      alcoves.push(G.step(alcoves[i - 1], word[i - 1]));
      const prefix = word.slice(0, i);
      const str = prefix.join("");
      const raw = M.logits(model, prefix);
      const probs = M.softmax(raw, 1);
      const sampled = temp > 0 ? M.softmax(raw, temp) : null;

      let next;
      if (i < word.length) next = word[i];               // still inside the seed
      else if (word.length >= MAXLEN) next = null;       // length cap
      else {
        next = temp > 0 ? M.sampleFrom(sampled, rng()) : M.argmax(raw);
        if (next !== 0) word.push(next);
      }

      const inLang = lang.has(str);
      const ell = (index.get(G.key(alcoves[i])) || "").length;
      frames.push({
        i, probs, sampled, next,
        source: i < seed.length ? "seed" : "model",
        legal: lang.legalNext(str),
        inLang, ell, reduced: ell === i,
      });

      if (next === 0) { stopped = true; break; }
      if (next === null) break;
    }
    return { word, seedLen: seed.length, frames, alcoves, stopped };
  }

  function rebuild(seed, opts) {
    const keep = (opts && opts.keepCursor) ? cursor : 1;
    traj = build(seed, temperature, rngSeed);
    cursor = Math.min(Math.max(1, keep), traj.frames.length);
    $("scrub").max = String(traj.frames.length);
    render();
    snapCamera(opts && opts.snap);
    if (opts && opts.autoplay) play(true);
  }

  // ----------------------------------------------------------------- camera

  let camera = null, target = null, raf = null;

  function boxFor(n) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i <= n; i++) {
      for (const p of traj.alcoves[i]) {
        const [x, y] = G.toXY(p);
        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
    }
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    const pad = 1.1;
    const span = Math.max((x1 - x0) + 2 * pad, ((y1 - y0) + 2 * pad) * (w / h), MIN_SPAN);
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, span };
  }

  function snapCamera(hard) {
    if (!traj) return;
    target = boxFor(cursor);
    if (hard || !camera || reduceMotion) camera = Object.assign({}, target);
    requestDraw();
  }

  function requestDraw() {
    if (raf) return;
    raf = requestAnimationFrame(function step() {
      raf = null;
      if (target && camera) {
        const k = reduceMotion ? 1 : 0.17;
        let moving = false;
        for (const key of ["cx", "cy", "span"]) {
          const d = target[key] - camera[key];
          if (Math.abs(d) > 1e-3) { camera[key] += d * k; moving = true; }
          else camera[key] = target[key];
        }
        draw();
        if (moving) requestDraw();
      } else draw();
    });
  }

  // ---------------------------------------------------------------- drawing

  let dpr = 1;
  function fitCanvas() {
    dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
  }

  function projector() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const scale = w / camera.span;
    const fromXY = (x, y) =>
      [(x - camera.cx) * scale + w / 2, h / 2 - (y - camera.cy) * scale];
    return {
      scale,
      fromXY,
      to: (p) => { const [x, y] = G.toXY(p); return fromXY(x, y); },
      toWorld: (sx, sy) => [
        (sx - w / 2) / scale + camera.cx,
        camera.cy - (sy - h / 2) / scale,
      ],
    };
  }

  function tracePath(P, tri) {
    const a = P.to(tri[0]), b = P.to(tri[1]), c = P.to(tri[2]);
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(c[0], c[1]);
    ctx.closePath();
  }

  function draw() {
    if (!traj || !camera) return;
    fitCanvas();
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = THEME.sunken;
    ctx.fillRect(0, 0, w, h);

    const P = projector();
    const edge = P.scale;                        // one lattice unit in pixels
    const half = camera.span / 2, halfY = (h / w) * camera.span / 2;

    // the tiling's mirrors, faint
    ctx.strokeStyle = THEME.hairline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const L of G.latticeLines(camera.cx - half, camera.cy - halfY,
                                   camera.cx + half, camera.cy + halfY)) {
      const a = P.fromXY(L[0], L[1]), b = P.fromXY(L[2], L[3]);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();

    // alcoves already visited, oldest palest
    for (let i = 1; i <= cursor; i++) {
      const t = cursor > 1 ? i / cursor : 1;
      tracePath(P, traj.alcoves[i]);
      ctx.fillStyle = THEME.trail(0.05 + 0.13 * t);
      ctx.fill();
    }

    // the identity alcove
    tracePath(P, G.IDENTITY);
    ctx.fillStyle = THEME.trail(0.06);
    ctx.fill();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = THEME.muted;
    ctx.lineWidth = 1.25;
    ctx.stroke();
    ctx.setLineDash([]);
    if (edge > 26) {
      const cen = G.centroid(G.IDENTITY);
      const sc = P.fromXY(cen[0], cen[1]);
      ctx.fillStyle = THEME.muted;
      ctx.font = `500 ${Math.min(15, edge * 0.34)}px ui-monospace, Menlo, monospace`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("e", sc[0], sc[1]);
    }

    // every wall the word has crossed, in the colour of the letter that crossed it
    ctx.lineCap = "round";
    for (let i = 1; i <= cursor; i++) {
      const g = traj.word[i - 1];
      const [p, q] = G.wall(traj.alcoves[i - 1], g);
      const a = P.to(p), b = P.to(q);
      ctx.strokeStyle = THEME.gen[g];
      ctx.lineWidth = i === cursor ? 3 : 2;
      ctx.globalAlpha = i === cursor ? 1 : 0.55;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // where we are now
    const here = traj.alcoves[cursor];
    tracePath(P, here);
    ctx.fillStyle = THEME.trail(0.16);
    ctx.fill();
    ctx.strokeStyle = THEME.ink;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // the three walls available from here, labelled
    for (let g = 1; g <= 3; g++) {
      const [p, q] = G.wall(here, g);
      const a = P.to(p), b = P.to(q);
      ctx.strokeStyle = THEME.gen[g];
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      ctx.stroke();
      if (edge > 44) {
        const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        const cc = centroidScreen(P, here);
        const dx = mid[0] - cc[0], dy = mid[1] - cc[1];
        const len = Math.hypot(dx, dy) || 1;
        ctx.fillStyle = THEME.gen[g];
        ctx.font = `600 ${Math.min(14, edge * 0.22)}px ui-monospace, Menlo, monospace`;
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(String(g), mid[0] + (dx / len) * 11, mid[1] + (dy / len) * 11);
      }
    }

    // the move about to happen
    const frame = traj.frames[cursor - 1];
    if (frame && frame.next) {
      const nxt = G.step(here, frame.next);
      const from = centroidScreen(P, here), to = centroidScreen(P, nxt);
      const dx = to[0] - from[0], dy = to[1] - from[1];
      const len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len;
      const s = [from[0] + ux * len * 0.18, from[1] + uy * len * 0.18];
      const e = [from[0] + ux * len * 0.92, from[1] + uy * len * 0.92];
      ctx.strokeStyle = THEME.gen[frame.next];
      ctx.fillStyle = THEME.gen[frame.next];
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();
      const head = Math.min(9, edge * 0.14);
      ctx.beginPath();
      ctx.moveTo(e[0], e[1]);
      ctx.lineTo(e[0] - ux * head - uy * head * 0.55, e[1] - uy * head + ux * head * 0.55);
      ctx.lineTo(e[0] - ux * head + uy * head * 0.55, e[1] - uy * head - ux * head * 0.55);
      ctx.closePath();
      ctx.fill();
    }
  }

  function centroidScreen(P, tri) {
    const a = P.to(tri[0]), b = P.to(tri[1]), c = P.to(tri[2]);
    return [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3];
  }

  // ------------------------------------------------------------------ panel

  const fmtPct = (p) => (p * 100).toFixed(1) + "%";

  function render() {
    const frame = traj.frames[cursor - 1];
    const word = traj.word;

    // status
    const status = $("status"), text = $("status-text");
    if (frame.inLang) {
      status.dataset.state = "nf";
      text.textContent = "ShortLex normal form";
    } else if (frame.reduced) {
      status.dataset.state = "";
      text.textContent = "reduced, but not ShortLex-least";
    } else {
      status.dataset.state = "off";
      text.textContent = `not reduced — ℓ = ${frame.ell} of ${cursor}`;
    }

    $("m-step").textContent = `${cursor}/${traj.frames.length}`;
    $("m-ell").textContent = String(frame.ell);
    $("m-legal").textContent = `${frame.legal.filter(Boolean).length} of 3`;

    // the word so far
    const strip = $("word-strip");
    strip.textContent = "";
    for (let i = 0; i < cursor; i++) {
      const el = document.createElement("span");
      el.className = "tok" + (i < traj.seedLen ? " seeded" : "") + (i === cursor - 1 ? " now" : "");
      el.dataset.g = String(word[i]);
      el.textContent = String(word[i]);
      strip.appendChild(el);
    }
    if (frame.next === 0) {
      const el = document.createElement("span");
      el.className = "tok stop now";
      el.textContent = "stop";
      strip.appendChild(el);
    }

    // caption
    $("source-caption").textContent =
      frame.next === null ? "36-letter cap reached"
        : frame.source === "seed" ? "given by the seed"
          : temperature > 0 ? `sampled at T = ${temperature.toFixed(1)}`
            : "chosen by the model";
    $("stage-caption").textContent = `${cursor} of ${traj.frames.length} letters`;

    // bars
    for (const row of document.querySelectorAll(".bar-row")) {
      const cls = Number(row.dataset.cls);
      const p = frame.probs[cls];
      row.querySelector(".bar-fill").style.width = (p * 100).toFixed(2) + "%";
      row.querySelector(".bar-value").textContent = fmtPct(p);
      const tick = row.querySelector(".bar-tick");
      if (frame.sampled) {
        tick.style.display = "";
        tick.style.left = `calc(${Math.min(frame.sampled[cls] * 100, 99.4).toFixed(2)}% - 1px)`;
      } else tick.style.display = "none";
      const legal = cls === 0 ? frame.inLang : frame.legal[cls - 1];
      row.querySelector(".bar-flag").innerHTML = legal
        ? (cls === 0 ? "✓ can end" : "✓ legal")
        : '<span class="no">✗ not NF</span>';
      row.classList.toggle("chosen", frame.next === cls);
    }

    $("legend-note").innerHTML = frame.sampled
      ? `Bars are the model's own probabilities; the tick on each is the distribution actually sampled from at T = ${temperature.toFixed(1)}.`
      : "Bars are the model's own probabilities. A letter is <b>legal</b> when it keeps the word a ShortLex normal form.";

    $("scrub").value = String(cursor);
    $("scrub-val").textContent = `${cursor} / ${traj.frames.length}`;
    $("play").textContent = playing ? "Pause" : (cursor >= traj.frames.length ? "Replay" : "Play");
    $("back").disabled = cursor <= 1;
    $("fwd").disabled = cursor >= traj.frames.length;
  }

  function goto(n) {
    cursor = Math.min(Math.max(1, n), traj.frames.length);
    render();
    target = boxFor(cursor);
    requestDraw();
  }

  // --------------------------------------------------------------- playback

  function play(on) {
    playing = on === undefined ? !playing : on;
    if (playing && cursor >= traj.frames.length) cursor = 1;
    clearInterval(timer);
    if (playing) {
      const sps = Number($("speed").value);
      timer = setInterval(() => {
        if (cursor >= traj.frames.length) { play(false); return; }
        goto(cursor + 1);
      }, 1000 / sps);
    }
    render();
  }

  // --------------------------------------------------------------- controls

  function parseSeed(raw) {
    const s = raw.replace(/[\s,]/g, "");
    if (!s.length) return { error: "Type at least one letter — the model has no start token." };
    if (!/^[123]+$/.test(s)) return { error: "Letters must be 1, 2 or 3." };
    if (s.length > MAXLEN) return { error: `At most ${MAXLEN} letters.` };
    return { seed: [...s].map(Number) };
  }

  function applySeed(raw, autoplay) {
    const r = parseSeed(raw);
    const hint = $("seed-hint");
    if (r.error) {
      hint.textContent = r.error;
      hint.classList.add("error");
      return;
    }
    hint.classList.remove("error");
    hint.textContent = lang.has(r.seed.join(""))
      ? "This seed is itself a normal form."
      : "Note: this seed is not a normal form, so the model is off its training distribution.";
    play(false);
    rebuild(r.seed, { autoplay: autoplay, snap: true });
  }

  $("seed-go").addEventListener("click", () => applySeed($("seed-input").value, true));
  $("seed-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") applySeed($("seed-input").value, true);
  });
  for (const b of document.querySelectorAll("[data-seed]")) {
    b.addEventListener("click", () => {
      $("seed-input").value = b.dataset.seed;
      applySeed(b.dataset.seed, true);
    });
  }
  $("seed-random").addEventListener("click", () => {
    const w = lang.words[Math.floor(Math.random() * lang.words.length)];
    const s = w.slice(0, Math.max(1, Math.ceil(w.length / 2)));
    $("seed-input").value = s;
    applySeed(s, true);
  });

  $("play").addEventListener("click", () => play());
  $("back").addEventListener("click", () => { play(false); goto(cursor - 1); });
  $("fwd").addEventListener("click", () => { play(false); goto(cursor + 1); });
  $("reset").addEventListener("click", () => { play(false); goto(1); snapCamera(true); });
  $("scrub").addEventListener("input", (e) => { play(false); goto(Number(e.target.value)); });
  $("speed").addEventListener("input", (e) => {
    $("speed-val").textContent = `${e.target.value} / s`;
    if (playing) play(true);
  });

  $("temp").addEventListener("input", (e) => {
    temperature = Number(e.target.value) / 10;
    $("temp-val").textContent = temperature === 0 ? "greedy" : "T = " + temperature.toFixed(1);
    $("reroll").disabled = temperature === 0;
    $("rng-val").textContent = temperature === 0 ? "deterministic" : "sample #" + rngSeed;
    play(false);
    rebuild(traj.word.slice(0, traj.seedLen), { keepCursor: true });
  });

  $("reroll").addEventListener("click", () => {
    rngSeed = 1 + Math.floor(Math.random() * 9999);
    $("rng-val").textContent = "sample #" + rngSeed;
    play(false);
    rebuild(traj.word.slice(0, traj.seedLen), { autoplay: true });
  });

  // ------------------------------------------------------------------ hover

  canvas.addEventListener("mousemove", (ev) => {
    if (!traj || !camera) return;
    const r = canvas.getBoundingClientRect();
    const P = projector();
    const [wx, wy] = P.toWorld(ev.clientX - r.left, ev.clientY - r.top);
    let hit = -1;
    for (let i = cursor; i >= 0; i--) {
      if (inTriangle([wx, wy], traj.alcoves[i])) { hit = i; break; }
    }
    if (hit < 0) { hover.classList.remove("on"); return; }
    const prefix = traj.word.slice(0, hit).join("") || "e";
    const ell = (index.get(G.key(traj.alcoves[hit])) || "").length;
    hover.innerHTML = hit === 0
      ? "<b>identity</b> e"
      : `<b>letter</b> ${hit} &nbsp; <b>ℓ</b> ${ell}<br>${prefix}`;
    hover.style.left = Math.min(ev.clientX - r.left + 14, r.width - 220) + "px";
    hover.style.top = Math.max(6, ev.clientY - r.top - 46) + "px";
    hover.classList.add("on");
  });
  canvas.addEventListener("mouseleave", () => hover.classList.remove("on"));

  function inTriangle(p, tri) {
    const [a, b, c] = tri.map(G.toXY);
    const d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    const l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d;
    const l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d;
    return l1 >= 0 && l2 >= 0 && l1 + l2 <= 1;
  }

  // ------------------------------------------------------------------ start

  new ResizeObserver(() => snapCamera(true)).observe(canvas.parentElement);

  $("prov-params").textContent = VIZ_META.n_params.toLocaleString();
  $("reroll").disabled = true;
  $("rng-val").textContent = "deterministic";

  const v = VIZ_META.verified;
  if (v && v.generation) {
    const g = v.generation;
    $("stop-note").innerHTML =
      `A word ends when the model emits <code>stop</code>, or when it runs into the ${g.max_len}-letter ` +
      `cap that the training words respect. Seeded with the first half of each of the ` +
      `${g.seeds} normal forms, it stops on its own <b>${g.stopped}</b> times and reaches the cap the ` +
      `other <b>${g.capped}</b>, writing words of ${g.min_len} to ${g.max_len} letters — every one of ` +
      `them a normal form.`;
  }
  $("colophon").innerHTML =
    `Weights exported from <code>workspace/_scratch/model.pth</code> at ${VIZ_META.dtype} ` +
    `(${VIZ_META.n_params.toLocaleString()} values) and run in your browser — no server, no upload. ` +
    (v ? `Checked against PyTorch on ${v.prefixes} test prefixes: identical top-1 choice on every one, ` +
         `largest probability deviation ${v.max_prob_diff.toFixed(4)}, and all ${v.rollouts} greedy ` +
         `rollouts reproduce the reference exactly. ` : "") +
    `Legality is checked against all ${VIZ_META.n_nf_words.toLocaleString()} normal-form words of ` +
    `length ≤ ${VIZ_META.max_len}, rebuilt by the same routine that made the training set. ` +
    `Which wall of the triangle belongs to which generator is a labelling convention — all three ` +
    `generators of Ã₂ are interchangeable (every m<sub>ij</sub> = 3).`;

  rebuild([1], { snap: true });
  setTimeout(() => play(true), 700);
})();
