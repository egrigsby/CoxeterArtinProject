/*
 * The word as a walk on the Ã₂ alcove tiling.
 *
 * Affine A2 is the symmetry group of the triangular tiling of the plane. Each
 * group element is one triangle (alcove), the identity being the fundamental
 * one, and appending a letter g on the right reflects the current alcove across
 * its g-wall — so a word is a chain of triangles, each sharing an edge with the
 * one before it.
 *
 * An alcove is stored as its three vertices, vertex i belonging to generator
 * i+1 (the g-wall is the edge opposite vertex g-1). Vertices live in lattice
 * coordinates (a, b) meaning a*(1,0) + b*(1/2, sqrt3/2), and because the
 * triangles are equilateral, the foot of the altitude from a vertex is the
 * midpoint of the opposite edge — so reflecting vertex i over that edge is just
 *
 *     p[i] <- p[j] + p[k] - p[i]
 *
 * Integer arithmetic, no floating point: the vertex triple is an exact key for
 * the group element, which is what lets us look up lengths and normal forms and
 * spot a walk doubling back on itself.
 *
 * Loads in a browser (window.NFGeom) and in node (module.exports).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.NFGeom = api;
})(typeof self !== "undefined" ? self : globalThis, function () {
  "use strict";

  const S = Math.sqrt(3) / 2; // height of a unit lattice step in y

  /** The fundamental alcove: the identity element. */
  const IDENTITY = [[0, 0], [1, 0], [0, 1]];

  /** Reflect across the g-wall (g in 1..3): the alcove of w*s_g, given w's. */
  function step(alcove, g) {
    const i = g - 1, j = (i + 1) % 3, k = (i + 2) % 3;
    const out = [alcove[0].slice(), alcove[1].slice(), alcove[2].slice()];
    out[i] = [alcove[j][0] + alcove[k][0] - alcove[i][0],
              alcove[j][1] + alcove[k][1] - alcove[i][1]];
    return out;
  }

  /** Exact identity of the group element this alcove represents. */
  function key(alcove) {
    return alcove[0][0] + "," + alcove[0][1] + "|" +
           alcove[1][0] + "," + alcove[1][1] + "|" +
           alcove[2][0] + "," + alcove[2][1];
  }

  /** Lattice coordinates -> drawing coordinates. */
  function toXY(p) {
    return [p[0] + p[1] / 2, p[1] * S];
  }

  function centroid(alcove) {
    const [a, b, c] = alcove.map(toXY);
    return [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3];
  }

  /** The two endpoints of the g-wall (the edge the letter g reflects across). */
  function wall(alcove, g) {
    const i = g - 1;
    return [alcove[(i + 1) % 3], alcove[(i + 2) % 3]];
  }

  /** Every alcove visited by a word, starting from the identity. */
  function path(word) {
    const out = [IDENTITY];
    let cur = IDENTITY;
    for (const g of word) { cur = step(cur, g); out.push(cur); }
    return out;
  }

  /**
   * Walk every normal-form word once to get a map from alcove key to that
   * element's normal form. The length of the normal form is the element's
   * Coxeter length, so this also answers "how far is this alcove from e?".
   */
  function buildIndex(words) {
    const index = new Map([[key(IDENTITY), ""]]);
    for (const w of words) {
      let cur = IDENTITY;
      for (let i = 0; i < w.length; i++) {
        cur = step(cur, +w[i]);
        const k = key(cur);
        if (!index.has(k)) index.set(k, w.slice(0, i + 1));
      }
    }
    return index;
  }

  /**
   * The tiling's mirror lines over a drawing-coordinate box, as three families
   * of parallel lines (horizontal, +60 degrees, -60 degrees). Returned as
   * [x1, y1, x2, y2] segments spanning the box; the SVG viewBox does the
   * clipping.
   */
  function latticeLines(x0, y0, x1, y1) {
    const lines = [];
    const half = 1 / (2 * S);
    for (let c = Math.ceil(y0 / S); c <= Math.floor(y1 / S); c++) {
      lines.push([x0, c * S, x1, c * S]);
    }
    for (let c = Math.ceil(x0 - y1 * half); c <= Math.floor(x1 - y0 * half); c++) {
      lines.push([c + y0 * half, y0, c + y1 * half, y1]);
    }
    for (let c = Math.ceil(x0 + y0 * half); c <= Math.floor(x1 + y1 * half); c++) {
      lines.push([c - y0 * half, y0, c - y1 * half, y1]);
    }
    return lines;
  }

  return { IDENTITY, S, step, key, toXY, centroid, wall, path, buildIndex, latticeLines };
});
