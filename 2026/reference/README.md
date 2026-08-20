# Reference Material

Supporting work that is **not part of the training pipeline**. Nothing here is imported by a
model or a builder; it exists to explain, validate, or historically situate the code that is.

---

## `descent_set_notebooks/`

Hand-worked left/right descent and ascent set computations for Ã₂ normal-form words, following
the pseudocode in Casselman's treatment of the minimal-root automaton. These **predate and
validate** the closed-form logic now in [`../shared/descents.py`](../shared/descents.py) — if you
want to convince yourself the descent labels are right, this is where to look.

| File | Contents |
|---|---|
| `LeftAscentSet.ipynb` | The minimal-root algorithm written out from the pseudocode, with `t`, `i`, `k`, and the extended minimal root `λ` as explicit state. |
| `RightDescentSet.ipynb` / `RightAscentSet.ipynb` | The right-hand counterparts, computed by reading the word right-to-left. Both notebooks note that this is **much** slower than the left versions. |
| `Pranav_LeftDescentSetCalculation2.ipynb` | An independent left-descent implementation. |
| `Pranav_CPlusPlusLeftDescentCode.cpp` | The same, in C++. |

> The left-is-easier-than-right asymmetry these notebooks remark on informally shows up
> quantitatively in the trained models too: left-descent runs converge roughly an order of
> magnitude faster than right-descent ones on identical data. See
> [`../RESULTS.md`](../RESULTS.md).

---

## `legacy_generators/`

The original dataset generators, all working over the string alphabet `{a, b, c}` with a
**minimal-root reflection table** rather than the geometric (Tits) representation. Superseded by
`shared/descents.py`, kept because the earliest datasets came from them and because
[`../arms/random/Set_Generation(Right_Descent).py`](../arms/random/) is a direct descendant still
in use.

| File | What it generates |
|---|---|
| `Pranav_Right_Descent.py` | The original: right descent sets for random words. |
| `Right_Descent_Amend.py` | Same, with CSV output and `INSTANCES` / `SEQUENCE_LENGTH` knobs — the direct ancestor of the random arm's generator. |
| `dataset_no_repeats.py` | The no-adjacent-repeat variant — ancestor of the reduced arm. |
| `Dataset_smallest_element.py` | Picks the smallest element per class — ancestor of the normal-form arm. |

The last two had **no file extension** before this reorganization, which is why they were easy to
miss. They are ordinary Python scripts.

---

## `minroots.cpp`

Minimal-roots computation from **Fokko du Cloux's Coxeter 3.0** (2002), included verbatim for
reference. The minimal-root machinery is what makes descent computation a finite-state problem in
the first place, which is the whole premise of research question (2) in the
[root README](../../README.md) — describing these algorithms as finite state automata.

Not compiled or called by anything in this repo.

---

## `hyperbolic/`

`Hyperbolic_LeftDescentSetCalculation2.ipynb` — the left-descent calculation for a **hyperbolic**
(non-affine) Coxeter group. All the trained models so far are on affine groups (Ã₂, Ã₃); this is
the exploratory step toward the hyperbolic case, where the Coxeter matrix has entries that make
the geometric representation indefinite.
