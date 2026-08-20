# CoxeterArtinProject

Exploring how transformers and other architectures handle the word problem in Coxeter and Artin
groups.

## Questions guiding us

**(0)** What is a Coxeter group? What are the different ways of presenting a Coxeter group
algebraically? What do these algebraic presentations represent geometrically?

**(1)** What are the key combinatorial and geometric ways of understanding how to do
multiplication in a fixed normal form (specifically, ReverseShortLex normal form, as described in
Casselman's work) for the standard presentation of Coxeter groups?

**(2)** How do we explicitly describe any of these algorithms as finite state automata?

**(3)** Can we find combinatorial signatures (circuits) of these finite state automata in ML
models that have learned to do multiplication in Coxeter groups?

**(4)** Can we find geometric signatures (Cayley graphs) of these finite state automata in ML
models that have learned to do multiplication in Coxeter groups?

---

## Where to start

| If you want to… | Go to |
|---|---|
| Set up an environment | [`Setup/README.md`](Setup/README.md) |
| See what has been done and how it scored | [`2026/RESULTS.md`](2026/RESULTS.md) |
| Reproduce a run | [`2026/REPLICATION.md`](2026/REPLICATION.md) |
| Understand the model and every config option | [`2026/shared/README.md`](2026/shared/README.md) |
| Run a job on the cluster | [`RUNNING_ON_ANDROMEDA.md`](RUNNING_ON_ANDROMEDA.md) |

## Layout

```
Setup/          environment: pinned requirements + verification notebook
2026/           current work — descent-set prediction (see 2026/README.md)
  shared/         the frozen model, one copy, used by every run
  arms/           the three-arm study: random / reduced / normal_form
  extensions/     experiments outside the controlled comparison
  reference/      supporting material, not part of the training pipeline
2025/           last year's arc — RNN / LSTM / transformer on the binary
                trivial-vs-nontrivial word problem, plus the group-generation code
                and datasets it used
```

## The 2026 work in one paragraph

A 1-layer `TransformerLens` `HookedTransformer` with causal attention predicts, at every prefix
of a group word, that prefix's **right descent set** — one independent sigmoid per generator,
trained with masked binary cross-entropy. The team runs the **same frozen model** on three
dataset types (random words, reduced words, normal-form words) so that any difference in
performance is attributable to the data alone. The finding so far: normal-form and globally
reduced data are solved perfectly, while merely locally reduced data degrades sharply with word
length. The longer-range goal is mechanistic — finding combinatorial and geometric signatures of
the finite-state descent algorithms inside the trained weights. Details in
[`2026/README.md`](2026/README.md); numbers in [`2026/RESULTS.md`](2026/RESULTS.md).
