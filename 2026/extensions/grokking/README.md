# Grokking

`Descent_Test_Transformer.ipynb` — a grokking-dynamics demo notebook, explored separately from
the main training loop. Despite the filename it is a Grokking demo (its own first cell says so),
originally written to run in Colab with a GPU runtime.

Kept because grokking is the phenomenon the long full-batch runs elsewhere in this repo are
watching for: the reduced arm trains to ~1.0 train accuracy within a few thousand epochs and then
runs for 50,000, which only makes sense if you are waiting for a delayed generalization
transition.

**Never run as a batch job** — there is no `config.py`, no `transformer_job.sl`, and no entry in
[`../../RESULTS.md`](../../RESULTS.md) beyond a row noting its absence.
