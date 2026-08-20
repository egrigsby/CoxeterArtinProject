# Environment Setup

Everything in this repo runs on **Python 3.11** with `torch 2.7.1+cu128` and
`transformer_lens 3.4.0`. There are two ways to get there: use the lab's shared environment on
Andromeda (what everyone on the team actually does), or build your own from
[`requirements.txt`](requirements.txt).

---

## Option A — the shared lab environment (Andromeda)

The environment already exists at `/projects/expmmllab/CoxeterEnv`. You do not install anything;
you activate it.

SSH into `andromeda.bc.edu`, then in a **bash** shell:

```bash
interactive -G 1                              # get onto a compute node with a GPU first
module use /m31/modulefiles/static
module load miniconda
conda activate /projects/expmmllab/CoxeterEnv
```

Verify:

```bash
python --version                              # Python 3.11.x
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformer_lens; print(transformer_lens.__version__)"
```

Two rules that will save you time:

- **Never run on a login node.** Even short data-generation scripts and notebook work go on a
  compute node. Get there with `interactive -G 1` (see above); long training runs go through
  `sbatch` instead — see [`../RUNNING_ON_ANDROMEDA.md`](../RUNNING_ON_ANDROMEDA.md).
- **Hold at most one `interactive` allocation at a time.** Check for one you already have with
  `squeue -u $USER` before requesting another.

Batch jobs activate this environment themselves — every `transformer_job.sl` in the repo carries
the four lines above, so you do not need an interactive session to submit one.

### GPU requirement

CUDA 12.8 needs an **A100 or newer**. The V100s on the cluster are too old and `torch` will fail
at runtime rather than at import. The `--gres=gpu:a100:1` line in every job script is what pins
this; do not relax it to a bare `gpu:1`.

> If you are only *analyzing* a checkpoint rather than training, you can drop the `--gres`
> request entirely and run on CPU. A100 requests sometimes sit in `PD` for a while, and
> checkpoint analysis does not need a GPU.

---

## Option B — your own environment

```bash
conda create -n coxeter python=3.11
conda activate coxeter
pip install -r requirements.txt
```

`requirements.txt` pins exactly what `/projects/expmmllab/CoxeterEnv` has installed, so an
environment built this way will load checkpoints produced on the cluster. The pins matter more
than usual here: **`transformer_lens` pickles are version-sensitive**, and a checkpoint's `cfg`
object saved by one version may not unpickle under another. If you hit what looks like a corrupt
checkpoint, check the interpreter and `transformer_lens` versions before assuming the file is
damaged. The analysis notebooks carry a `sys.modules` shim for the one rename that is known to
occur, but it cannot cover every version skew.

`torch==2.7.1` in `requirements.txt` resolves to the CUDA 12.8 build on the cluster
(`2.7.1+cu128`). On a machine without a GPU, pip will install the CPU build instead and
everything except training will still work.

---

## What `Torch Setup.ipynb` is for

It is a **verification** notebook, not an installer. Run it inside an already-activated
environment to confirm the kernel is on a GPU node and the libraries import. Its `%pip install`
cells exist for building an environment from scratch and should be skipped once `CoxeterEnv` is
active — the header cell says as much.

Useful output to check:

- `!nvidia-smi` — must show a GPU. If it shows nothing, your Jupyter kernel is on a login node.
- `torch.cuda.is_available()` — must be `True`.
- `torch.cuda.get_device_name(0)` — must be an A100 or newer.

---

## Environment gotchas

- **Set `PYTHONIOENCODING=utf-8`** for any Python invocation that prints text derived from
  Windows-origin files. Otherwise you get a `cp1252` `UnicodeEncodeError` that has nothing to do
  with your code.
- **`mkdir logs` before `sbatch`.** SLURM will not create the output directory named in
  `--output`, and a job whose log directory is missing fails without writing anything
  anywhere — it looks like the job simply vanished.

---

## Files here

| File | Purpose |
|---|---|
| `README.md` | This file. |
| `requirements.txt` | Exact pins matching `/projects/expmmllab/CoxeterEnv`. Verified by `2026/check_repo.py`. |
| `Torch Setup.ipynb` | Verification notebook — GPU visible, libraries importable. |

The group-generation usage example that used to live here moved to
[`../2025/GroupGeneration/UsageExample.ipynb`](../2025/GroupGeneration/UsageExample.ipynb), next
to the `CoxeterArtinGroupGeneration.py` it imports.
