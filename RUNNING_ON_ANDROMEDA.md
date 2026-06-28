# Running Jobs on Andromeda

Instructions for training the Categorical Transformer on Boston College's
Andromeda cluster (`andromeda.bc.edu`).

> **Job time estimate:** roughly
> `words (in thousands) × epochs (in thousands) × 7.2` seconds.
>
> - **Under ~1 hour** → run interactively in VS Code (faster turnaround).
> - **Longer jobs** → submit a remote SLURM batch job with `sbatch`.

---

## Interactive Run (VS Code — jobs under ~1 hour)

SSH into your `andromeda.bc.edu` account, then in a **bash** terminal:

```bash
interactive -G 1
module load miniconda
conda activate /projects/expmmllab/CoxeterEnv
```
Then cd into the working directory.

### 1. (Optional) Generate `data.csv`

Skip this if `data.csv` already exists.

- Set `INSTANCES` (number of words) in `Set_Generation(Right_Descent).py`.
- Set `SEQUENCE_LENGTH` (word length) in `config.py`.

Then run:

```bash
python "Set_Generation(Right_Descent).py"
```

### 2. Train a Model

- Make any edits to `config.py` and `Transformer.py`.

Then run:

```bash
python Transformer.py
```

The trained model is saved to a folder under `workspace/_scratch/`.

### 3. End the Session

Close the terminal (trash-can icon).

---

## Remote Batch Run (SLURM — longer jobs)

SSH into your `andromeda.bc.edu` account, then in a **bash** terminal:

```bash
cd transformer/2026/Categorical_Transformer/
sbatch transformer_job.sl
```

This submits the job to the cluster and saves the trained model to a folder
under `workspace/_scratch/`.

---

## Analysis

After training (either method):

1. Run the entire **`Torch Setup.ipynb`** notebook (at the repository root) to
   install/load all libraries.
2. Use **`Analysis.ipynb`** (in `transformer/2026/Categorical_Transformer/`)
   for analysis.
