#!/usr/bin/env python3
"""
Repo consistency checks.

Run from anywhere:  python 2026/check_repo.py
Or one check:       python 2026/check_repo.py --configs

These exist because the failure modes they catch have all actually happened here: a
reorganization left documentation pointing at moved files, a saved HTTP error page got
committed with a .ipynb extension, SEQUENCE_LENGTH silently disagreed with the generator's
FIXED_LENGTH, and results tables drifted from the logs behind them.

Exit status is 0 only if every selected check passes.
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
Y2026 = REPO / "2026"
ENV_SITE_PACKAGES = Path("/projects/expmmllab/CoxeterEnv/lib/python3.11/site-packages")

# Files that are legitimately identical and should not be reported as duplicates.
# Keep this list short. Every run's config.py opens with a docstring naming that run,
# so configs sharing identical settings are still distinct files -- that is deliberate,
# and it is why this list holds only datasets.
DUPLICATE_ALLOWLIST = {
    # The 2025 dataset is present both in its own tree and in the replication folder.
    "2026/extensions/replication_2025/train.csv",
    "2026/extensions/replication_2025/test.csv",
    "2025/transformer/train.csv",
    "2025/transformer/test.csv",
    "2025/datasets/0 . A2_tilde . 'coxeter' . 6-22 . pad 22 . size 129,300 . split 40 60/train.csv",
    "2025/datasets/0 . A2_tilde . 'coxeter' . 6-22 . pad 22 . size 129,300 . split 40 60/test.csv",
}

RESULTS_MD = Y2026 / "RESULTS.md"
RUNS_DIR = Path.home() / "Runs"          # where the source logs live (not in the repo)


# --------------------------------------------------------------------------- helpers

class Result:
    def __init__(self):
        self.failures = []

    def fail(self, msg):
        self.failures.append(msg)

    def report(self, name):
        if self.failures:
            print(f"FAIL  {name}")
            for f in self.failures:
                print(f"        {f}")
        else:
            print(f"ok    {name}")
        return not self.failures


def tracked_files():
    """Every git-tracked file, as repo-relative Paths. Falls back to a walk."""
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
        return [Path(p) for p in out.split("\0") if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [p.relative_to(REPO) for p in REPO.rglob("*")
                if p.is_file() and ".git" not in p.parts]


# --------------------------------------------------------------------------- checks

def check_doc_paths():
    """Every backticked repo-relative path and every markdown link resolves."""
    r = Result()
    # A backticked token is treated as a path only if it looks like one: contains a
    # slash or a known extension, and no spaces-with-prose or shell metacharacters.
    exts = (".py", ".ipynb", ".md", ".csv", ".sl", ".cpp", ".txt", ".html", ".js", ".mjs", ".css")
    # Paths produced at runtime and gitignored, so they legitimately do not exist in a
    # clean checkout. Documenting them is the point; asserting they exist is not.
    runtime = ("workspace", "logs/", "_scratch", "data.csv", "train.csv", "test.csv",
               "payload.json", "curriculum_data/", "Runs/")

    for md in sorted(REPO.rglob("*.md")):
        if ".git" in md.parts:
            continue
        text = md.read_text(errors="replace")

        # Markdown links: [label](target)
        for target in re.findall(r"\]\(([^)#\s]+)\)", text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                r.fail(f"{md.relative_to(REPO)}: link -> {target}")

        # Backticked paths
        for tok in re.findall(r"`([^`\n]+)`", text):
            tok = tok.strip()
            if " " in tok and not tok.endswith(exts):
                continue
            if not (tok.endswith(exts) or "/" in tok):
                continue
            if tok.startswith(("http", "-", "$", "#", "%", "/")) or "*" in tok or "<" in tok:
                continue
            if any(c in tok for c in "|=(){}"):
                continue
            if any(part in tok for part in runtime):
                continue
            # Resolve against the doc's folder, the repo root, and the 2026 root --
            # prose routinely says `shared/descents.py` from inside an arm folder.
            if any((base / tok.lstrip("/")).exists() for base in (md.parent, REPO, Y2026)):
                continue
            # Bare filenames are often referred to generically ("config.py"); only
            # flag a token that names a directory component, which asserts a location.
            if "/" in tok:
                r.fail(f"{md.relative_to(REPO)}: `{tok}`")
    return r.report("doc paths resolve")


def check_notebooks():
    """Every .ipynb is valid JSON with a cells list and no duplicate cell ids."""
    r = Result()
    for nb_path in sorted(REPO.rglob("*.ipynb")):
        if ".git" in nb_path.parts or ".ipynb_checkpoints" in nb_path.parts:
            continue
        try:
            nb = json.loads(nb_path.read_text(errors="replace"))
        except json.JSONDecodeError as e:
            r.fail(f"{nb_path.relative_to(REPO)}: not valid JSON ({e})")
            continue
        if "cells" not in nb:
            r.fail(f"{nb_path.relative_to(REPO)}: no 'cells' key")
            continue
        ids = [c["id"] for c in nb["cells"] if "id" in c]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            r.fail(f"{nb_path.relative_to(REPO)}: duplicate cell ids {sorted(dupes)}")
    return r.report("notebooks parse")


def check_python_compiles():
    """Every .py parses. Catches truncation and stray non-Python content."""
    r = Result()
    for py in sorted(REPO.rglob("*.py")):
        if ".git" in py.parts or "__pycache__" in py.parts:
            continue
        try:
            ast.parse(py.read_text(errors="replace"), filename=str(py))
        except SyntaxError as e:
            r.fail(f"{py.relative_to(REPO)}:{e.lineno}: {e.msg}")
    return r.report("python files compile")


def check_duplicates():
    """No two tracked text files have identical content, outside the allowlist."""
    r = Result()
    by_hash = defaultdict(list)
    for rel in tracked_files():
        p = REPO / rel
        if not p.is_file() or p.suffix in {".png", ".jpg", ".pth"}:
            continue
        by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(str(rel))

    for group in by_hash.values():
        if len(group) < 2:
            continue
        if all(g in DUPLICATE_ALLOWLIST for g in group):
            continue
        r.fail("identical: " + ", ".join(sorted(group)))
    return r.report("no unexpected duplicate files")


def _const(path, name):
    """Read a module-level `name = <literal>` from a Python file without importing it."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except ValueError:
                        return None
    return None


def check_configs():
    """SEQUENCE_LENGTH agrees with the builder's FIXED_LENGTH; vocab sizes are consistent."""
    r = Result()
    for cfg in sorted(Y2026.rglob("config*.py")):
        rel = cfg.relative_to(REPO)
        seq = _const(cfg, "SEQUENCE_LENGTH")
        if seq is None:
            r.fail(f"{rel}: no SEQUENCE_LENGTH")
            continue

        # TOKEN_TYPES = #generators + 1 pad; DIM_OUTPUT = #generators, for the
        # multi-label descent task. The classification/sequence variants break this
        # by design, so only check when both are present and the task is multi-label.
        tok, dim = _const(cfg, "TOKEN_TYPES"), _const(cfg, "DIM_OUTPUT")
        multilabel = "descent" in str(rel).lower() or "reduced" in str(rel).lower()
        if multilabel and tok is not None and dim is not None and tok != dim + 1:
            r.fail(f"{rel}: TOKEN_TYPES ({tok}) != DIM_OUTPUT + 1 ({dim + 1})")

        # A builder in this folder or its parent should agree on the padded length.
        for folder in (cfg.parent, cfg.parent.parent):
            for builder in sorted(folder.glob("build*.py")):
                fixed = _const(builder, "FIXED_LENGTH")
                if fixed is not None and fixed != seq:
                    r.fail(f"{rel}: SEQUENCE_LENGTH={seq} but "
                           f"{builder.relative_to(REPO)} FIXED_LENGTH={fixed}")
            break   # only the immediate folder; parent builders are shared across runs
    return r.report("configs agree with builders")


def check_results():
    """Every metric in RESULTS.md matches the .out log that row cites."""
    r = Result()
    if not RESULTS_MD.exists():
        r.fail("2026/RESULTS.md missing")
        return r.report("RESULTS.md matches its logs")
    if not RUNS_DIR.is_dir():
        print("skip  RESULTS.md matches its logs (no ~/Runs/ on this machine)")
        return True

    logs = {p.name: p for p in RUNS_DIR.rglob("logs/*.out")}
    text = RESULTS_MD.read_text()
    cited = re.findall(r"`([A-Za-z0-9_]+_\d+\.out)`", text)
    if not cited:
        r.fail("no source logs cited")

    for line in text.splitlines():
        m = re.search(r"`([A-Za-z0-9_]+_\d+\.out)`", line)
        if not m:
            continue
        name = m.group(1)
        if name not in logs:
            r.fail(f"cited log not found under ~/Runs/: {name}")
            continue
        epochs = re.findall(r"^Epoch .*$", logs[name].read_text(errors="replace"), re.M)
        if not epochs:
            r.fail(f"{name}: no training output")
            continue
        final = dict(re.findall(r"(Train Loss|Test Loss|Train Acc|Test Acc|Train Bit Acc|"
                                r"Test Bit Acc|Train Seq Acc|Test Seq Acc): ([0-9.]+)",
                                epochs[-1]))
        # Every 4-decimal number in the row must appear among that log's final metrics.
        claimed = set(re.findall(r"\b(\d\.\d{4})\b", line))
        actual = set(final.values())
        for c in claimed - actual:
            r.fail(f"{name}: table claims {c}, not in final log line {sorted(actual)}")
    return r.report("RESULTS.md matches its logs")


def check_requirements():
    """Setup/requirements.txt pins match what the shared env actually has installed."""
    r = Result()
    req = REPO / "Setup/requirements.txt"
    if not req.exists():
        r.fail("Setup/requirements.txt missing")
        return r.report("requirements match the shared env")
    if not ENV_SITE_PACKAGES.is_dir():
        print("skip  requirements match the shared env (env not mounted here)")
        return True

    installed = {}
    for d in ENV_SITE_PACKAGES.glob("*.dist-info"):
        name, _, version = d.name[: -len(".dist-info")].rpartition("-")
        installed[name.lower().replace("_", "-")] = version

    for line in req.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "git+")):
            continue
        if "==" not in line:
            continue
        name, _, want = line.partition("==")
        have = installed.get(name.strip().lower().replace("_", "-"))
        if have is None:
            r.fail(f"{name}: pinned {want} but not installed in the env")
        elif not have.startswith(want):        # torch reports 2.7.1+cu128 for 2.7.1
            r.fail(f"{name}: pinned {want}, env has {have}")
    return r.report("requirements match the shared env")


CHECKS = {
    "docs": check_doc_paths,
    "notebooks": check_notebooks,
    "python": check_python_compiles,
    "duplicates": check_duplicates,
    "configs": check_configs,
    "results": check_results,
    "requirements": check_requirements,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in CHECKS:
        ap.add_argument(f"--{name}", action="store_true", help=f"run only the {name} check")
    args = ap.parse_args()

    selected = [n for n in CHECKS if getattr(args, n)] or list(CHECKS)
    # Run every selected check -- do not short-circuit on the first failure.
    ok = all([CHECKS[n]() for n in selected])
    print()
    print("All checks passed." if ok else "Some checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
