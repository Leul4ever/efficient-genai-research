"""Single source of truth for on-disk layout.

Every artifact the pipeline produces is addressed through here so that the
laptop (selection) and Kaggle (training) halves agree without configuration.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"

# Frozen dataset split. Committed to git — this file IS the experiment's ground truth.
SPLIT = RESULTS / "split.json"

# One file per (method, ratio, seed). Produced on CPU, consumed on GPU. Committed.
SELECTIONS = RESULTS / "selections"

# Cached per-example scores, keyed by scorer config rather than ratio/seed.
# Score-based selectors are deterministic, so this is computed once per method.
SCORES = RESULTS / "scores"

# Append-only run log. One JSON object per training run.
RUNS = RESULTS / "runs.jsonl"

# Per-run raw artifacts (per-example losses, generations). Not committed.
RUN_ARTIFACTS = RESULTS / "runs"


def selection_path(method: str, ratio: float, seed: int) -> Path:
    return SELECTIONS / f"{method}__r{ratio:g}__s{seed}.json"
