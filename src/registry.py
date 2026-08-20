"""Append-only run log with content-addressed run IDs.

Kaggle sessions die at 12 hours and can be pre-empted sooner. The sweep driver
therefore has to be idempotent: re-running it must skip work already done rather
than duplicate it. A run's identity is the hash of the config that produced it, so
"have I already run this?" is answerable without trusting filenames or ordering.

Changing any field in the config -- learning rate, seed, selection artifact --
changes the run_id, which is what makes accidental result-mixing impossible.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from paths import RUNS

# Fields that define a run's identity. Anything not listed here (wall-clock,
# metrics, hostname) is an OUTPUT and must not affect the ID.
IDENTITY_FIELDS = (
    "study", "target_model", "selection_method", "ratio", "seed",
    "learning_rate", "lora_r", "lora_alpha", "epochs", "max_seq_len",
    "batch_size", "grad_accum", "template_hash",
)


def run_id(cfg: dict) -> str:
    missing = [f for f in IDENTITY_FIELDS if f not in cfg]
    if missing:
        raise KeyError(f"Config missing identity fields: {missing}")
    payload = json.dumps({f: cfg[f] for f in IDENTITY_FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    runs = []
    for line_no, line in enumerate(RUNS.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line is the expected symptom of a killed Kaggle
            # session. Warn and carry on rather than losing the whole log.
            print(f"WARN: results/runs.jsonl line {line_no} is malformed; skipping")
    return runs


def completed_ids() -> set[str]:
    return {r["run_id"] for r in load_runs() if r.get("status") == "ok"}


def append_run(record: dict) -> None:
    """Atomic-ish append. Write then flush+fsync so a session kill mid-write cannot
    leave a half-line that costs us an entire run's results."""
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    record = {**record, "env": {
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        **record.get("env", {}),
    }}
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(RUNS, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def to_dataframe():
    """Flatten the log for analysis. Metrics and cost are nested; flatten with prefixes."""
    import pandas as pd

    rows = []
    for r in load_runs():
        if r.get("status") != "ok":
            continue
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for key in ("metrics", "cost"):
            for k, v in (r.get(key) or {}).items():
                row[f"{key}.{k}"] = v
        for k, v in (r.get("config") or {}).items():
            row.setdefault(k, v)
        rows.append(row)
    return pd.DataFrame(rows)
