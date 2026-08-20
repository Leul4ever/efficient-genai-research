"""Frozen dataset split and prompt formatting.

The split is created ONCE by scripts/make_split.py and committed. Nothing in the
pipeline is ever allowed to re-shuffle it. Every number in the paper is traceable
to results/split.json.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from datasets import load_dataset

from paths import SPLIT

POOL_NAME = "databricks/databricks-dolly-15k"
N_TRAIN = 14_000
N_HELD_OUT = 1_000
SPLIT_SEED = 0

# Alpaca-style template. Frozen in Week 1; changing it invalidates every prior run,
# so the template string is hashed into the run record.
_WITH_CONTEXT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
)
_NO_CONTEXT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def template_hash() -> str:
    return hashlib.sha256((_WITH_CONTEXT + _NO_CONTEXT).encode()).hexdigest()[:12]


def build_prompt(ex: dict) -> str:
    ctx = (ex.get("context") or "").strip()
    if ctx:
        return _WITH_CONTEXT.format(instruction=ex["instruction"].strip(), context=ctx)
    return _NO_CONTEXT.format(instruction=ex["instruction"].strip())


def build_full(ex: dict, eos: str = "") -> str:
    """Prompt + response. This is what the model is trained on."""
    return build_prompt(ex) + ex["response"].strip() + eos


def make_split() -> dict:
    """Deterministic split. Called once; result is committed."""
    ds = load_dataset(POOL_NAME, split="train")
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(ds))
    held_out = sorted(int(i) for i in perm[:N_HELD_OUT])
    train = sorted(int(i) for i in perm[N_HELD_OUT : N_HELD_OUT + N_TRAIN])
    return {
        "pool": POOL_NAME,
        "pool_size": len(ds),
        "split_seed": SPLIT_SEED,
        "template_hash": template_hash(),
        "held_out_idx": held_out,
        "train_idx": train,
    }


def load_split() -> dict:
    if not SPLIT.exists():
        raise FileNotFoundError(
            f"{SPLIT} missing. Run `python scripts/make_split.py` once, then commit it."
        )
    split = json.loads(SPLIT.read_text())
    if split["template_hash"] != template_hash():
        raise RuntimeError(
            "Prompt template changed after the split was frozen. Either revert the "
            "template or re-run every experiment. Do not silently proceed."
        )
    return split


def load_pool():
    """The raw Dolly rows, unsplit. Index into these with split['train_idx'] etc."""
    return load_dataset(POOL_NAME, split="train")


def load_train_examples() -> list[dict]:
    split, pool = load_split(), load_pool()
    return [dict(pool[i]) for i in split["train_idx"]]


def load_held_out_examples() -> list[dict]:
    split, pool = load_split(), load_pool()
    return [dict(pool[i]) for i in split["held_out_idx"]]
