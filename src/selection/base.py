"""Selector interface and registry.

A selector maps (examples, k, seed) -> the indices of k examples, plus a CostRecord.
Subset-based methods (facility location) need the general form; score-based methods
get ScoreSelector, which handles top-k and direction.

`direction` is deliberately explicit on every score-based method. "Select the hard
examples" and "select the easy examples" are different hypotheses and the literature
disagrees on which wins; the flag makes the choice visible in the run record instead
of buried in a comparison operator.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from cost import CostRecord
from paths import SCORES

REGISTRY: dict[str, Callable[..., "Selector"]] = {}


def register(name: str):
    def deco(cls):
        if name in REGISTRY:
            raise KeyError(f"Selector {name!r} registered twice")
        REGISTRY[name] = cls
        cls.name = name
        return cls

    return deco


def build(name: str, **kwargs) -> "Selector":
    if name not in REGISTRY:
        raise KeyError(f"Unknown selector {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


class Selector:
    name: str = "unset"
    cost_class: str = "unset"  # "free" | "training-free" | "training-based"

    def select(self, examples: list[dict], k: int, seed: int) -> tuple[np.ndarray, CostRecord]:
        raise NotImplementedError

    def config(self) -> dict:
        """Serialised into the selection artifact so runs are reproducible."""
        return {"name": self.name, "cost_class": self.cost_class}


class ScoreSelector(Selector):
    """Selector defined by a per-example scalar, taken top-k or bottom-k."""

    direction: str = "high"  # "high" -> keep largest scores, "low" -> smallest

    def score(self, examples: list[dict]) -> tuple[np.ndarray, CostRecord]:
        raise NotImplementedError

    def select(self, examples: list[dict], k: int, seed: int) -> tuple[np.ndarray, CostRecord]:
        scores, cost = scores_with_cache(self, examples)
        if len(scores) != len(examples):
            raise ValueError(f"{self.name}: got {len(scores)} scores for {len(examples)} examples")
        finite = np.isfinite(scores)
        if finite.sum() < k:
            raise ValueError(
                f"{self.name}: only {finite.sum()} finite scores available for budget k={k}. "
                f"Refusing to silent-fallback to index ordering."
            )
        if not finite.all():
            # Non-finite scores are a real failure mode (empty responses, overflow).
            # Push them to the losing end rather than letting argsort place them arbitrarily.
            worst = -np.inf if self.direction == "high" else np.inf
            scores = np.where(finite, scores, worst)
            cost.notes["non_finite_scores"] = int((~finite).sum())
        order = np.argsort(-scores if self.direction == "high" else scores, kind="stable")
        idx = np.sort(order[:k])
        cost.notes["score_mean"] = float(np.mean(scores[finite])) if finite.any() else None
        return idx, cost

    def config(self) -> dict:
        return {**super().config(), "direction": self.direction}


def scores_with_cache(selector, examples: list[dict]) -> tuple[np.ndarray, CostRecord]:
    """Compute a selector's scores, or reuse them if an identical scorer config has
    already produced them. Used by ScoreSelector.select and by any composite
    selector that embeds a scorer -- HybridSelector wraps a perplexity scorer, and
    without going through here it would recompute a 70-minute CPU pass that is
    already sitting on disk."""
    cached = load_cached_scores(selector, len(examples))
    if cached is not None:
        return cached
    scores, cost = selector.score(examples)
    save_cached_scores(selector, scores, cost, len(examples))
    return scores, cost


def _cache_key(selector) -> str:
    """Identity of a SCORE, which depends on the scorer and its settings but NOT on
    ratio or seed. Score-based selectors are deterministic: the same scorer over the
    same pool yields the same numbers every time. Recomputing per (ratio, seed) --
    six times for the main grid -- is pure waste, and at measured CPU rates it is
    the difference between 8.5 hours and 51 hours for IFD.
    """
    import hashlib
    import json

    payload = json.dumps(selector.config(), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cached_scores(selector, n_examples: int):
    """Return (scores, cost) if a matching score file exists, else None."""
    import json

    path = SCORES / f"{selector.name}__{_cache_key(selector)}.npz"
    if not path.exists():
        return None
    blob = np.load(path, allow_pickle=False)
    if int(blob["n_examples"]) != n_examples:
        return None
    meta = json.loads(path.with_suffix(".json").read_text())
    cost = CostRecord(**meta["cost"])
    # Flag the reuse, but keep the ORIGINAL measured cost. Reusing a score across
    # ratios and seeds is what a practitioner would really do; pretending the
    # second use was free would understate the method's true price, which is
    # precisely the accounting error this project exists to criticise.
    cost.notes = {**cost.notes, "reused_from_cache": True}
    return blob["scores"], cost


def save_cached_scores(selector, scores, cost, n_examples: int) -> None:
    import json

    SCORES.mkdir(parents=True, exist_ok=True)
    stem = SCORES / f"{selector.name}__{_cache_key(selector)}"
    np.savez_compressed(stem.with_suffix(".npz"), scores=scores, n_examples=n_examples)
    stem.with_suffix(".json").write_text(json.dumps({
        "method": selector.name,
        "selector_config": selector.config(),
        "n_examples": n_examples,
        "cost": cost.as_dict(),
    }, indent=2))

