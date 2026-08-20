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
        scores, cost = self.score(examples)
        if len(scores) != len(examples):
            raise ValueError(f"{self.name}: got {len(scores)} scores for {len(examples)} examples")
        finite = np.isfinite(scores)
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
