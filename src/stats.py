"""Statistics: paired bootstrap, ranking stability, Pareto frontiers.

This module is where the paper's claims are actually made, so the conventions are
worth stating once:

* Every reported difference gets a 95% CI. A difference whose CI straddles zero is
  reported as "inside noise" -- named explicitly, not quietly omitted. Naming null
  results is the single cheapest way to look competent.
* Comparisons pair on held-out EXAMPLES, not on runs. Two methods evaluated on the
  same 1,000 examples share all the example-difficulty variance, and pairing
  removes it.
* Seeds are a separate variance source and are aggregated separately. Pooling
  seeds and examples into one bootstrap conflates two distinct noise sources and
  produces intervals that are too narrow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

DEFAULT_BOOT = 10_000


@dataclass
class Comparison:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_paired: int
    significant: bool

    def describe(self, name_a: str = "A", name_b: str = "B", unit: str = "") -> str:
        verdict = "differ" if self.significant else "are INSIDE NOISE"
        return (
            f"{name_a} - {name_b} = {self.mean_diff:+.4f}{unit} "
            f"[95% CI {self.ci_low:+.4f}, {self.ci_high:+.4f}], "
            f"p={self.p_value:.4f}, n={self.n_paired} -> they {verdict}"
        )


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = DEFAULT_BOOT, seed: int = 0
) -> Comparison:
    """Bootstrap over paired per-example scores (e.g. held-out loss vectors).

    Resamples EXAMPLE INDICES, so each replicate keeps a and b aligned. The p-value
    is a two-sided percentile-based test: the fraction of replicates whose sign
    disagrees with the observed mean difference, doubled.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) < 2:
        raise ValueError("fewer than 2 paired observations survive")

    diff = a - b
    observed = float(diff.mean())
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)

    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided p: how often does the bootstrap cross zero, relative to observed sign?
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return Comparison(observed, float(lo), float(hi), float(min(p, 1.0)),
                      len(diff), significant=not (lo <= 0.0 <= hi))


def bootstrap_ci(x: np.ndarray, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Unpaired CI on a mean. Used for seed-level aggregates, where pairing is
    impossible because different seeds are genuinely different runs."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.RandomState(seed)
    boot = x[rng.randint(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def ranking_stability(rankings: dict[str, dict[str, float]], higher_is_better: bool = False) -> dict:
    """RQ1's central statistic.

    `rankings` maps a condition label (e.g. "lr=1e-4") to {method: score}. Returns
    pairwise Spearman and Kendall correlations between the method orderings induced
    by each condition.

    If selection-method rankings were a stable property of the methods, these
    correlations would sit near 1.0 regardless of learning rate. The proxy-fragility
    literature reports that they do not. H1 predicts partial collapse here, and this
    function is what tests it.
    """
    labels = sorted(rankings)
    methods = sorted(set.intersection(*(set(rankings[c]) for c in labels)))
    if len(methods) < 3:
        raise ValueError(
            f"Rank correlation over {len(methods)} methods is meaningless. "
            "Need at least 3 methods present in every condition."
        )

    sign = -1.0 if higher_is_better else 1.0  # rank 1 = best in both cases
    vectors = {c: np.array([sign * rankings[c][m] for m in methods]) for c in labels}

    pairwise = []
    for i, ci in enumerate(labels):
        for cj in labels[i + 1 :]:
            rho, p_rho = stats.spearmanr(vectors[ci], vectors[cj])
            tau, p_tau = stats.kendalltau(vectors[ci], vectors[cj])
            pairwise.append({
                "condition_a": ci, "condition_b": cj,
                "spearman": float(rho), "spearman_p": float(p_rho),
                "kendall": float(tau), "kendall_p": float(p_tau),
            })

    rhos = [p["spearman"] for p in pairwise]
    return {
        "methods": methods,
        "conditions": labels,
        "orderings": {c: [methods[i] for i in np.argsort(vectors[c])] for c in labels},
        "pairwise": pairwise,
        "mean_spearman": float(np.mean(rhos)),
        "min_spearman": float(np.min(rhos)),
        # The ICLR proxy-fragility paper treats Spearman > 0.95 as "stable".
        # Borrowing their threshold makes the two results directly comparable.
        "stable_at_0.95": bool(np.min(rhos) > 0.95),
    }


def top_k_overlap(sel_a: list[int], sel_b: list[int]) -> float:
    """Jaccard overlap between two selections. Answers a question rank correlation
    cannot: do two methods that score similarly actually pick the same examples?"""
    a, b = set(sel_a), set(sel_b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def pareto_frontier(cost: np.ndarray, quality: np.ndarray, lower_quality_is_better: bool = True):
    """Indices on the cost/quality Pareto frontier -- Figure 1 of the paper.

    A point is dominated if another point is no worse on cost AND strictly better
    on quality. RQ2's claim is that adding selection cost to the x-axis moves
    training-based methods off this frontier.
    """
    cost = np.asarray(cost, dtype=float)
    q = np.asarray(quality, dtype=float) * (1.0 if lower_quality_is_better else -1.0)
    order = np.argsort(cost, kind="stable")

    frontier, best = [], np.inf
    for i in order:
        if q[i] < best:
            frontier.append(int(i))
            best = q[i]
    return sorted(frontier)


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Multiplicity correction across the main grid.

    Six methods x two ratios is enough comparisons that an uncorrected 0.05
    threshold will manufacture a "significant" result by chance. Holm is uniformly
    more powerful than Bonferroni and makes no independence assumption, so there is
    no reason to use plain Bonferroni here.
    """
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(ordered)
    out, still_rejecting = {}, True
    for rank, (name, p) in enumerate(ordered):
        threshold = alpha / (n - rank)
        if p > threshold:
            still_rejecting = False
        out[name] = {"p": p, "threshold": threshold, "reject": still_rejecting and p <= threshold}
    return out
