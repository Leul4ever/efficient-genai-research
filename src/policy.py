"""Budget-aware selection: the paper's constructive contribution.

The gap this closes
-------------------
The data-selection literature asks "which selection method is best?". That
question is ill-posed. Methods carry different FIXED costs -- IFD needs two
forward passes over the pool, learning-percentage needs a whole proxy training
epoch, random needs nothing -- and a fixed cost has to be paid before a single
target-training step runs. Under a finite budget, spending on selection is
spending you cannot spend on training. So the honest question is:

    given a total compute budget B, which (method, ratio) pair should I pick?

That question has a different answer at different budgets, which is why "method
X is best" is never a complete claim. This module answers it.

The model
---------
Two halves, both fitted from the main grid and neither invented.

1. QUALITY. Standard data-scaling law, with one addition. A selection method is
   summarised by a single scalar: its **effective data multiplier** e_m. Method m
   selecting a fraction r behaves like random selection at fraction e_m * r.

       L(m, r) = L_inf + A * (e_m * r) ** (-alpha),      e_random := 1

   e_m = 2.0 means "this method's 5% is worth 10% of randomly chosen data". That
   is precisely the claim the efficiency literature makes implicitly, reduced to
   one interpretable, falsifiable number. Fixing e_random = 1 is what makes the
   parameterisation identifiable -- otherwise e and A trade off freely.

   The exponent alpha is SHARED across methods, not fitted per method. It is a
   property of the dataset and task, and per-method exponents are not identifiable
   from two ratios. The residuals report whether that assumption held.

2. COST. Linear in the data fraction, plus the method's fixed selection cost:

       C(m, r) = C_sel(m) + c_train * r

The rule
--------
Loss decreases monotonically in r, so under a budget the best ratio for a method
is simply the largest one it can still afford after paying its own selection cost:

    r*(m) = clip((B - C_sel(m)) / c_train, 0, 1)
    choose argmin_m  L(m, r*(m))

Closed-form, no search. The tension it captures is real: an expensive method must
have an e_m high enough to repay the training budget its selection step consumed.

Why this is testable rather than decorative
-------------------------------------------
Fit on Study 1 (Qwen2.5-0.5B) and PREDICT for Studies 3 and 4 (Qwen2.5-1.5B,
Llama-3.2-1B). Those studies were already planned, so validation costs zero extra
GPU-hours -- and it upgrades Study 4 from "exploratory" to "held-out test of the
proposed rule". `policy_regret` measures how much loss the rule gives up against
an oracle that knew the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

BASELINE_METHOD = "random"  # the method whose effective multiplier is pinned to 1


@dataclass
class ScalingFit:
    """Fitted quality model. `multipliers` is the headline result."""

    l_inf: float
    amplitude: float
    alpha: float
    multipliers: dict[str, float]
    rmse: float
    n_points: int
    residuals: dict[tuple[str, float], float] = field(default_factory=dict)

    def predict(self, method: str, ratio: float) -> float:
        if ratio <= 0:
            return float("inf")
        e = self.multipliers.get(method, 1.0)
        return self.l_inf + self.amplitude * (e * ratio) ** (-self.alpha)

    def describe(self) -> str:
        ranked = sorted(self.multipliers.items(), key=lambda kv: -kv[1])
        lines = [
            f"L(m, r) = {self.l_inf:.4f} + {self.amplitude:.4f} * (e_m * r)^(-{self.alpha:.4f})",
            f"fitted on {self.n_points} points, RMSE {self.rmse:.5f}",
            "effective data multipliers (e_m > 1 means the method beats random):",
        ]
        for name, e in ranked:
            lines.append(f"    {name:>22}  e = {e:5.2f}x")
        return "\n".join(lines)


def fit_scaling(points: list[tuple[str, float, float]],
                baseline: str = BASELINE_METHOD) -> ScalingFit:
    """Fit the quality model.

    points: (method, ratio, held_out_loss). Pass one entry per seed -- repeated
    (method, ratio) pairs are exactly what constrains the fit, so do NOT average
    seeds away first.
    """
    # At ratio 1.0 every method selects the same thing -- the entire pool -- so a
    # full-data point says nothing about that method's multiplier. Attributing it
    # to the method lets the r=1 anchor be absorbed into e_m, which silently
    # inflates whichever method happened to supply the ceiling run.
    points = [(baseline if r >= 1.0 else m, r, l) for m, r, l in points]

    methods = sorted({m for m, _, _ in points})
    if baseline not in methods:
        raise ValueError(
            f"baseline method {baseline!r} absent from the fit data. Without it the "
            "multipliers have no anchor and e_m is not identifiable."
        )
    free = [m for m in methods if m != baseline]

    # 3 global params + one multiplier per non-baseline method.
    n_params = 3 + len(free)
    if len(points) < n_params + 1:
        raise ValueError(
            f"{len(points)} points cannot constrain {n_params} parameters. "
            f"Need at least {n_params + 1}; run more of the main grid first."
        )

    distinct_ratios = {r for _, r, _ in points}
    if len(distinct_ratios) < 2:
        raise ValueError(
            f"all points sit at ratio(s) {sorted(distinct_ratios)}. The scaling "
            "exponent is unidentifiable from a single ratio -- the data carries no "
            "information about the curve's slope. Fit needs at least two ratios."
        )

    idx = {m: i for i, m in enumerate(free)}
    losses = np.array([l for _, _, l in points], dtype=float)
    ratios = np.array([r for _, r, _ in points], dtype=float)
    which = np.array([idx.get(m, -1) for m, _, _ in points])

    def unpack(theta):
        l_inf, log_a, log_alpha = theta[0], theta[1], theta[2]
        log_e = theta[3:]
        return l_inf, np.exp(log_a), np.exp(log_alpha), log_e

    def residual(theta):
        l_inf, a, alpha, log_e = unpack(theta)
        # Least-squares probes wild parameter values en route to the optimum, where
        # this power overflows. The trajectory recovers; the warnings would only
        # mask genuine numerical problems, so silence them locally rather than
        # globally.
        # exp() keeps A, alpha, and every multiplier strictly positive without
        # handing the optimiser hard bounds it can get stuck against.
        if len(log_e):
            # np.where evaluates BOTH branches, so the index must be valid even
            # where the mask discards it. Clamp first, mask second.
            safe = np.where(which >= 0, which, 0)
            e = np.where(which >= 0, np.exp(log_e[safe]), 1.0)
        else:
            e = np.ones_like(ratios)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred = l_inf + a * (e * ratios) ** (-alpha)
        return np.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6) - losses

    theta0 = np.concatenate([[losses.min() * 0.9, np.log(0.1), np.log(0.3)],
                             np.zeros(len(free))])
    sol = optimize.least_squares(residual, theta0, method="lm", max_nfev=20000)

    l_inf, a, alpha, log_e = unpack(sol.x)
    multipliers = {baseline: 1.0}
    multipliers.update({m: float(np.exp(log_e[i])) for m, i in idx.items()})

    res = residual(sol.x)
    per_point = {}
    for (m, r, _), e in zip(points, res):
        per_point.setdefault((m, r), []).append(float(e))

    return ScalingFit(
        l_inf=float(l_inf), amplitude=float(a), alpha=float(alpha),
        multipliers=multipliers,
        rmse=float(np.sqrt(np.mean(res ** 2))), n_points=len(points),
        residuals={k: float(np.mean(v)) for k, v in per_point.items()},
    )


@dataclass
class Choice:
    method: str
    ratio: float
    predicted_loss: float
    selection_flops: float
    training_flops: float

    @property
    def total_flops(self) -> float:
        return self.selection_flops + self.training_flops

    def __str__(self) -> str:
        return (f"{self.method}@{self.ratio:.3f} "
                f"(predicted loss {self.predicted_loss:.4f}, "
                f"{self.total_flops:.2e} FLOPs total)")


def best_affordable_ratio(budget: float, sel_cost: float, full_train_cost: float) -> float:
    """Largest ratio still affordable after the method's selection cost is paid.

    Loss is monotonically decreasing in ratio, so there is never a reason to buy
    less data than the budget allows. That monotonicity is what removes the search.
    """
    if budget <= sel_cost:
        return 0.0
    return float(np.clip((budget - sel_cost) / full_train_cost, 0.0, 1.0))


def choose(budget: float, fit: ScalingFit, sel_costs: dict[str, float],
           full_train_cost: float, methods: list[str] | None = None) -> Choice | None:
    """The rule. Returns the (method, ratio) to use, or None if nothing is affordable."""
    candidates = methods or sorted(sel_costs)
    best: Choice | None = None

    for m in candidates:
        if m not in sel_costs:
            raise KeyError(f"no measured selection cost for {m!r}")
        r = best_affordable_ratio(budget, sel_costs[m], full_train_cost)
        if r <= 0:
            continue  # budget does not even cover this method's selection step
        c = Choice(m, r, fit.predict(m, r), sel_costs[m], full_train_cost * r)
        if best is None or c.predicted_loss < best.predicted_loss:
            best = c
    return best


def budget_sweep(budgets: np.ndarray, fit: ScalingFit, sel_costs: dict[str, float],
                 full_train_cost: float) -> list[tuple[float, Choice | None]]:
    """The rule's answer across a range of budgets. This is the figure: the winning
    method CHANGES with budget, which is the whole point."""
    return [(float(b), choose(float(b), fit, sel_costs, full_train_cost)) for b in budgets]


def crossover_budget(method_a: str, method_b: str, fit: ScalingFit,
                     sel_costs: dict[str, float], full_train_cost: float,
                     lo: float | None = None, hi: float | None = None) -> float | None:
    """The budget at which method_b overtakes method_a.

    The single most quotable number this module produces: "IFD does not pay for
    itself below X FLOPs." Bisection on the predicted-loss difference; returns None
    if no crossing exists in the bracket, which is itself a finding (one method
    dominates at every budget worth considering).
    """
    lo = lo if lo is not None else max(sel_costs.values()) * 1.001
    hi = hi if hi is not None else max(sel_costs.values()) + full_train_cost

    def gap(b: float) -> float:
        ra = best_affordable_ratio(b, sel_costs[method_a], full_train_cost)
        rb = best_affordable_ratio(b, sel_costs[method_b], full_train_cost)
        la = fit.predict(method_a, ra) if ra > 0 else float("inf")
        lb = fit.predict(method_b, rb) if rb > 0 else float("inf")
        return la - lb  # > 0 once b is the better choice

    g_lo, g_hi = gap(lo), gap(hi)
    if not np.isfinite(g_lo) or not np.isfinite(g_hi) or g_lo * g_hi > 0:
        return None  # no sign change: no crossover in this range
    return float(optimize.brentq(gap, lo, hi, xtol=hi * 1e-9))


def policy_regret(fit: ScalingFit, sel_costs: dict[str, float], full_train_cost: float,
                  observed: dict[tuple[str, float], float], budget: float) -> dict:
    """Validate the rule on held-out targets (Studies 3 and 4).

    `observed` maps (method, ratio) -> measured loss on a target the fit never saw.
    Compares three policies at one budget:

      * ORACLE  -- the best affordable option, known only in hindsight;
      * OURS    -- what the rule picks from the fit;
      * STATIC  -- "always use the method that won the main grid", which is what
                   the literature's fixed-budget framing implies.

    Regret is measured in held-out loss. A rule that cannot beat STATIC is not
    worth proposing, and reporting that honestly is a valid outcome.
    """
    affordable = {
        (m, r): loss for (m, r), loss in observed.items()
        if sel_costs.get(m, 0.0) + full_train_cost * r <= budget
    }
    if not affordable:
        return {"budget": budget, "feasible": False}

    oracle_key = min(affordable, key=affordable.get)
    picked = choose(budget, fit, sel_costs, full_train_cost)

    def nearest_observed(method: str, ratio: float):
        same = [(m, r) for (m, r) in affordable if m == method]
        if not same:
            return None
        return min(same, key=lambda k: abs(k[1] - ratio))

    ours_key = nearest_observed(picked.method, picked.ratio) if picked else None
    static_method = min(fit.multipliers, key=lambda m: -fit.multipliers[m])
    static_key = nearest_observed(static_method, 1.0)

    def loss_of(key):
        return affordable[key] if key else float("inf")

    return {
        "budget": budget,
        "feasible": True,
        "oracle": {"choice": oracle_key, "loss": affordable[oracle_key]},
        "ours": {"choice": ours_key, "loss": loss_of(ours_key),
                 "regret": loss_of(ours_key) - affordable[oracle_key]},
        "static": {"choice": static_key, "loss": loss_of(static_key),
                   "regret": loss_of(static_key) - affordable[oracle_key]},
        "beats_static": loss_of(ours_key) <= loss_of(static_key),
    }
