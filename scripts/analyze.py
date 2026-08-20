"""Stage 3: turn results/runs.jsonl into the paper's three results sections.

Runs on the laptop, needs no GPU. Every claim it prints carries a CI, and
differences whose CI straddles zero are printed as INSIDE NOISE rather than being
quietly dropped -- that phrasing should survive into the paper verbatim.

    python scripts/analyze.py --study study1_main_grid
    python scripts/analyze.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paths import RUN_ARTIFACTS  # noqa: E402
from registry import load_runs  # noqa: E402
from policy import choose, crossover_budget, fit_scaling  # noqa: E402
from stats import (  # noqa: E402
    bootstrap_ci, holm_bonferroni, paired_bootstrap, pareto_frontier, ranking_stability,
)

BASELINE = "random"


def ok_runs(study: str | None = None) -> list[dict]:
    runs = [r for r in load_runs() if r.get("status") == "ok"]
    return [r for r in runs if study is None or r.get("study") == study]


def per_example_losses(run: dict) -> np.ndarray | None:
    path = RUN_ARTIFACTS / run["run_id"] / "held_out_per_example_loss.npy"
    return np.load(path) if path.exists() else None


def mean_over_seeds(runs: list[dict], key: str = "metrics.held_out_loss") -> dict:
    """Group by (method, ratio, lr, model) and average across seeds.

    Seeds are a separate variance source from examples, so they are aggregated
    here and NOT pooled into the per-example bootstrap.
    """
    buckets: dict[tuple, list[float]] = {}
    metric = key.split(".", 1)[1]
    for r in runs:
        c = r["config"]
        k = (c["selection_method"], c["ratio"], c["learning_rate"], c["target_model"])
        v = r["metrics"].get(metric)
        if v is not None and np.isfinite(v):
            buckets.setdefault(k, []).append(float(v))
    return {k: bootstrap_ci(np.array(v)) for k, v in buckets.items() if v}


def rq1_ranking_stability(runs: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("RQ1 -- do selection-method rankings survive re-tuning the learning rate?")
    print("=" * 78)

    agg = mean_over_seeds(runs)
    by_lr: dict[str, dict[str, float]] = {}
    for (method, ratio, lr, model), (mean, _, _) in agg.items():
        by_lr.setdefault(f"lr={lr:g}", {})[method] = mean

    common = set.intersection(*(set(v) for v in by_lr.values())) if by_lr else set()
    by_lr = {k: {m: v[m] for m in common} for k, v in by_lr.items()}
    if len(common) < 3 or len(by_lr) < 2:
        print(f"  not enough data yet: {len(by_lr)} LRs x {len(common)} shared methods")
        return

    # Numeric, not alphabetical: "lr=2e-04" sorts before "lr=2e-05" as a string.
    order = sorted(by_lr, key=lambda s: float(s.split("=", 1)[1]))
    for label in order:
        ordering = sorted(by_lr[label], key=by_lr[label].get)
        print(f"  {label:>12}: " + " > ".join(ordering))

    r = ranking_stability(by_lr, higher_is_better=False, order=order)
    print(f"\n  mean Spearman across LR pairs : {r['mean_spearman']:+.3f}")
    print(f"  min  Spearman across LR pairs : {r['min_spearman']:+.3f}")
    print(f"  stable at the 0.95 threshold  : {r['stable_at_0.95']}")
    print("\n  " + ("H1 NOT supported: the ranking held. Report that plainly -- it "
                    "contradicts\n  the proxy-fragility result and is a finding."
                    if r["stable_at_0.95"] else
                    "H1 supported: the ranking moved with the learning rate. Part of the\n"
                    "  reported advantage of selection methods is a fixed-hyperparameter artifact."))


def rq2_net_cost(runs: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("RQ2 -- which methods win once SELECTION cost is added to training cost?")
    print("=" * 78)

    rows = []
    for r in runs:
        c, cost = r["config"], r["cost"]
        loss = r["metrics"].get("held_out_loss")
        if loss is None or not np.isfinite(loss):
            continue
        rows.append({
            "method": c["selection_method"], "ratio": c["ratio"], "loss": float(loss),
            "sel": cost["selection_flops"], "train": cost["training_flops"],
            "total": cost["total_flops"], "share": cost["selection_share_of_total_flops"],
            "class": cost.get("selection_cost_class", "?"),
        })
    if not rows:
        print("  no runs yet")
        return

    print(f"\n  {'method':>20} {'ratio':>6} {'class':>15} {'sel%':>7} "
          f"{'train FLOPs':>12} {'total FLOPs':>12} {'loss':>7}")
    for row in sorted(rows, key=lambda x: x["total"]):
        print(f"  {row['method']:>20} {row['ratio']:>6.2f} {row['class']:>15} "
              f"{row['share'] * 100:>6.1f}% {row['train']:>12.3e} "
              f"{row['total']:>12.3e} {row['loss']:>7.4f}")

    train_only = pareto_frontier(np.array([r["train"] for r in rows]),
                                 np.array([r["loss"] for r in rows]))
    with_sel = pareto_frontier(np.array([r["total"] for r in rows]),
                               np.array([r["loss"] for r in rows]))

    def label(i):
        return f"{rows[i]['method']}@{rows[i]['ratio']:g}"

    print(f"\n  Pareto frontier, TRAINING cost only : {[label(i) for i in train_only]}")
    print(f"  Pareto frontier, TOTAL cost         : {[label(i) for i in with_sel]}")
    dropped = set(train_only) - set(with_sel)
    if dropped:
        print(f"\n  H2 SUPPORTED for: {[label(i) for i in dropped]}")
        print("  These sit on the frontier only because prior work does not price selection.")
    else:
        print("\n  H2 not supported at this pool size: no method leaves the frontier "
              "once\n  selection is priced in. Worth reporting -- it bounds when the "
              "critique bites.")


def rq3_transfer(runs: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("RQ3 -- does the selection signal transfer across scale and model family?")
    print("=" * 78)

    agg = mean_over_seeds(runs)
    by_model: dict[str, dict[str, float]] = {}
    for (method, ratio, lr, model), (mean, _, _) in agg.items():
        by_model.setdefault(model.split("/")[-1], {})[method] = mean

    common = set.intersection(*(set(v) for v in by_model.values())) if by_model else set()
    if len(common) < 3 or len(by_model) < 2:
        print(f"  not enough data yet: {len(by_model)} models x {len(common)} shared methods")
        return

    by_model = {k: {m: v[m] for m in common} for k, v in by_model.items()}
    for model in sorted(by_model):
        print(f"  {model:>20}: " + " > ".join(sorted(by_model[model], key=by_model[model].get)))

    r = ranking_stability(by_model, higher_is_better=False)
    print(f"\n  mean Spearman across targets: {r['mean_spearman']:+.3f}")
    for p in r["pairwise"]:
        print(f"    {p['condition_a']:>18} vs {p['condition_b']:<18} "
              f"rho={p['spearman']:+.3f}  tau={p['kendall']:+.3f}")
    print("\n  Study 4 (cross-family) is underpowered by design -- 3 methods x 2 seeds.")
    print("  Report effect sizes with CIs and say so; do not make a significance claim.")


def paired_vs_baseline(runs: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"Paired bootstrap vs the '{BASELINE}' baseline (per held-out example)")
    print("=" * 78)

    base = {(r["config"]["ratio"], r["config"]["seed"]): r
            for r in runs if r["config"]["selection_method"] == BASELINE}
    if not base:
        print(f"  no '{BASELINE}' runs to compare against")
        return

    p_values, lines = {}, []
    for r in runs:
        c = r["config"]
        if c["selection_method"] == BASELINE:
            continue
        b = base.get((c["ratio"], c["seed"]))
        if b is None:
            continue
        a_loss, b_loss = per_example_losses(r), per_example_losses(b)
        if a_loss is None or b_loss is None:
            continue
        cmp = paired_bootstrap(a_loss, b_loss)
        name = f"{c['selection_method']}@{c['ratio']:g}/s{c['seed']}"
        p_values[name] = cmp.p_value
        lines.append(f"  {name:>32}  " + cmp.describe(c["selection_method"], BASELINE))

    if not lines:
        print("  no per-example loss artifacts found (results/runs/<run_id>/*.npy)")
        return
    for line in lines:
        print(line)

    print("\n  Holm-Bonferroni across the grid (six methods x two ratios manufactures")
    print("  false positives at an uncorrected 0.05):")
    for name, h in holm_bonferroni(p_values).items():
        if h["reject"]:
            print(f"    SURVIVES correction: {name}  (p={h['p']:.4f} <= {h['threshold']:.4f})")
    if not any(h["reject"] for h in holm_bonferroni(p_values).values()):
        print("    NOTHING survives correction. Report this as the headline result.")


def rq4_cost_aware_policy(runs: list[dict]) -> None:
    """The paper's constructive contribution.

    Fits the budget-aware selection rule on the main grid and reports the two
    numbers it exists to produce: each method's effective data multiplier, and the
    budget below which that method does not pay for itself.
    """
    print()
    print("=" * 78)
    print("RQ4 -- the cost-aware selection rule: what should you pick at budget B?")
    print("=" * 78)

    points, sel_costs, full_train = [], {}, None
    for r in runs:
        c, cost = r["config"], r["cost"]
        loss = r["metrics"].get("held_out_loss")
        if loss is None or not np.isfinite(loss):
            continue
        points.append((c["selection_method"], float(c["ratio"]), float(loss)))
        sel_costs.setdefault(c["selection_method"], cost["selection_flops"])
        if c["ratio"] >= 1.0:
            full_train = cost["training_flops"]

    if not points:
        print("  no runs yet")
        return

    if full_train is None:
        # No full-data anchor logged yet. The cost model is linear in ratio, so a
        # single run extrapolates it -- but say so, because the extrapolation is an
        # assumption and not a measurement.
        anchor = next(r for r in runs if r["metrics"].get("held_out_loss") is not None)
        full_train = anchor["cost"]["training_flops"] / max(anchor["config"]["ratio"], 1e-9)
        print("  note: no ratio=1.0 run logged; full-data cost extrapolated linearly")

    try:
        fit = fit_scaling(points)
    except ValueError as exc:
        print(f"  cannot fit yet: {exc}")
        return

    print()
    for line in fit.describe().splitlines():
        print("  " + line)

    worst = max((abs(v) for v in fit.residuals.values()), default=0.0)
    print()
    print(f"  largest per-condition residual: {worst:.5f}")
    print("  Compare that against the spread of the multipliers. If it is the same")
    print("  size, the shared-exponent assumption failed and the multipliers are not")
    print("  trustworthy -- report that rather than the multipliers.")

    if fit.degenerate:
        print()
        print("  The budget table and crossovers below are therefore NOT reportable:")
        print("  they are derived from a fit that does not identify its parameters.")
        print("  Report the measured selection-cost shares instead -- those are")
        print("  direct measurements and stand on their own.")
        return

    print()
    print("  What the rule picks, by budget:")
    print(f"  {'budget (FLOPs)':>16} {'method':>22} {'ratio':>7} {'pred. loss':>11}")
    # Start from the CHEAPEST method's reach, not the most expensive one. Starting
    # high hides the low-budget regime, which is exactly where the rule disagrees
    # with "just use the best method".
    floor = min(sel_costs.values()) + full_train * 0.01
    ceiling = max(sel_costs.values()) + full_train
    for b in np.logspace(np.log10(floor), np.log10(ceiling), 8):
        pick = choose(float(b), fit, sel_costs, full_train)
        if pick:
            print(f"  {b:>16.3e} {pick.method:>22} {pick.ratio:>7.3f} "
                  f"{pick.predicted_loss:>11.4f}")

    print()
    print("  Crossover budgets vs the random baseline")
    print("  (below this figure, the method does NOT pay for itself):")
    for m in sorted(sel_costs):
        if m in ("random", "full"):
            continue
        x = crossover_budget("random", m, fit, sel_costs, full_train)
        if x is None:
            print(f"    {m:>22}: no crossover -- one option dominates at every budget")
        else:
            print(f"    {m:>22}: {x:.3e} FLOPs")

    print()
    print("  These crossover figures are the quotable result: a selection method is")
    print("  not 'better' or 'worse' outright, it is better above a stated budget.")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        runs = ok_runs()
        # Select by what the runs CONTAIN, not by hardcoded study names. The fast
        # track renamed the studies (fast_grid, fast_lr_sweep) and the hardcoded
        # filters silently matched nothing, reporting "no runs yet" on a full grid.
        lr_runs = [r for r in runs
                   if len({x["config"]["learning_rate"] for x in runs
                           if x["study"] == r["study"]}) > 1]
        grid_runs = [r for r in runs if r not in lr_runs]
        if not grid_runs:
            grid_runs = runs
        print(f"{len(grid_runs)} grid runs, {len(lr_runs)} LR-sweep runs")
        rq1_ranking_stability(lr_runs or runs)
        rq2_net_cost(grid_runs)
        rq3_transfer(runs)
        paired_vs_baseline(grid_runs)
        rq4_cost_aware_policy(grid_runs)
        return

    if False:
        runs = ok_runs()
        print(f"{len(runs)} completed runs across "
              f"{len({r['study'] for r in runs})} studies")
        rq1_ranking_stability([r for r in runs if r["study"] == "study2_lr_sweep"])
        rq2_net_cost([r for r in runs if r["study"] == "study1_main_grid"])
        rq3_transfer([r for r in runs if r["study"] in
                      ("study1_main_grid", "study3_scale_transfer", "study4_cross_family")])
        paired_vs_baseline([r for r in runs if r["study"] == "study1_main_grid"])
        rq4_cost_aware_policy([r for r in runs if r["study"] == "study1_main_grid"])
        return

    runs = ok_runs(args.study)
    print(f"{len(runs)} completed runs" + (f" in {args.study}" if args.study else ""))
    rq1_ranking_stability(runs)
    rq2_net_cost(runs)
    rq3_transfer(runs)
    paired_vs_baseline(runs)
    rq4_cost_aware_policy(runs)


if __name__ == "__main__":
    main()
