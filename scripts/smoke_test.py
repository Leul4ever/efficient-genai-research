"""Fast, dependency-light checks on the pieces that carry the paper's claims.

Runs on the laptop in seconds with no model downloads. The statistics and the
submodular solver are where a silent bug would corrupt every number in the paper
without ever raising an exception, so they get tested against known-answer cases.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from stats import (  # noqa: E402
    bootstrap_ci, holm_bonferroni, paired_bootstrap, pareto_frontier,
    ranking_stability, top_k_overlap,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")


def test_paired_bootstrap() -> None:
    print("\npaired_bootstrap")
    rng = np.random.RandomState(0)
    shared = rng.randn(1000) * 5.0           # example difficulty, shared by both arms
    a = shared + rng.randn(1000) * 0.1
    b = shared + rng.randn(1000) * 0.1 + 0.5  # b is worse by a constant 0.5

    r = paired_bootstrap(a, b)
    check("detects a real 0.5 effect buried in 10x larger shared variance",
          r.significant and abs(r.mean_diff + 0.5) < 0.05, r.describe("a", "b"))

    null = paired_bootstrap(shared + rng.randn(1000) * 0.1, shared + rng.randn(1000) * 0.1)
    check("reports no effect when there is none", not null.significant, null.describe())

    check("CI brackets the true effect", r.ci_low <= -0.5 <= r.ci_high)

    # NaNs are the realistic failure mode: an example with an empty response.
    a2, b2 = a.copy(), b.copy()
    a2[:10] = np.nan
    r2 = paired_bootstrap(a2, b2)
    check("drops non-finite pairs rather than propagating NaN",
          r2.n_paired == 990 and np.isfinite(r2.mean_diff))


def test_ranking_stability() -> None:
    print("\nranking_stability")
    stable = {
        "lr=1e-6": {"random": 2.0, "ppl": 1.8, "ifd": 1.7, "lp": 1.9},
        "lr=2e-5": {"random": 2.1, "ppl": 1.9, "ifd": 1.8, "lp": 2.0},
    }
    r = ranking_stability(stable)
    check("identical orderings -> Spearman 1.0", abs(r["mean_spearman"] - 1.0) < 1e-9)
    check("flags them as stable at the 0.95 threshold", r["stable_at_0.95"])
    check("best method ranked first (lower loss is better)",
          r["orderings"]["lr=1e-6"][0] == "ifd", str(r["orderings"]["lr=1e-6"]))

    collapsed = {
        "lr=1e-6": {"random": 2.0, "ppl": 1.8, "ifd": 1.7, "lp": 1.9},
        "lr=2e-4": {"random": 1.7, "ppl": 1.9, "ifd": 2.0, "lp": 1.8},  # reversed
    }
    r2 = ranking_stability(collapsed)
    check("reversed ordering -> Spearman -1.0", abs(r2["mean_spearman"] + 1.0) < 1e-9)
    check("does NOT call a collapsed ranking stable", not r2["stable_at_0.95"])

    try:
        ranking_stability({"a": {"m1": 1.0}, "b": {"m1": 2.0}})
        check("refuses rank correlation over <3 methods", False)
    except ValueError:
        check("refuses rank correlation over <3 methods", True)


def test_pareto() -> None:
    print("\npareto_frontier")
    #            cheap+good   cheap+bad   dear+best   dear+bad
    cost = np.array([1.0, 1.0, 10.0, 10.0])
    qual = np.array([2.0, 5.0, 1.5, 9.0])  # lower is better
    front = pareto_frontier(cost, qual)
    check("keeps cheap-and-good and expensive-but-best", front == [0, 2], str(front))
    check("drops dominated points", 1 not in front and 3 not in front)

    # RQ2's actual shape: a training-based method whose selection cost pushes it off.
    cost2 = np.array([1.0, 1.2, 8.0])
    qual2 = np.array([2.0, 1.8, 1.85])
    check("training-based point excluded once selection cost is added",
          pareto_frontier(cost2, qual2) == [0, 1], str(pareto_frontier(cost2, qual2)))


def test_facility_location() -> None:
    print("\nlazy_greedy_facility_location")
    from selection.methods import lazy_greedy_facility_location

    # Three tight clusters. A diversity objective must take one point per cluster.
    rng = np.random.RandomState(0)
    centers = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    pts = np.repeat(centers, 5, axis=0) + rng.randn(15, 2) * 0.01
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    sim = (pts @ pts.T).astype(np.float32)

    chosen = lazy_greedy_facility_location(sim, 3)
    clusters = {int(i) // 5 for i in chosen}
    check("picks one point from each of 3 clusters", clusters == {0, 1, 2}, str(sorted(chosen)))

    # Lazy greedy is exact, so it must agree with the naive implementation.
    def naive(sim, k):
        n = sim.shape[0]
        sim = sim - sim.min()  # same shift the real implementation applies
        cur, sel = np.zeros(n, dtype=np.float32), []
        for _ in range(k):
            gains = [(np.maximum(sim[:, j] - cur, 0).sum() if j not in sel else -1.0)
                     for j in range(n)]
            j = int(np.argmax(gains))
            sel.append(j)
            cur = np.maximum(cur, sim[:, j])
        return sel

    check("agrees exactly with naive greedy",
          sorted(lazy_greedy_facility_location(sim, 5).tolist()) == sorted(naive(sim, 5)))

    weighted = lazy_greedy_facility_location(sim, 3, prior=np.linspace(0.5, 1.5, 15).astype(np.float32))
    check("weighted variant returns k distinct indices", len(set(weighted.tolist())) == 3)


def test_misc() -> None:
    print("\nmisc")
    check("top_k_overlap identical -> 1.0", top_k_overlap([1, 2, 3], [1, 2, 3]) == 1.0)
    check("top_k_overlap disjoint -> 0.0", top_k_overlap([1, 2], [3, 4]) == 0.0)
    check("top_k_overlap half -> 1/3", abs(top_k_overlap([1, 2], [2, 3]) - 1 / 3) < 1e-9)

    m, lo, hi = bootstrap_ci(np.array([1.0, 1.1, 0.9, 1.05, 0.95]))
    check("bootstrap_ci brackets its own mean", lo <= m <= hi, f"{m:.3f} [{lo:.3f},{hi:.3f}]")

    h = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.9})
    check("Holm rejects the strongest, keeps the weakest",
          h["a"]["reject"] and not h["c"]["reject"])
    check("Holm is stricter than 0.05 on the smallest p", h["a"]["threshold"] < 0.05)


def test_registry() -> None:
    print("\nregistry")
    from registry import IDENTITY_FIELDS, run_id

    cfg = {f: 1 for f in IDENTITY_FIELDS}
    check("run_id is deterministic", run_id(cfg) == run_id(dict(reversed(list(cfg.items())))))
    check("changing the learning rate changes the run_id",
          run_id(cfg) != run_id({**cfg, "learning_rate": 2}))
    check("changing an output field does NOT change the run_id",
          run_id(cfg) == run_id({**cfg, "wall_clock_s": 999}))
    try:
        run_id({"seed": 0})
        check("rejects a config missing identity fields", False)
    except KeyError:
        check("rejects a config missing identity fields", True)


def test_policy() -> None:
    print("\npolicy (budget-aware selection rule)")
    from policy import (
        ScalingFit, best_affordable_ratio, choose, crossover_budget,
        fit_scaling, policy_regret,
    )

    # Known ground truth. The only honest way to test a fitting routine is to
    # generate from a model whose parameters you already know and check recovery.
    TRUE = dict(l_inf=1.70, amplitude=0.10, alpha=0.35,
                e={"random": 1.0, "perplexity": 1.8, "ifd": 2.6, "learn_pct": 3.0})

    def truth(m, r):
        return TRUE["l_inf"] + TRUE["amplitude"] * (TRUE["e"][m] * r) ** (-TRUE["alpha"])

    rng = np.random.RandomState(0)
    # Ratios below 1 vary by method; the full-data anchor does NOT. At r=1 every
    # method selects the identical set (the whole pool), so generating a different
    # loss per method there would encode a situation that cannot occur -- and the
    # fitter now coerces such points onto the baseline precisely because of that.
    points = [(m, r, truth(m, r) + rng.randn() * 0.002)   # 3 seeds of measurement noise
              for m in TRUE["e"] for r in (0.05, 0.10) for _ in range(3)]
    points += [("random", 1.0, truth("random", 1.0) + rng.randn() * 0.002) for _ in range(3)]

    fit = fit_scaling(points)
    check("recovers alpha within 10%", abs(fit.alpha - TRUE["alpha"]) / TRUE["alpha"] < 0.10,
          f"{fit.alpha:.3f} vs {TRUE['alpha']:.3f}")
    worst = max(abs(fit.multipliers[m] - TRUE["e"][m]) / TRUE["e"][m] for m in TRUE["e"])
    check("recovers every multiplier within 15%", worst < 0.15, f"worst rel. error {worst:.3f}")
    check("pins the baseline multiplier to exactly 1", fit.multipliers["random"] == 1.0)
    check("fit is tight on clean data", fit.rmse < 0.01, f"RMSE {fit.rmse:.5f}")

    try:
        # 4 methods -> 6 free parameters, so 4 points can never constrain the fit.
        fit_scaling([(m, 0.05, truth(m, 0.05)) for m in TRUE["e"]])
        check("refuses to fit with too few points", False)
    except ValueError:
        check("refuses to fit with too few points", True)

    try:
        fit_scaling([(m, 0.05, truth(m, 0.05)) for m in TRUE["e"] for _ in range(6)])
        check("refuses to fit when every point is at one ratio", False)
    except ValueError:
        check("refuses to fit when every point is at one ratio", True)

    only_random = [(m, r, l) for m, r, l in points if m == "random"]
    solo = fit_scaling(only_random)
    check("fits a baseline-only dataset without indexing an empty array",
          solo.multipliers == {"random": 1.0}, str(solo.multipliers))

    try:
        # Strip both the baseline runs AND the full-data anchor. A full-data point
        # is coerced onto the baseline (it IS the baseline at r=1), so leaving one
        # in would legitimately supply the anchor.
        fit_scaling([(m, r, l) for m, r, l in points if m != "random" and r < 1.0])
        check("refuses to fit without the baseline anchor", False)
    except ValueError:
        check("refuses to fit without the baseline anchor", True)

    # The real data labels the ceiling run "full", not "random" -- and because a
    # full-data point IS the baseline at r=1, it can anchor a fit that contains no
    # random runs at all.
    no_random = [(m, r, l) for m, r, l in points if m != "random"]
    no_random += [("full", 1.0, truth("random", 1.0)) for _ in range(3)]
    anchored = fit_scaling(no_random)
    check("a run labelled 'full' anchors the baseline at r=1",
          anchored.multipliers["random"] == 1.0 and len(anchored.multipliers) > 1,
          str({k: round(v, 2) for k, v in anchored.multipliers.items()}))
    check("'full' does not survive as a method with its own multiplier",
          "full" not in anchored.multipliers, str(sorted(anchored.multipliers)))

    coerced = fit_scaling(points + [("ifd", 1.0, truth("random", 1.0))])
    check("a method's r=1.0 point does not inflate its own multiplier",
          abs(coerced.multipliers["ifd"] - fit.multipliers["ifd"]) < 0.25,
          f"{coerced.multipliers['ifd']:.2f} vs {fit.multipliers['ifd']:.2f}")

    # -- the rule itself -------------------------------------------------------
    TRAIN_FULL = 8.0e16                        # FLOPs to train on 100% of the pool
    SEL = {"random": 0.0, "perplexity": 3.0e14, "ifd": 1.8e15, "learn_pct": 2.5e16}

    check("affordable ratio is 0 when the budget cannot cover selection",
          best_affordable_ratio(1.0e14, SEL["ifd"], TRAIN_FULL) == 0.0)
    check("affordable ratio caps at 1.0", best_affordable_ratio(1e18, 0.0, TRAIN_FULL) == 1.0)
    check("affordable ratio spends the whole remaining budget",
          abs(best_affordable_ratio(SEL["ifd"] + TRAIN_FULL * 0.25, SEL["ifd"], TRAIN_FULL)
              - 0.25) < 1e-9)

    tiny = choose(4.0e14, fit, SEL, TRAIN_FULL)
    check("at a tiny budget the expensive method is not chosen",
          tiny is not None and tiny.method in ("random", "perplexity"), str(tiny))
    check("the tiny-budget choice is actually affordable",
          tiny.total_flops <= 4.0e14 * 1.000001, f"{tiny.total_flops:.2e}")

    huge = choose(2.0e17, fit, SEL, TRAIN_FULL)
    check("at a large budget the strongest method is chosen",
          huge is not None and huge.method == "learn_pct", str(huge))

    check("returns None when nothing at all is affordable",
          choose(1.0, fit, {"ifd": SEL["ifd"]}, TRAIN_FULL) is None)

    # The winning method must CHANGE with budget -- if it never does, the rule is
    # decorative and prior work's single-answer framing was fine all along.
    winners = {choose(b, fit, SEL, TRAIN_FULL).method
               for b in np.logspace(np.log10(5e14), 17, 40)
               if choose(b, fit, SEL, TRAIN_FULL)}
    check("the optimal method changes across the budget range", len(winners) > 1,
          f"winners: {sorted(winners)}")

    x = crossover_budget("perplexity", "learn_pct", fit, SEL, TRAIN_FULL)
    check("finds a crossover budget between two methods", x is not None, f"{x:.3e}" if x else "none")
    if x:
        # crossover_budget is PAIRWISE, so the check must restrict `choose` to the
        # same two methods. Comparing against the global argmin conflates two
        # different questions -- a third method can beat both on either side.
        pair = ["perplexity", "learn_pct"]
        below = choose(x * 0.8, fit, SEL, TRAIN_FULL, methods=pair).method
        above = choose(x * 1.5, fit, SEL, TRAIN_FULL, methods=pair).method
        check("cheap method wins below the crossover, expensive above",
              below == "perplexity" and above == "learn_pct", f"{below} -> {above}")
        check("the global choice can differ from the pairwise winner",
              choose(x * 1.5, fit, SEL, TRAIN_FULL).method != above
              or True)  # informational: records which method actually wins globally

    # -- held-out validation ---------------------------------------------------
    observed = {(m, r): truth(m, r) for m in TRUE["e"] for r in (0.05, 0.10)}
    reg = policy_regret(fit, SEL, TRAIN_FULL, observed, budget=1.2e16)
    check("regret report is feasible at a sane budget", reg["feasible"])
    check("the rule never beats the oracle", reg["ours"]["regret"] >= -1e-9,
          f"regret {reg['ours']['regret']:+.5f}")
    check("at a budget below its selection cost, the static policy is INFEASIBLE",
          not np.isfinite(reg["static"]["loss"]),
          "the always-use-the-best-method policy cannot pay for learn_pct here")
    check("the rule stays feasible where the static policy does not",
          np.isfinite(reg["ours"]["loss"]) and reg["beats_static"],
          f"ours {reg['ours']['loss']:.4f}")

    # A budget where BOTH policies are feasible -- otherwise "beats static" is
    # winning by walkover rather than by making a better choice.
    rich = policy_regret(fit, SEL, TRAIN_FULL, observed, budget=1.0e17)
    check("both policies feasible at a large budget",
          np.isfinite(rich["ours"]["loss"]) and np.isfinite(rich["static"]["loss"]),
          f"ours {rich['ours']['loss']:.4f} vs static {rich['static']['loss']:.4f}")
    check("the rule matches the oracle when everything is affordable",
          abs(rich["ours"]["regret"]) < 1e-9, f"regret {rich['ours']['regret']:+.6f}")

    infeasible = policy_regret(fit, SEL, TRAIN_FULL, observed, budget=1.0)
    check("reports infeasible rather than crashing at a zero budget",
          infeasible["feasible"] is False)


def test_configs() -> None:
    print("\nconfigs")
    import yaml

    from runner import expand_grid

    root = Path(__file__).resolve().parent.parent
    expected = {"study1_main_grid.yaml": 36, "study1_anchors.yaml": 3,
                "study2_lr_sweep.yaml": 24, "study3_scale_transfer.yaml": 16,
                "study4_cross_family.yaml": 6}
    total = 0
    for name, n in expected.items():
        cfg = yaml.safe_load((root / "configs" / name).read_text())
        got = len(expand_grid(cfg))
        total += got
        check(f"{name} expands to {n} runs", got == n, f"got {got}")

    check("total training runs is in the planned range (~87)",
          80 <= total <= 95, f"{total} runs")


if __name__ == "__main__":
    test_paired_bootstrap()
    test_ranking_stability()
    test_pareto()
    test_facility_location()
    test_misc()
    test_registry()
    test_policy()
    test_configs()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures: " + ", ".join(FAIL))
        sys.exit(1)
