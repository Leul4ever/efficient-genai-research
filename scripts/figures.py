"""Stage 4: results/runs.jsonl -> the paper's three figures.

    python scripts/figures.py            # from real results
    python scripts/figures.py --demo     # synthetic data, to check layout before Week 3

Design notes, so nobody "improves" these into unreadability later:

* Figure 1 is TWO PANELS sharing a y-axis, not one panel with two x-scales. The
  argument is "the frontier changes when you price selection in", and a dual-axis
  chart would make that comparison impossible to read. Two panels, one x-scale each.
* Colour encodes COST CLASS, not method. Three classes is also what keeps the
  categorical palette inside its validated all-pairs limit for a scatter, but the
  real reason is that cost class is the axis the paper's argument rides on.
* Every point and line is directly labelled. Three of the palette slots sit below
  3:1 contrast on a light surface, so identity may never rest on colour alone.
* Figure 2's heatmap is DIVERGING (two hues, neutral grey at zero) because Spearman
  is a polarity scale on [-1, 1]. A sequential ramp here would hide sign.
* Light mode only, deliberately: these are vector figures for a PDF paper, not a
  web page. Both PDF (for LaTeX) and 300 dpi PNG are written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from registry import load_runs  # noqa: E402
from stats import pareto_frontier, ranking_stability  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "results" / "figures"

# Validated categorical slots (light surface #fcfcfb). Assigned in fixed order,
# never cycled. See scripts/validate_palette.js in the dataviz reference.
CLASS_COLOR = {
    "free": "#2a78d6",            # slot 1 blue
    "training-free": "#1baf7a",   # slot 3 aqua
    "training-based": "#eb6834",  # slot 2 orange
    "none": "#52514e",            # full-data ceiling: text-secondary, not a series hue
}
# Secondary encoding, so cost class survives greyscale print and CVD.
CLASS_MARKER = {"free": "o", "training-free": "s", "training-based": "^", "none": "D"}

METHOD_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # adjacent-validated

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#b5b3ab"
SURFACE = "#fcfcfb"

# Diverging pair for Spearman in [-1, 1]: red (disagreement) - grey - blue (agreement).
DIVERGING = LinearSegmentedColormap.from_list(
    "spearman", ["#e34948", "#f2f1ee", "#2a78d6"]
)


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.labelcolor": INK, "text.color": INK,
        "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
        "lines.linewidth": 2.0, "lines.markersize": 8,
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGDIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        print(f"  wrote {path.relative_to(FIGDIR.parent.parent)}")
    plt.close(fig)


# --------------------------------------------------------------------- figure 1
def figure1_pareto(rows: list[dict]) -> None:
    """Total-FLOPs vs quality. The paper's central cost claim."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)

    for ax, xkey, title in (
        (axes[0], "train", "Training cost only\n(how prior work accounts)"),
        (axes[1], "total", "Training + selection cost\n(what it actually costs)"),
    ):
        x = np.array([r[xkey] for r in rows])
        y = np.array([r["loss"] for r in rows])
        front = pareto_frontier(x, y)

        fx, fy = x[front], y[front]
        order = np.argsort(fx)
        ax.step(fx[order], fy[order], where="post", color=MUTED, lw=1.4, zorder=1)

        for i, r in enumerate(rows):
            on = i in front
            ax.scatter(x[i], y[i], s=88 if on else 52,
                       c=CLASS_COLOR[r["class"]], marker=CLASS_MARKER[r["class"]],
                       edgecolors=SURFACE, linewidths=1.6,  # 2px surface ring on overlap
                       alpha=1.0 if on else 0.55, zorder=3 if on else 2)
            # Direct label: identity never rests on colour (three slots are <3:1).
            if on:
                ax.annotate(r["label"], (x[i], y[i]), textcoords="offset points",
                            xytext=(9, 4), fontsize=7.5, color=INK, zorder=4)

        ax.set_xscale("log")
        ax.set_xlabel("FLOPs")
        ax.set_title(title, color=INK, pad=10, loc="left", fontsize=9.5)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Held-out loss  (lower is better)")

    handles = [Line2D([], [], marker=CLASS_MARKER[c], color="none",
                      markerfacecolor=CLASS_COLOR[c], markeredgecolor=SURFACE,
                      markersize=9, label=c)
               for c in ("free", "training-free", "training-based", "none")
               if any(r["class"] == c for r in rows)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, -0.06), handletextpad=0.4, columnspacing=1.6)
    fig.suptitle("Pricing selection in pushes the training-based method off the frontier",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.04)
    save(fig, "fig1_pareto")


# --------------------------------------------------------------------- figure 2
def lr_order(by_lr: dict) -> list[str]:
    """Sort "lr=..." labels NUMERICALLY. Alphabetical order puts 2e-04 before
    2e-05, which turns the bump chart into a trajectory that never happened."""
    return sorted(by_lr, key=lambda s: float(s.split("=", 1)[1]))


def figure2_stability(by_lr: dict[str, dict[str, float]]) -> None:
    """RQ1: does the method ranking move when the learning rate moves?"""
    r = ranking_stability(by_lr, higher_is_better=False, order=lr_order(by_lr))
    labels, methods = r["conditions"], r["methods"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2),
                             gridspec_kw={"width_ratios": [1.0, 1.3], "wspace": 0.42})

    # -- left: Spearman matrix, diverging around zero
    n = len(labels)
    m = np.eye(n)
    for p in r["pairwise"]:
        i, j = labels.index(p["condition_a"]), labels.index(p["condition_b"])
        m[i, j] = m[j, i] = p["spearman"]

    ax = axes[0]
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels, rotation=30, ha="right")
    ax.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            # Text wears ink tokens, never the cell colour.
            ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color=INK if abs(m[i, j]) < 0.6 else "#ffffff")
    ax.set_title("Rank correlation between\nlearning-rate conditions",
                 loc="left", pad=10, fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05,
                      pad=0.22, ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["reversed", "unrelated", "identical"], fontsize=7.5)
    cb.ax.tick_params(length=0)
    cb.outline.set_visible(False)

    # -- right: bump chart. A heatmap says "how much"; this says "what moved".
    ax = axes[1]
    ranks = {m_: [r["orderings"][c].index(m_) + 1 for c in labels] for m_ in methods}
    for k, (name, seq) in enumerate(sorted(ranks.items())):
        color = METHOD_COLOR[k % len(METHOD_COLOR)]
        ax.plot(range(n), seq, "-o", color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.6, zorder=3)
        ax.annotate(f" {name}", (n - 1, seq[-1]), fontsize=8, color=INK,
                    va="center", ha="left")

    ax.set_xticks(range(n), labels, rotation=30, ha="right")
    ax.set_yticks(range(1, len(methods) + 1))
    ax.set_ylabel("Rank  (1 = best)")
    ax.invert_yaxis()
    ax.set_xlim(-0.25, n - 0.25 + 0.9)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Where each method ranks\nat each learning rate", loc="left",
                 pad=10, fontsize=9.5)

    verdict = ("rankings HELD" if r["stable_at_0.95"] else "rankings MOVED")
    fig.suptitle(f"Across the learning-rate sweep, {verdict} "
                 f"(min Spearman {r['min_spearman']:+.2f})",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.02)
    save(fig, "fig2_ranking_stability")


# --------------------------------------------------------------------- figure 3
def figure3_transfer(series: dict[str, list[tuple[float, float, str]]]) -> None:
    """RQ3: does the proxy signal survive a growing size gap?"""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    for k, (name, pts) in enumerate(sorted(series.items())):
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = METHOD_COLOR[k % len(METHOD_COLOR)]
        ax.plot(xs, ys, "-o", color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.6, zorder=3)
        ax.annotate(f"  {name}", (xs[-1], ys[-1]), fontsize=8.5, color=INK,
                    va="center", ha="left")

    ax.axhline(0, color=MUTED, lw=1.2, ls="--", zorder=1)
    # Axes-fraction x, data y: pinning this to a data coordinate breaks as soon as
    # the x-limits are extended below to make room for the direct labels.
    ax.annotate("no better than random", xy=(0.012, 0), xycoords=("axes fraction", "data"),
                fontsize=7.5, color=INK_2, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("Target model parameters")
    ax.set_ylabel("Held-out loss vs random\n(negative = selection helps)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax * 1.55)  # just enough for the right-hand direct labels
    fig.suptitle("Selection advantage as the proxy-to-target gap widens",
                 x=0.005, ha="left", fontsize=11, color=INK, y=1.0)
    save(fig, "fig3_transfer")


# ------------------------------------------------------------------------ data
def rows_from_runs(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        c, cost = r["config"], r["cost"]
        loss = r["metrics"].get("held_out_loss")
        if loss is None or not np.isfinite(loss):
            continue
        out.append({
            "label": f"{c['selection_method']}@{c['ratio']:g}",
            "class": cost.get("selection_cost_class", "none"),
            "train": cost["training_flops"], "total": cost["total_flops"],
            "loss": float(loss),
        })
    return out


def demo_data():
    """Synthetic data with the shape H1/H2/H3 predict. For checking layout only --
    every figure rendered from this is watermarked, so it can never be mistaken for
    a result."""
    rows = [
        {"label": "random@0.05", "class": "free", "train": 4.1e15, "total": 4.1e15, "loss": 1.94},
        {"label": "perplexity@0.05", "class": "training-free", "train": 4.0e15, "total": 4.3e15, "loss": 1.89},
        {"label": "diversity@0.05", "class": "training-free", "train": 4.0e15, "total": 4.2e15, "loss": 1.91},
        {"label": "hybrid@0.05", "class": "training-free", "train": 4.0e15, "total": 4.5e15, "loss": 1.88},
        {"label": "ifd@0.05", "class": "training-free", "train": 4.0e15, "total": 5.8e15, "loss": 1.87},
        {"label": "learn%@0.05", "class": "training-based", "train": 4.0e15, "total": 2.9e16, "loss": 1.86},
        {"label": "random@0.10", "class": "free", "train": 8.2e15, "total": 8.2e15, "loss": 1.87},
        {"label": "ifd@0.10", "class": "training-free", "train": 8.1e15, "total": 9.9e15, "loss": 1.84},
        {"label": "full@1.0", "class": "none", "train": 8.2e16, "total": 8.2e16, "loss": 1.82},
    ]
    by_lr = {
        "lr=1e-06": {"random": 1.99, "perplexity": 1.96, "ifd": 1.94, "learn%": 1.95},
        "lr=2e-05": {"random": 1.95, "perplexity": 1.92, "ifd": 1.91, "learn%": 1.93},
        "lr=2e-04": {"random": 1.88, "perplexity": 1.91, "ifd": 1.93, "learn%": 1.90},
    }
    transfer = {
        "perplexity": [(4.94e8, -0.05, ""), (1.24e9, -0.03, ""), (1.54e9, -0.02, "")],
        "ifd": [(4.94e8, -0.07, ""), (1.24e9, -0.02, ""), (1.54e9, -0.03, "")],
        "learn%": [(4.94e8, -0.08, ""), (1.24e9, 0.01, ""), (1.54e9, -0.01, "")],
        "diversity": [(4.94e8, -0.03, ""), (1.24e9, -0.01, ""), (1.54e9, 0.00, "")],
    }
    return rows, by_lr, transfer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="render from synthetic data to check layout before results exist")
    args = ap.parse_args()
    style()

    if args.demo:
        rows, by_lr, transfer = demo_data()
        print("rendering DEMO figures from synthetic data (watermarked):")
        _original_save = globals()["save"]

        def save_with_watermark(fig, name):
            fig.text(0.5, 0.5, "SYNTHETIC", fontsize=54, color="#e34948", alpha=0.12,
                     ha="center", va="center", rotation=28, zorder=100,
                     transform=fig.transFigure)
            _original_save(fig, name + "_demo")

        globals()["save"] = save_with_watermark
        figure1_pareto(rows)
        figure2_stability(by_lr)
        figure3_transfer(transfer)
        globals()["save"] = _original_save
        return

    runs = [r for r in load_runs() if r.get("status") == "ok"]
    if not runs:
        print("no completed runs yet. Use --demo to check figure layout meanwhile.")
        return

    print(f"rendering figures from {len(runs)} runs:")
    rows = rows_from_runs([r for r in runs if r["study"] == "study1_main_grid"])
    if rows:
        figure1_pareto(rows)
    else:
        print("  fig1 skipped: no study1 runs")

    sweep = [r for r in runs if r["study"] == "study2_lr_sweep"]
    by_lr: dict[str, dict[str, list[float]]] = {}
    for r in sweep:
        c = r["config"]
        loss = r["metrics"].get("held_out_loss")
        if loss is not None and np.isfinite(loss):
            by_lr.setdefault(f"lr={c['learning_rate']:g}", {}) \
                 .setdefault(c["selection_method"], []).append(float(loss))
    by_lr_mean = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in by_lr.items()}
    common = set.intersection(*(set(v) for v in by_lr_mean.values())) if by_lr_mean else set()
    if len(common) >= 3 and len(by_lr_mean) >= 2:
        figure2_stability({k: {m: v[m] for m in common} for k, v in by_lr_mean.items()})
    else:
        print(f"  fig2 skipped: {len(by_lr_mean)} LRs x {len(common)} shared methods")

    print("  fig3: needs Studies 1+3+4 complete; see analyze.py for the same numbers")


if __name__ == "__main__":
    main()
