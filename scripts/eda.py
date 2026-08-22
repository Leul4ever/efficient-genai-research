"""EDA (Exploratory Data Analysis) for the paper.

Produces results/figures/ plots:
  eda1_dataset_overview.png  -- instruction/response length distributions + category pie
  eda2_score_distributions.png -- selection pool position histograms per method
  eda3_selection_overlap.png -- Jaccard heatmap between methods at ratio=0.05, seed=0
  eda4_length_bias.png -- response length: perplexity vs random vs pool
  eda5_cost_breakdown.png -- stacked bar: selection FLOPs vs training FLOPs per method
  eda6_seed_variance.png -- box plot of held-out loss across seeds per method

Run from the project root:
    python scripts/eda.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_split, load_train_examples  # noqa: E402
from paths import SELECTIONS  # noqa: E402

FIGDIR = ROOT / "results" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style (matches figures.py)
# ---------------------------------------------------------------------------
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#b5b3ab"
SURFACE  = "#fcfcfb"
BLUE     = "#2a78d6"
ORANGE   = "#eb6834"
GREEN    = "#1baf7a"
AMBER    = "#eda100"
PINK     = "#e87ba4"
TEAL     = "#008300"

METHOD_COLORS = {
    "random":              BLUE,
    "perplexity":          ORANGE,
    "ifd":                 GREEN,
    "diversity":           AMBER,
    "hybrid":              PINK,
    "learning_percentage": TEAL,
}
CLASS_COLOR = {"free": BLUE, "training-free": GREEN, "training-based": ORANGE}

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
    "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        p = FIGDIR / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        print(f"  wrote {p.relative_to(ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
examples  = load_train_examples()
split     = load_split()

instr_lens = [len(e["instruction"].split()) for e in examples]
resp_lens  = [len(e["response"].split()) for e in examples]
categories = {}
for e in examples:
    cat = e.get("category", "unknown")
    categories[cat] = categories.get(cat, 0) + 1

print(f"  pool size: {len(examples)}")
print(f"  instruction length: mean={np.mean(instr_lens):.1f}  median={np.median(instr_lens):.1f}")
print(f"  response length:    mean={np.mean(resp_lens):.1f}  median={np.median(resp_lens):.1f}")


# ---------------------------------------------------------------------------
# EDA 1 -- Dataset overview
# ---------------------------------------------------------------------------
print("\nEDA 1: dataset overview...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

ax = axes[0]
ax.hist(instr_lens, bins=60, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.4)
ax.set_xlabel("Instruction length (words)")
ax.set_ylabel("Count")
ax.set_title("Instruction length distribution", loc="left")
ax.axvline(np.median(instr_lens), color=INK_2, lw=1.2, ls="--")
ax.text(np.median(instr_lens) + 2, ax.get_ylim()[1] * 0.9,
        f"median={np.median(instr_lens):.0f}", fontsize=7.5, color=INK_2)
ax.grid(axis="y")
ax.set_axisbelow(True)

ax = axes[1]
ax.hist(resp_lens, bins=80, color=ORANGE, alpha=0.85, edgecolor=SURFACE, linewidth=0.4)
ax.set_xlabel("Response length (words)")
ax.set_ylabel("Count")
ax.set_title("Response length distribution", loc="left")
ax.axvline(np.median(resp_lens), color=INK_2, lw=1.2, ls="--")
ax.text(np.median(resp_lens) + 3, ax.get_ylim()[1] * 0.9,
        f"median={np.median(resp_lens):.0f}", fontsize=7.5, color=INK_2)
ax.grid(axis="y")
ax.set_axisbelow(True)

ax = axes[2]
top_n = 10
sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
top_cats = sorted_cats[:top_n]
other_count = sum(v for _, v in sorted_cats[top_n:])
labels = [k for k, _ in top_cats] + (["other"] if other_count else [])
counts = [v for _, v in top_cats] + ([other_count] if other_count else [])
colors_pie = [plt.cm.tab20(i / max(len(labels), 1)) for i in range(len(labels))]
wedges, texts, autotexts = ax.pie(
    counts, labels=None, autopct="%1.0f%%", colors=colors_pie,
    startangle=90, pctdistance=0.82,
    wedgeprops={"edgecolor": SURFACE, "linewidth": 1.2},
)
for t in autotexts:
    t.set_fontsize(6.5)
ax.legend(wedges, labels, loc="lower left", bbox_to_anchor=(-0.25, -0.18),
          fontsize=6.5, ncol=2, frameon=False)
ax.set_title("Category distribution (top 10)", loc="left")

fig.suptitle(f"Databricks-Dolly-15k training pool  (n={len(examples):,})",
             x=0.01, ha="left", fontsize=11, y=1.02)
save(fig, "eda1_dataset_overview")


# ---------------------------------------------------------------------------
# EDA 2 -- Pool position distributions per method
# ---------------------------------------------------------------------------
print("EDA 2: score distributions...")

sel_info = {}
for method in ("random", "perplexity", "ifd", "diversity"):
    p = SELECTIONS / f"{method}__r0.05__s0.json"
    if p.exists():
        d = json.loads(p.read_text())
        sel_info[method] = {
            "local_idx": set(d["local_idx"]),
            "k": d["k"],
            "pool": d["pool_size"],
        }

if len(sel_info) >= 2:
    n_methods = len(sel_info)
    fig, axes = plt.subplots(1, n_methods, figsize=(3.5 * n_methods, 4.0))
    if n_methods == 1:
        axes = [axes]

    for ax, (method, info) in zip(axes, sorted(sel_info.items())):
        selected     = sorted(info["local_idx"])
        not_selected = [i for i in range(info["pool"]) if i not in info["local_idx"]]
        ax.hist(not_selected, bins=100, alpha=0.40, color=MUTED, label="not selected",
                density=True, edgecolor="none")
        ax.hist(selected, bins=40, alpha=0.85,
                color=METHOD_COLORS.get(method, BLUE), label="selected",
                density=True, edgecolor=SURFACE, linewidth=0.3)
        ax.set_xlabel("Pool index")
        ax.set_ylabel("Density")
        ax.set_title(f"{method}  (k={info['k']})", loc="left")
        ax.legend(frameon=False, fontsize=7.5)
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    fig.suptitle("Where each method selects from the pool  (ratio=5%, seed=0)",
                 x=0.01, ha="left", fontsize=11, y=1.02)
    save(fig, "eda2_score_distributions")
else:
    print("  not enough selection files -- skipping eda2")


# ---------------------------------------------------------------------------
# EDA 3 -- Selection overlap (Jaccard heatmap)
# ---------------------------------------------------------------------------
print("EDA 3: selection overlap...")

methods_for_jaccard = [
    "random", "perplexity", "ifd", "diversity", "hybrid", "learning_percentage"
]
ratio_j, seed_j = 0.05, 0
sel_sets = {}
for method in methods_for_jaccard:
    p = SELECTIONS / f"{method}__r{ratio_j}__s{seed_j}.json"
    if p.exists():
        d = json.loads(p.read_text())
        sel_sets[method] = set(d["local_idx"])

avail = list(sel_sets.keys())
n = len(avail)
if n >= 2:
    J = np.zeros((n, n))
    for i, a in enumerate(avail):
        for j, b in enumerate(avail):
            inter = len(sel_sets[a] & sel_sets[b])
            union = len(sel_sets[a] | sel_sets[b])
            J[i, j] = inter / union if union else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(J, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), avail, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(n), avail, fontsize=8)
    for i in range(n):
        for j in range(n):
            txt = f"{J[i, j]:.3f}" if i != j else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if J[i, j] > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Jaccard similarity", fontsize=8)
    cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    k_example = len(list(sel_sets.values())[0])
    ax.set_title(
        f"Pairwise Jaccard similarity between selections\n"
        f"(ratio={ratio_j}, seed={seed_j}, k={k_example} from {list(sel_info.values())[0]['pool']:,})",
        loc="left", fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "eda3_selection_overlap")
else:
    print(f"  only {n} selection files found -- need >= 2, skipping")


# ---------------------------------------------------------------------------
# EDA 4 -- Length bias in perplexity selection
# ---------------------------------------------------------------------------
print("EDA 4: length bias...")

perp_path = SELECTIONS / "perplexity__r0.05__s0.json"
rand_path  = SELECTIONS / "random__r0.05__s0.json"

if perp_path.exists() and rand_path.exists():
    perp_sel = set(json.loads(perp_path.read_text())["local_idx"])
    rand_sel  = set(json.loads(rand_path.read_text())["local_idx"])

    perp_lengths = [resp_lens[i] for i in range(len(examples)) if i in perp_sel]
    rand_lengths  = [resp_lens[i] for i in range(len(examples)) if i in rand_sel]
    rest_lengths  = [resp_lens[i] for i in range(len(examples))
                     if i not in perp_sel and i not in rand_sel]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    bins = np.linspace(0, min(max(resp_lens), 800), 60)
    ax.hist(rest_lengths, bins=bins, alpha=0.4, color=MUTED, label="not selected",
            density=True, edgecolor="none")
    ax.hist(rand_lengths, bins=bins, alpha=0.65, color=BLUE,   label="random",
            density=True, edgecolor="none")
    ax.hist(perp_lengths, bins=bins, alpha=0.75, color=ORANGE, label="perplexity",
            density=True, edgecolor="none")
    ax.set_xlabel("Response length (words)")
    ax.set_ylabel("Density")
    ax.set_title("Perplexity selection is length-biased", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    groups  = ["Pool\n(all)", "Random\nselected", "Perplexity\nselected"]
    medians = [np.median(resp_lens), np.median(rand_lengths), np.median(perp_lengths)]
    means   = [np.mean(resp_lens),   np.mean(rand_lengths),   np.mean(perp_lengths)]
    colors_b = [MUTED, BLUE, ORANGE]
    x = np.arange(len(groups))
    ax.bar(x, medians, color=colors_b, alpha=0.85, edgecolor=SURFACE, linewidth=1.2)
    ax.scatter(x, means, marker="D", s=45, color=INK, zorder=4, label="mean")
    for xi, (m, mn) in enumerate(zip(medians, means)):
        ax.text(xi, m + 3, f"med={m:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylabel("Response length (words)")
    ax.set_title("Median / mean response length by group", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.suptitle("Perplexity-based selection is biased toward longer responses",
                 x=0.01, ha="left", fontsize=11, y=1.02)
    save(fig, "eda4_length_bias")
else:
    print("  selection files missing -- skipping eda4")


# ---------------------------------------------------------------------------
# EDA 5 -- Cost breakdown per method
# ---------------------------------------------------------------------------
print("EDA 5: cost breakdown...")

sel_cost_data = {}
for method in ("random", "perplexity", "ifd", "diversity", "learning_percentage"):
    p = SELECTIONS / f"{method}__r0.05__s0.json"
    if p.exists():
        d = json.loads(p.read_text())
        sel_cost_data[method] = {
            "sel_flops":  d["cost"].get("flops", 0),
            "cost_class": d.get("cost_class", "?"),
        }

runs_file = ROOT / "results" / "runs.jsonl"
training_cost = {}
if runs_file.exists():
    for line in runs_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("study") != "fast_grid":
            continue
        m = r["config"]["selection_method"]
        if r["config"]["ratio"] == 0.05 and r["config"].get("seed") == 0:
            training_cost[m] = r.get("cost", {}).get("training_flops", 0)

methods_plot = [
    m for m in ("random", "perplexity", "ifd", "diversity", "learning_percentage")
    if m in sel_cost_data and m in training_cost
]

if methods_plot:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sel_flops   = np.array([sel_cost_data[m]["sel_flops"]  for m in methods_plot])
    train_flops = np.array([training_cost[m]               for m in methods_plot])
    total_flops = sel_flops + train_flops
    sel_share   = sel_flops / np.where(total_flops > 0, total_flops, 1)
    x = np.arange(len(methods_plot))

    ax = axes[0]
    ax.bar(x, train_flops / 1e14, color=BLUE,   alpha=0.85,
           label="training FLOPs", edgecolor=SURFACE, lw=1)
    ax.bar(x, sel_flops / 1e14,   color=ORANGE, alpha=0.85,
           label="selection FLOPs", bottom=train_flops / 1e14,
           edgecolor=SURFACE, lw=1)
    ax.set_xticks(x, methods_plot, rotation=20, ha="right")
    ax.set_ylabel("FLOPs (x10^14)")
    ax.set_title("Total compute per method  (ratio=5%, seed=0)", loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax = axes[1]
    bar_colors = [CLASS_COLOR.get(sel_cost_data[m]["cost_class"], MUTED) for m in methods_plot]
    ax.bar(x, sel_share * 100, color=bar_colors, alpha=0.85, edgecolor=SURFACE, lw=1)
    for xi, v in enumerate(sel_share):
        ax.text(xi, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, methods_plot, rotation=20, ha="right")
    ax.set_ylabel("Selection share of total FLOPs (%)")
    ax.set_title("How much of total compute is selection?", loc="left")
    ax.set_ylim(0, 105)
    handles = [mpatches.Patch(color=CLASS_COLOR[c], alpha=0.85, label=c)
               for c in CLASS_COLOR
               if any(sel_cost_data[m]["cost_class"] == c for m in methods_plot)]
    ax.legend(handles=handles, frameon=False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    fig.suptitle("Selection is NOT free: it dominates total compute for scored methods",
                 x=0.01, ha="left", fontsize=11, y=1.02)
    save(fig, "eda5_cost_breakdown")
else:
    print(f"  insufficient data -- skipping eda5 (methods_plot={methods_plot})")


# ---------------------------------------------------------------------------
# EDA 6 -- Seed variance per method
# ---------------------------------------------------------------------------
print("EDA 6: seed variance...")

runs_by_method: dict = defaultdict(list)
if runs_file.exists():
    for line in runs_file.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") != "ok" or r.get("study") != "fast_grid":
            continue
        m     = r["config"]["selection_method"]
        ratio = r["config"]["ratio"]
        loss  = r["metrics"].get("held_out_loss")
        if loss is not None and np.isfinite(loss):
            runs_by_method[(m, ratio)].append(float(loss))

plot_data = {k: v for k, v in runs_by_method.items() if len(v) >= 2}

if plot_data:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, ratio in zip(axes, [0.05, 0.1]):
        sub = {m: v for (m, r), v in plot_data.items() if r == ratio}
        if not sub:
            ax.set_visible(False)
            continue
        methods_sv = sorted(sub.keys())
        data_sv    = [sub[m] for m in methods_sv]
        colors_sv  = [METHOD_COLORS.get(m, BLUE) for m in methods_sv]
        bp = ax.boxplot(data_sv, patch_artist=True, widths=0.5,
                        medianprops={"color": INK, "lw": 2},
                        whiskerprops={"color": INK_2},
                        capprops={"color": INK_2},
                        flierprops={"marker": "o", "markerfacecolor": INK_2, "ms": 4})
        for patch, col in zip(bp["boxes"], colors_sv):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
            patch.set_edgecolor(SURFACE)
        for i, (m, vals) in enumerate(zip(methods_sv, data_sv), 1):
            ax.scatter([i] * len(vals), vals,
                       color=METHOD_COLORS.get(m, BLUE),
                       s=35, zorder=4, edgecolors=SURFACE, lw=0.8)
        ax.set_xticks(range(1, len(methods_sv) + 1), methods_sv, rotation=20, ha="right")
        ax.set_ylabel("Held-out loss" if ratio == 0.05 else "")
        ax.set_title(f"ratio={ratio}", loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)

    fig.suptitle("Held-out loss across seeds per method  (lower spread = more stable)",
                 x=0.01, ha="left", fontsize=11, y=1.02)
    save(fig, "eda6_seed_variance")
else:
    print("  not enough seed-complete runs yet -- will re-run after seed 2 finishes")


print("\nAll EDA figures done.")
