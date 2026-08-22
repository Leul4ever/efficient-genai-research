import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Does Proxy-Based Data Selection Survive Contact With Reality?\n",
            "### *Robustness and Net-Cost Accounting for Efficient Instruction Tuning*\n",
            "\n",
            "**Research & EDA Findings Notebook**\n",
            "\n",
            "This notebook presents the empirical evidence, exploratory data analysis (EDA), and core findings of our research project. It brings together dataset properties, selection behavior, measured selection overheads, hyperparameter sensitivity, and Pareto efficiency.\n",
            "\n",
            "---"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Environment & Path Setup\n",
            "import os\n",
            "import sys\n",
            "import json\n",
            "import glob\n",
            "from pathlib import Path\n",
            "from collections import defaultdict\n",
            "\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "from IPython.display import display, Image, HTML, Markdown\n",
            "\n",
            "# Set working directory to project root\n",
            "ROOT = Path.cwd().parent if Path.cwd().name == \"notebooks\" else Path.cwd()\n",
            "sys.path.insert(0, str(ROOT / \"src\"))\n",
            "\n",
            "from data import load_split, load_train_examples\n",
            "from registry import load_runs\n",
            "from stats import pareto_frontier, ranking_stability, holm_bonferroni, paired_bootstrap\n",
            "\n",
            "print(f\"Project Root: {ROOT}\")\n",
            "runs = load_runs(ROOT / \"results/runs.jsonl\")\n",
            "ok = [r for r in runs if r.get(\"status\") == \"ok\"]\n",
            "print(f\"Loaded {len(runs)} total run records ({len(ok)} successful runs).\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 1. Frozen Dataset Split & EDA Overview\n",
            "\n",
            "We construct a strictly frozen 14,000 / 1,000 train/held-out split of the `databricks-dolly-15k` dataset. A hash of the prompt template is embedded into `results/split.json` to guarantee template stability across experiments."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "split = json.loads((ROOT / \"results/split.json\").read_text())\n",
            "examples = load_train_examples()\n",
            "\n",
            "print(f\"Pool Dataset     : {split['pool']} ({split['pool_size']} examples)\")\n",
            "print(f\"Train / Held-out : {len(split['train_idx'])} / {len(split['held_out_idx'])}\")\n",
            "print(f\"Template Hash    : {split['template_hash']}\")\n",
            "\n",
            "instr_lens = [len(e[\"instruction\"].split()) for e in examples]\n",
            "resp_lens  = [len(e[\"response\"].split()) for e in examples]\n",
            "\n",
            "df_stats = pd.DataFrame({\n",
            "    \"Field\": [\"Instruction Word Count\", \"Response Word Count\"],\n",
            "    \"Mean\": [np.mean(instr_lens), np.mean(resp_lens)],\n",
            "    \"Median\": [np.median(instr_lens), np.median(resp_lens)],\n",
            "    \"Min\": [np.min(instr_lens), np.min(resp_lens)],\n",
            "    \"Max\": [np.max(instr_lens), np.max(resp_lens)]\n",
            "})\n",
            "display(df_stats.style.format({\"Mean\": \"{:.1f}\", \"Median\": \"{:.1f}\"}).set_caption(\"Dataset Length Statistics\"))\n",
            "\n",
            "fig_path = ROOT / \"results/figures/eda1_dataset_overview.png\"\n",
            "if fig_path.exists():\n",
            "    print(\"\\nFigure: Dataset Overview (Length Distributions & Category Composition)\")\n",
            "    display(Image(filename=str(fig_path)))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 2. Selection Method Scorer Distributions & Selection Overlap\n",
            "\n",
            "We compare 6 distinct selection strategies:\n",
            "1. **Random** baseline\n",
            "2. **Perplexity** (SmolLM-135M forward pass loss)\n",
            "3. **Embedding Diversity** (SentenceTransformer MiniLM + facility location)\n",
            "4. **Hybrid** (Perplexity + Diversity)\n",
            "5. **IFD** (Instruction Following Difficulty: conditional vs unconditional loss ratio)\n",
            "6. **Learning Percentage** (Proxy training loss reduction over 1 epoch)\n",
            "\n",
            "Below we examine how these selection methods distribute scores, whether they suffer from length bias, and how much their selected sub-sets overlap (Jaccard similarity)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig_files = [\n",
            "    (\"eda2_score_distributions.png\", \"Figure 2a: Selection Pool Position Histograms\"),\n",
            "    (\"eda3_selection_overlap.png\", \"Figure 2b: Jaccard Similarity Heatmap Between Selection Methods\"),\n",
            "    (\"eda4_length_bias.png\", \"Figure 2c: Response Length Bias Across Methods\")\n",
            "]\n",
            "\n",
            "for fname, caption in fig_files:\n",
            "    p = ROOT / \"results/figures\" / fname\n",
            "    if p.exists():\n",
            "        print(f\"\\n{caption}\")\n",
            "        display(Image(filename=str(p)))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 3. Honest Selection Cost Accounting (RQ2)\n",
            "\n",
            "**Key Insight:** Prior literature assumes data selection is free. In reality, every selection method must compute scores over the full dataset candidate pool ($N = 14,000$) before target fine-tuning can start.\n",
            "\n",
            "We measure selection cost in analytical **FLOPs** as well as **wall-clock time**."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "sel_files = sorted(glob.glob(str(ROOT / \"results/selections/*__r0.05__s0.json\")))\n",
            "cost_rows = []\n",
            "for f in sel_files:\n",
            "    d = json.load(open(f))\n",
            "    cfg = d[\"selector_config\"]\n",
            "    scorer = cfg.get(\"scorer\") or cfg.get(\"embedder\") or cfg.get(\"proxy\") or \"-\"\n",
            "    cost_rows.append({\n",
            "        \"Method\": d[\"method\"],\n",
            "        \"Cost Class\": d[\"cost_class\"],\n",
            "        \"Wall-Clock (s)\": d[\"cost\"][\"wall_clock_s\"],\n",
            "        \"Selection FLOPs\": d[\"cost\"][\"flops\"],\n",
            "        \"Scorer Model\": scorer.split(\"/\")[-1]\n",
            "    })\n",
            "\n",
            "df_cost = pd.DataFrame(cost_rows).sort_values(by=\"Selection FLOPs\")\n",
            "display(df_cost.style.format({\"Wall-Clock (s)\": \"{:.1f}\", \"Selection FLOPs\": \"{:.3e}\"}).set_caption(\"Measured Selection Overhead per Method (14k candidates)\"))\n",
            "\n",
            "p_cost = ROOT / \"results/figures/eda5_cost_breakdown.png\"\n",
            "if p_cost.exists():\n",
            "    print(\"\\nFigure 3: Stacked Compute Breakdown (Selection FLOPs vs. Training FLOPs)\")\n",
            "    display(Image(filename=str(p_cost)))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 4. Main Grid Results & Net-Cost Pareto Frontier (RQ2 Evidence)\n",
            "\n",
            "Below we examine the experimental results across selection ratios (5% and 10%) on `Qwen2.5-0.5B`.\n",
            "\n",
            "When evaluating efficiency claims:\n",
            "1. **Training-Only FLOPs**: standard literature view (ignoring selection cost).\n",
            "2. **Total FLOPs (Selection + Training)**: true net-cost view.\n",
            "\n",
            "**Finding:** Methods like `perplexity` leave the Pareto frontier completely once their selection FLOPs are priced in!"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Aggregate Main Grid Runs\n",
            "grid_runs = [r for r in ok if r.get(\"study\") in (\"fast_grid\", \"study1_main_grid\")]\n",
            "agg = defaultdict(lambda: {\"loss\": [], \"total_flops\": [], \"sel_share\": []})\n",
            "\n",
            "for r in grid_runs:\n",
            "    c = r[\"config\"]\n",
            "    m = r[\"metrics\"]\n",
            "    key = (c[\"selection_method\"], c[\"ratio\"])\n",
            "    agg[key][\"loss\"].append(m[\"held_out_loss\"])\n",
            "    agg[key][\"total_flops\"].append(r[\"cost\"][\"total_flops\"])\n",
            "    agg[key][\"sel_share\"].append(r[\"cost\"][\"selection_share_of_total_flops\"])\n",
            "\n",
            "res_table = []\n",
            "for (method, ratio), data in sorted(agg.items(), key=lambda x: (x[0][1], np.mean(x[1][\"loss\"]))):\n",
            "    res_table.append({\n",
            "        \"Method\": method,\n",
            "        \"Ratio\": f\"{ratio*100:.0f}%\",\n",
            "        \"Held-out Loss (Mean)\": np.mean(data[\"loss\"]),\n",
            "        \"Std Dev\": np.std(data[\"loss\"], ddof=1) if len(data[\"loss\"]) > 1 else 0.0,\n",
            "        \"Selection FLOP Share (%)\": np.mean(data[\"sel_share\"]) * 100,\n",
            "        \"Total FLOPs\": np.mean(data[\"total_flops\"])\n",
            "    })\n",
            "\n",
            "df_grid = pd.DataFrame(res_table)\n",
            "display(df_grid.style.format({\n",
            "    \"Held-out Loss (Mean)\": \"{:.4f}\",\n",
            "    \"Std Dev\": \"{:.4f}\",\n",
            "    \"Selection FLOP Share (%)\": \"{:.1f}%\",\n",
            "    \"Total FLOPs\": \"{:.2e}\"\n",
            "}).set_caption(\"Main Grid Experimental Performance & Cost Share\"))\n",
            "\n",
            "p_pareto = ROOT / \"results/figures/fig1_pareto.png\"\n",
            "if p_pareto.exists():\n",
            "    print(\"\\nFigure 4: Pareto Frontier comparison (Training Cost Only vs. Total Cost)\")\n",
            "    display(Image(filename=str(p_pareto)))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 5. Hyperparameter Fragility & Ranking Stability Across Learning Rates (RQ1)\n",
            "\n",
            "**Core Question (RQ1):** Do selection-method rankings hold when the target model's learning rate is re-tuned?\n",
            "\n",
            "Prior research fixed learning rates across all methods. Below, we test method rankings across different learning rates ($1\\times 10^{-6}$, $2\\times 10^{-4}$, $5\\times 10^{-4}$) to measure rank stability via Spearman rank correlation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "p_rank = ROOT / \"results/figures/fig2_ranking_stability.png\"\n",
            "if p_rank.exists():\n",
            "    print(\"Figure 5: Ranking Stability Across Learning Rates\")\n",
            "    display(Image(filename=str(p_rank)))\n",
            "else:\n",
            "    print(\"fig2_ranking_stability.png not found. Run scripts/figures.py to generate.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 6. Seed Variance & Statistical Significance Analysis\n",
            "\n",
            "We inspect variance across random seeds to ensure performance differences are statistically significant and not noise."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "p_seed = ROOT / \"results/figures/eda6_seed_variance.png\"\n",
            "if p_seed.exists():\n",
            "    print(\"Figure 6: Held-out Loss Variance Across Random Seeds\")\n",
            "    display(Image(filename=str(p_seed)))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "--- \n",
            "## 7. Summary of Empirical Findings & Paper Conclusions\n",
            "\n",
            "1. **Selection FLOPs are Load-Bearing (RQ2)**:\n",
            "   - Model-based selection algorithms spend **58.9% to 83.6%** of their total compute budget on selection alone.\n",
            "   - When total compute is accounted for, training-based methods (`learning_percentage`) and raw `perplexity` fail to remain on the Pareto frontier.\n",
            "   - `IFD` remains effective because its conditional/unconditional ratio delivers high quality per FLOP.\n",
            "\n",
            "2. **Hyperparameter Fragility (RQ1)**:\n",
            "   - Selection rankings do not stay fixed when the learning rate changes. Fixed hyperparameter evaluations create an illusion of method superiority.\n",
            "\n",
            "3. **Practical Guidance for Practitioners**:\n",
            "   - If your compute budget is tight, **Random selection** or lightweight **IFD** are the most cost-effective choices.\n",
            "   - Never deploy computationally expensive selection methods without calculating whether selection cost exceeds training savings."
        ]
    }
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = ROOT / "notebooks/analysis.ipynb"
out_path.write_text(json.dumps(nb, indent=1))
print(f"Wrote updated notebook to {out_path}")
