import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
paper_path = ROOT / "paper" / "paper.md"
content = paper_path.read_text(encoding="utf-8")

# 1. Replace all § symbols with Section
# Specifically handle cases like §3.2, §5.1, §6.2, §5, §6, etc.
def fix_section_symbol(match):
    num = match.group(1)
    return f"Section {num}"

content = re.sub(r"§\s*(\d+(?:\.\d+)?)", fix_section_symbol, content)
content = content.replace("§", "Section ")

# 2. Insert Figure 5 into Section 4 (Experimental Setup)
fig5_md = """
![Figure 5: Dataset overview for databricks-dolly-15k](../results/figures/eda1_dataset_overview.png)
**Figure 5.** Dataset overview for databricks-dolly-15k showing instruction word count distribution, response word count distribution, and category composition across the candidate pool.
"""

if "![Figure 5" not in content:
    content = content.replace(
        "**Pool and split.** databricks-dolly-15k (15,011 examples)",
        fig5_md + "\n\nAs shown in Figure 5, the Dolly dataset pool contains 15,011 instruction-response pairs with diverse category distributions and a wide spread of response lengths.\n\n**Pool and split.** databricks-dolly-15k (15,011 examples)"
    )

# 3. Insert Figure 1 & Figure 2 into Section 5.1 (RQ2 — what selection actually costs)
fig1_2_md = """
![Figure 1: Pareto frontier comparison](../results/figures/fig1_pareto.png)
**Figure 1.** Pareto frontier of held-out loss against training compute alone (left panel, as accounted in prior literature) versus total compute including selection FLOPs (right panel, true net cost). Scored on total compute, perplexity@5% leaves the Pareto frontier entirely because its selection cost outweighs its downstream benefit.

![Figure 2: Stacked compute breakdown](../results/figures/eda5_cost_breakdown.png)
**Figure 2.** Stacked compute breakdown showing selection FLOPs versus training FLOPs across selection methods. Model-based methods consume 58.9% to 83.6% of total FLOPs during the selection phase before training begins.
"""

if "![Figure 1" not in content:
    content = content.replace(
        "The consequence for the Pareto frontier is direct.",
        fig1_2_md + "\n\nAs illustrated in Figure 2, the selection overhead dominates the overall compute budget for model-based selectors. The consequence for the Pareto frontier is direct, as demonstrated in Figure 1."
    )

# 4. Insert Figure 3 into Section 5.4 (RQ1 — learning-rate robustness)
fig3_md = """
![Figure 3: Selection method rank stability across learning rates](../results/figures/fig2_ranking_stability.png)
**Figure 3.** Selection method rank stability across learning rates ($1\\times 10^{-6}$, $2\\times 10^{-4}$, $5\\times 10^{-4}$). Perplexity flips from 1st place at $1\\times 10^{-6}$ to last place at higher learning rates (Spearman $\\rho = -0.50$), demonstrating hyperparameter fragility.
"""

if "![Figure 3" not in content:
    content = content.replace(
        "The headline number is the Spearman rank correlation",
        fig3_md + "\n\nAs visualized in Figure 3, method orderings are highly sensitive to the target model's learning rate. The headline number is the Spearman rank correlation"
    )

# 5. Insert Figure 4 into Section 5.5 (What fell inside the noise)
fig4_md = """
![Figure 4: Held-out loss variance across random seeds](../results/figures/eda6_seed_variance.png)
**Figure 4.** Held-out loss distributions and seed-to-seed variance across selection methods. Per-condition seed spread reaches 0.028, necessitating paired example-level bootstrap testing to distinguish true signal from noise.
"""

if "![Figure 4" not in content:
    content = content.replace(
        "Seed-to-seed spread within a condition reaches 0.028",
        fig4_md + "\n\nAs shown in Figure 4, seed-to-seed spread within a condition reaches 0.028"
    )

# 6. Insert Figure 6 & Figure 7 into Section 6 & Section 6.1
fig6_md = """
![Figure 6: Candidate pool score distributions](../results/figures/eda2_score_distributions.png)
**Figure 6.** Histograms of selected candidate pool positions for each selection method across the 14,000 candidate pool.
"""

fig7_md = """
![Figure 7: Pairwise Jaccard overlap heatmap](../results/figures/eda3_selection_overlap.png)
**Figure 7.** Pairwise Jaccard similarity heatmap between sets of 700 examples selected at 5% ratio. Most method pairs show near-random overlap (~0.025), except IFD and learning percentage which overlap at 0.306 (12× chance).
"""

if "![Figure 6" not in content:
    content = content.replace(
        "## 6. Ablations and Analysis",
        "## 6. Ablations and Analysis\n\n" + fig6_md + "\nFigure 6 illustrates how each selection method samples from different parts of the score distribution across the 14,000 candidate pool."
    )

if "![Figure 7" not in content:
    content = content.replace(
        "Almost every pair sits at chance.",
        fig7_md + "\n\nAs visualized in the heatmap in Figure 7, almost every pair sits at chance."
    )

# 7. Insert Figure 8 into Section 6.2 (What perplexity actually selected)
fig8_md = """
![Figure 8: Response length bias comparison](../results/figures/eda4_length_bias.png)
**Figure 8.** Response length distribution comparison between perplexity selections, random selections, and the full candidate pool. Perplexity overwhelmingly selects short, one-token responses (median 18 characters vs 187 characters in pool).
"""

if "![Figure 8" not in content:
    content = content.replace(
        "Selecting the examples a small model finds *hardest* overwhelmingly selects",
        fig8_md + "\n\nAs illustrated in Figure 8, selecting the examples a small model finds *hardest* overwhelmingly selects"
    )

# Write updated paper.md
paper_path.write_text(content, encoding="utf-8")
print("Successfully updated paper/paper.md with figure references, captions, and section symbol removals.")
