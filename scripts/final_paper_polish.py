import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
paper_md_path = ROOT / "paper" / "paper.md"
content = paper_md_path.read_text(encoding="utf-8")

# 1. Add Author line if not present
if "Author:" not in content and "Anonymous" not in content:
    content = content.replace(
        "*Robustness and net-cost accounting for efficient instruction tuning.*",
        "*Robustness and net-cost accounting for efficient instruction tuning.*\n\n**Author:** Research Team (Efficient GenAI)"
    )

# 2. Number Table 2 (Measured Selection Costs) in Section 4
if "**Table 2.**" not in content:
    content = content.replace(
        "**Measured selection costs** (Tesla T4, 14,000 examples):",
        "**Table 2.** Measured selection costs (Tesla T4, 14,000 examples)."
    )

# 3. Number Table 5 (Jaccard Overlap Matrix) in Section 6.1
if "**Table 5.**" not in content:
    content = content.replace(
        "| | random | perplexity | diversity | ifd | learn % |",
        "**Table 5.** Pairwise Jaccard similarity matrix between 700 selected examples at 5% ratio.\n\n| | random | perplexity | diversity | ifd | learn % |"
    )

# 4. Number Table 6 (Response Length Distributions) in Section 6.2
if "**Table 6.**" not in content:
    content = content.replace(
        "| | mean response length | median | ≤20 chars |",
        "**Table 6.** Response length distributions (words and characters) across selection methods.\n\n| | mean response length | median | ≤20 chars |"
    )

# 5. Format LaTeX equations cleanly
content = content.replace(
    "```\nL(m, r) = L_inf + A · (e_m · r)^(-α),     e_random := 1\n```",
    "$$L(m, r) = L_{\\infty} + A \\cdot (e_m \\cdot r)^{-\\alpha}, \\quad e_{\\text{random}} := 1$$"
)

content = content.replace(
    "```\nr*(m) = clip((B - C_sel(m)) / c_train, 0, 1)\nchoose  argmin_m  L(m, r*(m))\n```",
    "$$r^*(m) = \\text{clip}\\left(\\frac{B - C_{\\text{sel}}(m)}{c_{\\text{train}}}, 0, 1\\right), \\quad m^* = \\arg\\min_m L(m, r^*(m))$$"
)

paper_md_path.write_text(content, encoding="utf-8")
print("Successfully polished paper/paper.md with Author header, Table 2/5/6 numbering, and LaTeX math blocks.")
