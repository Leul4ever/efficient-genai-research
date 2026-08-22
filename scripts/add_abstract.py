from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
paper_md_path = ROOT / "paper" / "paper.md"
content = paper_md_path.read_text(encoding="utf-8")

abstract_text = """
## Abstract

Data selection methods for instruction tuning report matching or exceeding full-corpus performance using only a small fraction of candidate data. However, existing efficiency claims rely on two untested assumptions: (1) that the computational overhead of selecting data is negligible, and (2) that selection method rankings remain stable across target model hyperparameters. In this paper, we conduct a rigorous measurement and robustness audit of proxy-based data selection. First, we price selection compute into the net efficiency ledger using analytical FLOPs and wall-clock accounting. We show that model-based selection algorithms consume between **42.0% and 83.6% of total compute** during selection alone, causing methods like perplexity to fall off the total-compute Pareto frontier. Second, we test selection ranking stability across learning rates ($1\\times 10^{-6}$ to $2\\times 10^{-4}$); we observe significant rank instability (Spearman $\\rho = -0.50$), demonstrating that fixed-hyperparameter benchmarks produce fragile method orderings. Finally, an empirical audit reveals how un-audited selection pipelines can fail silently due to FP16 numerical overflow or inverted filter logic, defaulting to trivial low-index corpus fallbacks while producing plausible metrics. We provide an open-source, reproducible suite and advocate for budget-aware, audited accounting in data selection research.
"""

if "## Abstract" not in content:
    content = content.replace(
        "## 1. Introduction",
        abstract_text.strip() + "\n\n---\n\n## 1. Introduction"
    )
    paper_md_path.write_text(content, encoding="utf-8")
    print("Successfully added Abstract section to paper/paper.md")
else:
    print("Abstract already present in paper/paper.md")
