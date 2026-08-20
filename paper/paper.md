# Does Proxy-Based Data Selection Survive Contact With Reality?

*Robustness and net-cost accounting for efficient instruction tuning.*

<!--
STRUCTURE FROM THE PLAN'S §6. Word budgets are targets, not suggestions: ~6,000
words total, 10-12 pages. Each section below names the exact command that produces
its numbers, so nothing here is written from memory or from a screenshot.

Week 4 rule: experiments stopped at the end of Week 3. If a sentence here needs a
number that does not yet exist in results/runs.jsonl, cut the sentence -- do not
run the experiment.
-->

## 1. Introduction  *(700 words)*

<!-- The two-sentence version of the contribution, which everything else expands:

     Efficiency claims for instruction-tuning data selection rest on two unexamined
     assumptions: that the cost of SELECTING data is negligible, and that method
     rankings measured at one hyperparameter setting hold at others. We test both.

     Land the gap explicitly: arXiv:2402.10430 validates proxy-based selection under
     a single fixed configuration and states that it does not price the selection
     step; arXiv:2512.24503 (ICLR 2026) shows proxy-model rankings are fragile to
     learning rate for data-RECIPE evaluation. Nobody has asked whether
     selection-METHOD rankings survive the same test. -->

**Contributions.**
1. TODO
2. TODO
3. TODO

## 2. Related Work  *(800 words)*

<!-- Six groups. End EVERY group on the sentence "what this line of work holds
     fixed" -- that is what makes the gap visible rather than asserted.

     coreset selection | IFD / Cherry-LLM | DEITA | LESS |
     proxy-model selection (2402.10430) | proxy fragility (2512.24503, 2602.17835) -->

## 3. Methodology  *(1,000 words)*

### 3.1 Selection methods
<!-- Table of the six, grouped by COST CLASS (free / training-free / training-based).
     Keep the cost-class column visible in every table in the paper.

     MUST STATE: our IFD is the training-free variant computed with an off-the-shelf
     pretrained scorer, NOT Cherry-LLM's brief-experience model. The two are not
     interchangeable and the difference is exactly a cost-class difference. -->

### 3.2 The total-cost model
<!-- src/cost.py. Forward = 2ND; LoRA training ~= 4ND; full fine-tune ~= 6ND.
     State the LoRA approximation as an approximation, and repeat it in §7.

     MUST STATE: wall-clock is not comparable across cost classes, because
     learning_percentage runs on a GPU while the training-free methods run on CPU.
     All cross-method cost claims use analytical FLOPs. -->

### 3.3 Budget-aware selection (the proposed rule)
<!-- src/policy.py. This is the paper's constructive contribution -- without it the
     work is a pure audit and the assignment's "design a novel or improved
     methodology" requirement goes unmet.

     Quality model, a data-scaling law with one addition:
         L(m, r) = L_inf + A * (e_m * r)^(-alpha),   e_random := 1
     e_m is the method's EFFECTIVE DATA MULTIPLIER: "this method's 5% is worth
     e_m * 5% of randomly chosen data". One interpretable, falsifiable number per
     method. alpha is shared across methods (a property of the task, and not
     identifiable per-method from two ratios) -- report the residuals so the reader
     can judge whether that assumption held.

     Cost model: C(m, r) = C_sel(m) + c_train * r.

     The rule, closed-form because loss is monotone in r:
         r*(m) = clip((B - C_sel(m)) / c_train, 0, 1)
         choose argmin_m L(m, r*(m))

     Note the r=1 subtlety: at full data every method selects the identical set, so
     a ratio-1.0 point carries no information about any method's multiplier and is
     attributed to the baseline. -->

### 3.4 The robustness protocol
<!-- Spearman/Kendall between method rankings across LR conditions.
     Borrow the ICLR paper's Spearman > 0.95 "stable" threshold so the two results
     are directly comparable. -->

## 4. Experimental Setup  *(600 words)*

<!-- Fill from configs/base.yaml and results/split.json; do not retype from memory.

     Pool: databricks-dolly-15k, 14,000 train / 1,000 held-out, split seed 0,
       template hash cb11391ee245.
     Targets: Qwen2.5-0.5B (main), Qwen2.5-1.5B (scale), Llama-3.2-1B (family).
     Proxies: SmolLM-135M, Qwen2.5-0.5B.
     Adaptation: QLoRA 4-bit, r=16, alpha=32, seq 512, 3 epochs, identical everywhere.
     Ratios: 5%, 10%, with 0% and 100% anchors.
     Seeds: 3 main grid, 2 transfer.
     Hardware: 2x T4 (Kaggle), selection on laptop CPU.
     Statistics: paired bootstrap over held-out examples, 95% CI on every number,
       Holm-Bonferroni across the main grid. -->

## 5. Results  *(1,400 words)*

### 5.1 RQ1 — do rankings survive re-tuning the learning rate?
<!-- python scripts/analyze.py --study study2_lr_sweep
     Figure 2: results/figures/fig2_ranking_stability.pdf

     Report mean and MIN Spearman. If min < 0.95, H1 is supported: part of the
     reported advantage of selection methods is a fixed-hyperparameter artifact.
     If it held, say so equally plainly -- that contradicts the proxy-fragility
     result and is itself the finding. -->

### 5.2 RQ2 — who wins once selection is priced in?
<!-- python scripts/analyze.py --study study1_main_grid
     Figure 1: results/figures/fig1_pareto.pdf

     The two-panel figure IS the argument: same points, x-axis changes from
     training-only to total cost, and the frontier membership changes. Name which
     methods leave the frontier. -->

### 5.3 RQ3 — transfer across scale and family
<!-- Figure 3: results/figures/fig3_transfer.pdf
     Study 4 is EXPLORATORY and underpowered by design. Report effect sizes with
     CIs; make no significance claim. Say the word "underpowered" in the text. -->

### 5.4 RQ4 — the cost-aware rule
<!-- python scripts/analyze.py --study study1_main_grid   (RQ4 section)
     Figure 4: results/figures/fig4_budget_policy.pdf

     Two numbers to report:
       1. the effective data multiplier per method, with the fit's residuals;
       2. the CROSSOVER BUDGET per method -- below it, the method does not repay
          its own selection cost. This is the quotable result: a selection method
          is not better or worse outright, it is better above a stated budget.

     Validation: the rule is fitted on Study 1 (Qwen2.5-0.5B) and PREDICTS for
     Studies 3 and 4. `policy.policy_regret` compares three policies at each budget
     -- ORACLE (hindsight), OURS, and STATIC ("always use the method that won the
     main grid", which is what the literature's fixed-budget framing implies).
     If the rule cannot beat STATIC, say so; that is a valid and publishable
     outcome, and it is what the Limitations section is for. -->

### 5.5 What fell inside the noise
<!-- Do not skip this subsection. Every difference whose CI straddles zero, named.
     analyze.py prints these as INSIDE NOISE; carry the list over verbatim.
     Include the per-method, per-LR divergence (nan) failure rate. -->

## 6. Ablations and Analysis  *(600 words)*

<!-- - Hybrid components: perplexity alone vs diversity alone vs weighted combination.
     - stats.top_k_overlap between method selections: do methods that score alike
       actually pick the same examples?
     - Qualitative: inspect 20 selected examples per method. Cheap, and it is where
       "high perplexity selects noise" becomes visible rather than speculative. -->

## 7. Limitations  *(400 words)*

<!-- State all of these; each is already true and hiding one costs more than it saves:
     one pool (Dolly only); scales <= 1.5B; QLoRA not full fine-tuning; analytical
     rather than measured FLOPs; the scaling law is fitted from only two ratios
     plus an anchor, and its shared exponent is an assumption; single judge model (Qwen2.5-7B), which is a weak
     judge; held-out Dolly loss rewards Dolly-typical selections; wall-clock not
     comparable across cost classes; Study 4 underpowered. -->

## 8. Conclusion  *(300 words)*

## References

<!-- Verified live, 2026-08-20 -->
- Smaller Language Models are capable of selecting Instruction-Tuning Training Data
  for Larger Language Models. arXiv:2402.10430
- Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model
  Practice. arXiv:2512.24503 (ICLR 2026)
- Influence-Preserving Proxies for Gradient-Based Data Selection in LLM
  Fine-tuning. arXiv:2602.17835
- LESS: Selecting Influential Data for Targeted Instruction Tuning. arXiv:2402.04333
- What Makes Good Data for Alignment? (DEITA). OpenReview BTKAeLqLMw
- The Best Instruction-Tuning Data are Those That Fit. OpenReview 4jFSekBaDT
- LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning. arXiv:2505.07437

## Reproducibility

<!-- Fill in at submission:
     git commit SHA, results/split.json template hash, exact commands from the
     README, and the total measured GPU-hours. -->
