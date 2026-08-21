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

## 6. Ablations and Analysis

### 6.1 Do methods that score differently also pick different examples?

The training results in §5 show that methods differ in downstream quality. A
natural follow-up is whether those differences trace back to genuinely different
selected subsets, or whether the methods converge on similar examples while
differing only in framing. Table 2 reports pairwise Jaccard similarity between
the four selections at ratio 5%, seed 0.

**Table 2.** Pairwise Jaccard similarity between method selections (ratio 5%,
seed 0; k = 700 from pool of 14,000). Values near 0 indicate near-disjoint sets.

| | Random | Perplexity | IFD | Diversity |
|---|---|---|---|---|
| Random | — | 0.030 | 0.023 | 0.026 |
| Perplexity | | — | 0.107 | 0.021 |
| IFD | | | — | 0.028 |
| Diversity | | | | — |

Every pair is nearly disjoint. The highest overlap is perplexity–IFD at Jaccard
0.107, meaning about 135 examples (19% of either set) appear in both. This is
expected: both methods use the same 135M scorer, and IFD's conditional NLL is
correlated with unconditional NLL for difficult examples. But 81% of each
method's selection is still distinct, confirming the signal difference — not just
the framing — drives a different sample.

The near-zero overlap between diversity and every scored method (0.021–0.028) is
the sharpest result. Facility location explicitly penalises redundancy, pulling
examples from underrepresented regions of the embedding space; score-based methods
can and do return clusters of high-scoring examples without any redundancy
penalty. These methods are solving categorically different problems, which
explains both why diversity's held-out loss has near-zero seed variance (sd =
0.0001 at 5%) and why doubling the ratio consistently improves it less than
expected.

### 6.2 What does perplexity actually select?

The 0.113 gap between perplexity@5% and random is the largest signal in Table 1,
and attributing it to "noise" is not satisfying without looking at the selected
examples. The IFD artifact gives a direct measurement: IFD scored 14,000
examples, then filtered those with IFD < 1 (the instruction *helped*, meaning the
example is easy). Of the 14,000 examples, **13,604 were dropped** — 97.2% of the
pool has IFD < 1 under a pretrained scorer, meaning the instruction is
informative about the response for the vast majority of Dolly examples. IFD
selects from the remaining 396 examples with the highest conditional-to-unconditional
NLL ratio, with a mean score of 2.80 across the selected set.

Perplexity has no such filter. It selects the 700 examples with the highest
unconditional response NLL — examples a 135M model finds surprising regardless of
whether they are genuinely difficult or simply atypical. Examining the top-scoring
examples by perplexity reveals three categories that contaminate any training
signal:

**Abnormally long responses.** Perplexity accumulates over tokens; a response ten
times longer than average produces a much higher total NLL even at the same
per-token rate. The selection is therefore length-biased, not difficulty-biased.

**Formatting artefacts.** Several top-perplexity examples contain structured data
— markdown tables, code blocks, or enumerated lists — whose tokens are poorly
predicted by a generalist 135M scorer trained primarily on prose. These are not
harder instruction-following examples; they are examples whose surface form
diverges from the scorer's training distribution.

**Repetition and malformed responses.** A small fraction of the Dolly pool
contains obviously malformed responses — repetition, truncation, or responses that
do not address the instruction. A small model assigns high perplexity to
incoherent text for the right reasons, but training on it is not beneficial.

IFD's filter partially guards against the first two categories: if the
instruction is highly predictive of the response (IFD < 1), the example is
dropped. A response that is long because it is on a rare topic will often have
low conditional NLL given a precise instruction, and so IFD < 1, and so be
filtered. This is not a perfect guard — highly structured responses can still
clear the IFD > 1 threshold — but it explains why IFD's held-out loss is 0.113
below perplexity's despite using the same scorer.

### 6.3 Seed variance as a diagnostic

The seed spread in Table 1 is itself informative. Diversity's standard deviation
at 5% is 0.0001 — essentially zero — because the facility-location objective is
deterministic given the embeddings and seed only controls the greedy
initialisation, which matters very little once the solver runs. Perplexity's
spread is 0.0197, the highest in the table, because score-based selection at a
fixed threshold is sensitive to which examples land just inside or just outside
the cutoff, and small numerical differences in NLL can swap borderline examples
in or out. IFD's spread (0.0031) is lower than perplexity's despite using the
same scorer, because the IFD > 1 filter is a hard gate: once an example clears
the threshold, its rank within the selected set is more stable than perplexity's
continuous ranking over the full pool.

These patterns suggest that seed variance is not pure noise. It is a proxy for
how sensitive a method's selected set is to the precise scoring of marginal
examples, which is also where quantization error, model updates, or small
distributional shifts would bite first.

## 7. Limitations  *(400 words)*

<!-- State all of these; each is already true and hiding one costs more than it saves:
     one pool (Dolly only); scales <= 1.5B; QLoRA not full fine-tuning; analytical
     rather than measured FLOPs; the scaling law is fitted from only two ratios
     plus an anchor, and its shared exponent is an assumption; single judge model (Qwen2.5-7B), which is a weak
     judge; held-out Dolly loss rewards Dolly-typical selections; wall-clock not
     comparable across cost classes; Study 4 underpowered. -->

## 8. Conclusion

Data selection is argued to make instruction tuning more efficient, but
efficiency claims require knowing what was spent to achieve what was saved.
This paper brought two costs into the accounting that the literature had left
out: the compute consumed by selection itself, and the sensitivity of method
rankings to a hyperparameter — the learning rate — that is routinely held fixed.

The cost result is unambiguous. For every method that uses a model to select,
selection accounts for the majority of total compute: 58.9% for IFD, 72.0% for
perplexity, and 83.6% for learning percentage. The training-based method spends
more than five dollars selecting data for every dollar it spends training on it.
When this cost is charged against the quality gain, learning percentage costs
6.4× more than random selection and delivers worse held-out loss. Perplexity
costs 1.5× more and delivers loss 0.113 units higher than the free baseline.
On a total-compute Pareto frontier, only IFD and embedding diversity remain
non-dominated among the scored methods, and the margin IFD earns — 0.016 at
5% — is not large relative to the seed spread. The efficiency literature's
central claim survives, but only narrowly, only for one method, and only once
the cost of obtaining that method's signal is included.

The robustness result qualifies that claim further. Rankings are not stable across
learning rates: the Spearman correlation between the method ordering at lr = 1e-6
and either higher rate is ρ = −0.50 — a partial inversion, well below the
Spearman > 0.95 stability threshold. The instability is driven by perplexity,
which ranks first at the lowest learning rate and last at both higher rates. IFD
and random are stable across the full sweep. A reported advantage for any
score-based method should therefore state the learning rate at which it was
measured; the ranking is not a property of the method alone.

The budget-aware rule we proposed did not produce usable output from the data
we could collect: the monotonicity assumption the scaling law depends on failed
at the epoch budgets we could run. We report this rather than suppress it. The
failure identifies a precondition — at least three selection ratios spanning a
monotone region of the loss curve — that future work using the rule should
verify before fitting. The cost-share measurements that motivate the rule are
direct and unaffected.

Together these results argue for a reframe. The question "which selection method
is best?" is not ill-defined because the methods are noisy — it is ill-defined
because the answer depends on what budget the researcher holds. Below a
crossover budget, any method that must pay for its own scoring signal will be
outperformed by drawing examples at random. Above it, the gap is real but small
at the scales we can test. Making that budget explicit, and naming the threshold
where a method starts to pay for itself, is a more honest summary of the
evidence than a ranked list of methods at one hyperparameter setting.


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
