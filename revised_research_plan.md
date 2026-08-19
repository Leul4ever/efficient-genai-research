# Revised Research Plan (Quality-First)
## Does Proxy-Based Data Selection Survive Contact With Reality? Robustness and Net-Cost Accounting for Efficient Instruction Tuning

**Planning horizon:** the assignment's full four weeks. Scope is limited on the *breadth* axis (one pool, two model scales) so it can be deep on the *rigor* axis (seeds, significance tests, robustness sweeps). Compute: free Kaggle, 2×T4, ~120 GPU-hours over four weeks.

---

## 1. Why the original framing had to change

I ran a literature check before committing you to a topic. Two papers reshape the plan:

**[Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models](https://arxiv.org/abs/2402.10430)** already establishes the headline idea I first proposed. A 350M selector curates data for models up to 13B, on Alpaca and Dolly, with hardness transferring across scales (Kendall-τ 0.52–0.75) and a 13B model beating the full-data baseline using 3% of the data. **"Small proxy picks data for big model" is not a novel claim.** Proposing it would have been the single worst outcome of this project.

But it leaves two things open, and states one of them itself: it **does not account for the cost of selection**. Its "learning percentage" signal requires actually *training* the proxy for an epoch before target training begins. That cost never enters the efficiency ledger.

**[Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice](https://arxiv.org/html/2512.24503)** finds that proxy-model rankings are **fragile to hyperparameters** — minor learning-rate changes reorder data-recipe rankings, and small-scale results correlate poorly with large-model outcomes once hyperparameters are tuned per-dataset. Rankings only stabilise (Spearman > 0.95) under very small learning rates (~1e-6).

Put those together and a real gap appears between them:

> Proxy-based **data selection** is validated under a single fixed hyperparameter configuration, while proxy-based **data-recipe evaluation** is now known to be hyperparameter-fragile. Nobody has tested whether selection-method rankings survive re-tuning the target's learning rate — and nobody prices the selection step into the efficiency claim.

That is your contribution. It sits precisely in the seam between two recent papers, neither of which closes it.

---

## 2. Why this topic is *well-matched to weak compute*

This is a **measurement and robustness paper**, not a SOTA-chasing paper. You are not trying to beat anyone. You are asking whether existing claims hold up when you vary one axis they held fixed, and whether they hold up economically. Papers of this shape:

- do not require large models to be valid;
- produce publishable findings whether the answer is yes or no;
- are graded on experimental design, which is exactly where careful work with 3 seeds and significance tests beats a bigger GPU.

This is the highest-quality result you can get from a laptop with no discrete GPU. It converts your constraint into a methodological virtue.

---

## 3. Research questions and hypotheses

**RQ1 (Robustness).** Do data-selection method rankings persist when the target model's learning rate is re-tuned?
*H1:* Rankings partially collapse. Methods that beat random at one LR will not reliably do so across an LR sweep, and part of the reported advantage of selection methods is an artifact of fixed-hyperparameter evaluation.

**RQ2 (Net cost).** When selection cost is added to training cost, which methods actually improve quality-per-total-FLOP?
*H2:* Training-free signals (perplexity, embedding diversity) dominate the Pareto frontier at small pool sizes, because training-based signals (learning percentage, IFD) spend more compute on selection than they save on training.

**RQ3 (Transfer).** Does the proxy→target selection signal degrade with the size gap, and does it survive a change of model family?
*H3:* Transfer degrades gracefully within a family but weakens across families, because part of the difficulty signal is tokenizer- and pretraining-specific rather than intrinsic to the example.

All three are falsifiable, and **every possible outcome is reportable.** There is no result here that leaves you with nothing to write.

---

## 4. Frozen experimental design

| Component | Choice |
|---|---|
| Pool | databricks-dolly-15k → 14,000 train / 1,000 held-out, fixed split |
| Targets | Qwen2.5-0.5B (main grid), Qwen2.5-1.5B (scale transfer), Llama-3.2-1B (cross-family) |
| Proxy scorers | SmolLM-135M, Qwen2.5-0.5B |
| Adaptation | QLoRA 4-bit, r=16, α=32, seq 512, 3 epochs — identical across all conditions |
| Ratios | 5%, 10% (with 0% and 100% anchors) |
| Seeds | 3 (main grid), 2 (transfer studies) |

### Selection methods compared

| Method | Signal | Cost class |
|---|---|---|
| Random | — | free |
| Perplexity (135M) | forward only | **training-free** |
| Embedding diversity | MiniLM + facility location | **training-free** |
| Perplexity + diversity | hybrid | **training-free** |
| IFD (Cherry-LLM style) | forward, conditional vs unconditional loss | **training-free** |
| Learning percentage | requires 1 epoch of proxy training | **training-based** |
| Full data | — | ceiling |

The training-free / training-based split is the axis your cost argument rides on. Keep it visible in every table.

### Four studies

1. **Main grid** — 6 methods × 2 ratios × 3 seeds on Qwen-0.5B, plus full-data ceiling. Establishes the standard result and its error bars.
2. **LR-robustness sweep (RQ1, the novel core)** — 4 methods × 3 learning rates × 2 seeds at 5%. Does the ranking hold?
3. **Scale transfer (RQ3)** — 4 methods × 2 ratios × 2 seeds on Qwen-1.5B.
4. **Cross-family replication (RQ3)** — 3 methods at 5% × 2 seeds on Llama-3.2-1B.

### Verified compute budget

| | |
|---|---|
| Training runs | **87** |
| Training time | 38.5 h |
| Evaluation overhead | 13.0 h |
| **Total** | **51.5 h of 120 h Kaggle quota** |
| Headroom | **133%** (68 h spare) |

The headroom is deliberate. It absorbs failed runs, a re-run after a bug, and the reruns you will want after seeing Week 3 results.

Selection costs (your laptop CPU, 14k examples): perplexity ≈ 2 min, embeddings ≈ 2 min, IFD with 0.5B ≈ 8 min, learning-percentage ≈ 17 min. **Measure all of these with a stopwatch** — they are primary results, not footnotes.

### Evaluation (freeze in Week 1, never change)

- **Primary:** held-out loss on the fixed 1,000-example split.
- **Instruction following:** IFEval, prompt-level strict.
- **Capability retention:** ARC-Easy, HellaSwag.
- **Generation quality:** AlpacaEval-style pairwise on 200 held-out prompts, judged by a local Qwen2.5-7B-Instruct on Kaggle. Include position-swap debiasing.
- **Statistics:** paired bootstrap over held-out examples, 95% CIs on every number. Report which differences fall inside noise — explicitly.

That last line is worth more marks than any accuracy gain. Most student papers report point estimates from one seed; reporting CIs and naming your null results reads as competence.

---

## 5. Four-week schedule

**Week 1 — Infrastructure and baselines.**
Kaggle pipeline end-to-end. Frozen split, frozen eval, config-driven `train.py`, results appended to JSONL. Run random vs full-data on Qwen-0.5B with 3 seeds. Deliverable: *a working pipeline and a plot with error bars.* In parallel, read and take structured notes on the 8 core papers.
**Gate:** if random-5% already matches full-data within CI, say so loudly — it reframes the whole paper and is itself a finding.

**Week 2 — Methods and the main grid.**
Implement all six selection methods, on CPU where possible. Time each. Run Study 1 in full. Write §3 Methodology and §4 Setup — these do not depend on results.

**Week 3 — The novel studies.**
Study 2 (LR robustness) first; it carries the paper. Then Studies 3 and 4. Build the rank-correlation analysis: Spearman/Kendall between method rankings across LRs and across model scales. Begin Results.

**Week 4 — Analysis and writing.**
Pareto plots, significance tests, ablations, error analysis on which *examples* each method picks (a qualitative section — inspect 20 examples per method — adds real depth cheaply). Full draft by Day 24, then two revision passes. Repo cleaned, seeds fixed, README with exact repro commands.

**Rule:** experiments stop at end of Week 3. Week 4 is writing. A brilliant experiment described badly scores below a modest experiment described well.

---

## 6. Paper outline (~6,000 words, 10–12 pages)

1. **Introduction** (700) — efficiency claims rest on unpriced selection and untested hyperparameter assumptions.
2. **Related Work** (800) — coreset selection; IFD/Cherry; DEITA; LESS; proxy-model selection; the proxy-fragility literature. Each group ends on what it holds fixed.
3. **Methodology** (1,000) — selection methods, the total-cost model, the robustness protocol.
4. **Experimental Setup** (600) — models, data, splits, metrics, hardware, statistics.
5. **Results** (1,400) — RQ1, RQ2, RQ3 in turn. Figure 1 = total-FLOPs vs quality Pareto. Figure 2 = ranking stability across LRs. Figure 3 = transfer vs size gap.
6. **Ablations and Analysis** (600) — components; qualitative look at selected examples.
7. **Limitations** (400) — one pool, ≤1.5B scales, QLoRA not full fine-tuning, analytical FLOPs, single judge model.
8. **Conclusion** (300).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| All methods land inside each other's CIs | That **is** RQ1's answer. Report it as the finding; it's consistent with the proxy-fragility literature. |
| Kaggle quota or session limits bite | 68 h headroom; drop Study 4 first, then Study 3. Studies 1+2 alone are a complete paper. |
| The 7B judge won't fit or is too slow | Fall back to held-out loss + IFEval as primary. Judge eval is an enhancement, not a dependency. |
| Scope creep into new datasets/models | Forbidden after Week 2. Depth on one pool beats shallowness on three. |
| Someone published this exact study | Re-run the literature check at the start of Week 3; if scooped, pivot to the cross-family axis, which is thinner in the literature. |

---

## Sources

- [Smaller Language Models are capable of selecting Instruction-Tuning Training Data for Larger Language Models](https://arxiv.org/abs/2402.10430)
- [Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model Practice](https://arxiv.org/html/2512.24503)
- [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/pdf/2402.04333)
- [What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning](https://openreview.net/forum?id=BTKAeLqLMw)
- [The Best Instruction-Tuning Data are Those That Fit](https://openreview.net/forum?id=4jFSekBaDT)
- [Large-Scale Data Selection for Instruction Tuning](https://www.researchgate.net/publication/389580547_Large-Scale_Data_Selection_for_Instruction_Tuning)
- [LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning](https://arxiv.org/pdf/2505.07437)
- [Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-tuning](https://arxiv.org/html/2602.17835v1)
