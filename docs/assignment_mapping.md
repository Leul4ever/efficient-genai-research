# What this research answers, from the assignment

The assignment says "investigate **one or more** research questions" and lists
areas "including, but not limited to". Full coverage is neither expected nor
desirable — the plan's own rule is that depth on one pool beats shallowness on
three. This document records exactly which items the work answers, so §1 can name
them and so nothing gets claimed that the experiments do not support.

The distinction that matters throughout: **studying** something is not the same as
**using** it. This work uses QLoRA; it does not research quantization. Claiming
otherwise is padding, and a marker will spot it.

## Research questions

| # | Assignment question | Coverage | Where |
|---|---|---|---|
| 6 | *What methodologies provide the greatest improvement in performance per unit of computation?* | **Primary** | RQ2 + RQ4. The whole paper. |
| 1 | *How can GenAI models learn effectively from significantly smaller datasets?* | **Direct** | 5% and 10% ratios against a full-data ceiling |
| 2 | *How can computational requirements be reduced without substantially sacrificing model quality?* | **Direct** | RQ2 — net-cost accounting including the selection step |
| 3 | *Which data contributes most to model learning, and how can it be identified automatically?* | **Direct** | Six selection methods across three cost classes |
| 5 | *How can existing knowledge be transferred more effectively across models, domains, or tasks?* | **Direct** | RQ3 — proxy→target transfer across scale and model family |
| 4 | *Can alternative optimization or training strategies improve convergence speed and sample efficiency?* | **Partial** | RQ1 tests sensitivity to the learning rate; it does **not** propose a new optimizer |
| 7 | *Can entirely new training paradigms outperform conventional large-scale training under constrained resources?* | **Not addressed** | No new paradigm is proposed. Do not claim it. |

**Question 6 is the anchor.** It is the one question the work answers more
completely than the literature it builds on, because prior work measures
performance per unit of *training* compute while leaving selection compute
unpriced. Lead the Introduction with it.

## Research areas

### Answered

| Area | How |
|---|---|
| **Data-centric AI** | The entire premise: improve the data, not the model or its scale |
| **Intelligent dataset selection** | Six methods — random, perplexity, embedding diversity, hybrid, IFD, learning-percentage |
| **Transfer learning** | RQ3 tests whether a proxy's difficulty signal survives a scale gap and a change of model family |
| **Parameter-efficient adaptation** | QLoRA is the adaptation mechanism in all 85 runs, held identical across conditions |
| **Any original methodology that improves the efficiency of GenAI systems** | The budget-aware selection rule (`src/policy.py`): effective data multipliers plus per-method crossover budgets |

That last row is the one the assignment's "novel or improved methodology"
requirement is graded against. Without it the work is an audit, which the earlier
plan did not satisfy.

### Adjacent — mention only with the qualifier attached

| Area | The honest position |
|---|---|
| **Active learning** | Shares the "which examples are worth training on" question, but selection here is **one-shot and static**. There is no acquisition loop and no model-in-the-loop iteration. |
| **Curriculum learning** | We choose *which* examples, never *in what order*. Ordering the selected subset is a natural extension and is explicitly out of scope. |
| **Knowledge distillation** | A small model transferring a signal to a larger one is a cousin of distillation, but no logits, soft targets, or student-teacher objective are involved. |
| **Representation learning** | Used instrumentally — MiniLM embeddings drive the facility-location objective. Not studied. |
| **Quantization-aware training** | 4-bit QLoRA is the tool, held constant across every condition precisely so it is *not* a variable. Not a research contribution here. |

### Not addressed

Sparse computation · dynamic architectures · model compression · self-supervised
learning · semi-supervised learning · synthetic data generation ·
retrieval-augmented learning · reinforcement learning for optimization · hybrid
training methodologies · efficient inference strategies · novel optimization
algorithms.

Note that "hybrid" in this project means hybrid *selection* (perplexity-weighted
diversity), not hybrid *training* — do not let the shared word imply coverage.

## For the Introduction

One paragraph, roughly:

> This work addresses the assignment's question of what methodologies deliver the
> greatest improvement in performance per unit of computation, and does so by
> challenging how that quantity is currently measured. Existing instruction-tuning
> data-selection results report quality against *training* compute while treating
> the cost of selection itself as free, and validate method rankings at a single
> fixed hyperparameter configuration. We test both assumptions, and propose a
> budget-aware selection rule that makes the resulting trade-off explicit.

Then state plainly which questions are **not** addressed. Naming the boundary
costs nothing and reads as control over the scope rather than a gap in it.
