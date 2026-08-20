# 1. Introduction

Recent progress in generative AI has come overwhelmingly from scale: more
parameters, more data, more compute. The results are real, but the recipe is
expensive, hard to reproduce, and closed to most researchers. A large body of
work therefore asks whether the same quality can be reached with less — and one
of the most active answers is *data selection*: if most of an instruction-tuning
corpus is redundant, then identifying the useful fraction should buy most of the
benefit at a fraction of the cost.

The reported results are striking. Methods based on instruction difficulty,
embedding diversity, gradient influence, and small proxy models all report
matching or beating full-data training using a few percent of the pool. One
result is directly upstream of this work: Zhang et al. (arXiv:2402.10430) show
that a 350M model can curate instruction data for models up to 13B, with a 13B
target beating full-data training on 3% of the pool.

This paper does not propose a better way to select data. It asks whether the
existing evidence that selection helps survives two things it was never tested
against.

**First, selection is not free, but it is priced as though it were.** Every
selection method must compute something over the whole candidate pool before a
single target-training step can run — forward passes at minimum, and for the
strongest methods a full training epoch of a proxy model. That compute is
reported, when it is reported at all, as a footnote rather than as part of the
efficiency claim. Zhang et al. state explicitly that their accounting excludes
it. But under a finite budget, compute spent selecting data is compute not spent
training on it, and a method that spends more on selection than it saves on
training has not made anything more efficient. The comparison that matters is
quality per unit of *total* compute, and it is not the comparison the literature
reports.

**Second, method rankings are measured at one hyperparameter setting.** Every
selection method in the literature is validated by training a target model under
a single fixed configuration and comparing final quality. Running against this,
recent work on proxy models (arXiv:2512.24503, ICLR 2026) finds that rankings of
*data recipes* produced by small proxy runs are fragile to hyperparameters: using
identical settings across datasets in the name of fairness is itself a source of
error, because the optimal configuration is dataset-specific, and rankings only
stabilise under deliberately reduced learning rates.

Those two results have not been put together. Proxy-based *data selection* is
validated under one fixed configuration; proxy-based *data recipe evaluation* is
known to be hyperparameter-fragile. Nobody has asked whether selection-*method*
rankings survive re-tuning the target's learning rate, and nobody prices the
selection step into the claim it supports. That seam is what this paper occupies.

**Contributions.**

1. **A robustness test of selection-method rankings (RQ1).** We re-tune the
   target's learning rate across three orders of magnitude and measure whether
   the ranking of selection methods persists, using the rank-correlation
   threshold from the proxy-fragility literature so the results are directly
   comparable.

2. **A net-cost accounting of selection (RQ2).** We measure the compute each
   method spends on selection, in wall-clock and analytical FLOPs, and recompute
   the quality-per-compute frontier with that cost included. Methods are grouped
   by cost class — free, training-free, training-based — because that grouping,
   not the individual method, is what the accounting turns on.

3. **Budget-aware selection, a decision rule (RQ4).** We summarise each method by
   a single interpretable scalar, its *effective data multiplier*, fit a scaling
   law over selection ratio, and derive in closed form which (method, ratio) pair
   maximises quality under a stated compute budget. From it we derive each
   method's **crossover budget**: the budget below which it does not repay its
   own selection cost. This replaces "method X is better" with the answerable
   claim "method X is better above budget B."

We are explicit about what this paper does not do. It does not propose a new
selection signal, does not chase state of the art, and does not run at scales
where selection is most economically consequential. Its claim is narrower and,
we argue, more useful: that two assumptions underpinning a widely reported
efficiency result are testable, have not been tested, and change the conclusion
when they are.

---

# 7. Limitations

The limitations below are ordered by how much they constrain the conclusions, and
none is incidental.

**A single pool and a single target scale.** All results come from
databricks-dolly-15k adapted onto Qwen2.5-0.5B. Dolly is small, human-written,
and stylistically narrow; larger and noisier pools are exactly where selection
should matter most, and where redundancy is highest. Nothing here establishes
that the observed behaviour holds at 7B or beyond, and the proxy-selection
literature we build on reports its strongest effects at 13B. The cost argument in
particular changes shape with scale: selection cost is roughly fixed in the pool
size while training cost grows with the target, so a method that does not repay
itself at 0.5B may repay itself comfortably at 13B. Our crossover budgets should
be read as a method for asking the question, not as universal constants.

**LoRA, not full fine-tuning, and not QLoRA.** Adaptation is low-rank throughout.
Full fine-tuning has a different sample-efficiency profile, and a selection method
that helps a rank-16 adapter need not help a full update. We dropped 4-bit
quantization deliberately — at 0.5B it is unnecessary, and it is a confound,
because quantization error interacts with the selected subset — but this does mean
our results do not speak to the QLoRA setting the original papers used.

**Statistical power in the learning-rate sweep.** The sweep runs one seed per
cell. With `n=1` the ranking at any single learning rate could move through seed
noise alone, so RQ1's evidence is suggestive rather than conclusive. The paired
bootstrap over held-out examples carries what significance we can claim; the
seed-level variance is not estimated in that study. A stronger test needs at
least three seeds per cell and was not affordable here.

**Held-out loss is not a neutral arbiter.** Our primary metric is response
negative log-likelihood on held-out Dolly examples. This rewards selections that
are Dolly-typical, and so slightly favours methods whose selection criterion
correlates with the pool's dominant style. Capability retention is measured only
by ARC-Easy on 300 examples. IFEval, HellaSwag, and pairwise judgement by a
larger model were all planned and cut for time; their absence means we measure
what the model *predicts* far better than what it *does*.

**Analytical rather than measured FLOPs.** Costs use the standard `2ND` / `4ND` /
`6ND` approximations. The LoRA multiplier in particular is an estimate: the
backward pass propagates activation gradients through frozen layers but computes
no weight gradients for them, and the true constant depends on the
implementation. Wall-clock is reported alongside and is comparable only within a
device — we measured the same selection at 11,943 s on CPU and 493.7 s on a T4.

**The scaling law's shared exponent is an assumption.** Fitting a per-method
exponent is not identifiable from two selection ratios, so the exponent is shared
across methods and only the multiplier varies. We report the fit's residuals; if
they are the same size as the spread of the multipliers, the multipliers should
not be trusted.

**IFD as implemented is not Cherry-LLM's IFD.** We use an off-the-shelf
pretrained scorer rather than a briefly fine-tuned "experience" model, which moves
the method from training-based to training-free. That is the correct variant for
the question we are asking, but it means our IFD numbers are not directly
comparable to the original paper's.
