# 2. Related Work

Instruction-tuning data selection asks which subset of a large instruction pool
should be used to adapt a model, on the premise that most of the pool is
redundant. Work in the area falls into groups that differ in the signal they use,
and each group ends here on the thing it holds fixed — because those fixed
assumptions, taken together, are what this paper tests.

**Coreset and diversity selection.** The oldest line treats selection as subset
selection under a submodular objective: choose examples maximising coverage of a
representation space. Facility-location maximisation over sentence embeddings is
the canonical instance, with a greedy solver carrying the standard `(1 - 1/e)`
guarantee. These methods are attractive because they need no gradient information
and no training, but they optimise a proxy — embedding coverage — with no
guarantee that coverage in an embedding space corresponds to usefulness for the
target model. *Held fixed: that the embedding space used to measure diversity is
the relevant one.*

**Difficulty and quality scoring.** A second line scores each example
individually. Cherry-LLM introduced Instruction-Following Difficulty (IFD), the
ratio of the response's perplexity conditioned on its instruction to its
unconditional perplexity; a ratio below one means the instruction *helped*, so
the example is easy and can be discarded. DEITA combines learned complexity and
quality scores with an embedding-diversity filter. These methods report large
gains from small fractions of data. *Held fixed: a single scoring model, and a
single training configuration for the model being tuned.*

**Influence and gradient-based selection.** LESS selects examples whose low-rank
gradient features align with a target task's gradients, making selection
task-directed rather than generic. It is the most expensive family: it requires
gradient computation over the whole candidate pool, and a warmup training run to
produce the gradient features. Recent work (Iprox) attacks precisely that cost by
compressing the model used to compute gradient features. *Held fixed: that the
cost of computing the selection signal is worth paying, without an explicit
accounting of what else that compute could have bought.*

**Proxy-model selection.** The closest antecedent to this work shows that a small
model can curate data for a much larger one. Zhang et al. (arXiv:2402.10430)
define a *learning percentage* — how much one epoch of training reduces an
example's loss — and show that a 350M model's ranking transfers to targets from
1B to 13B, with a 13B model matching full-data training on 3% of the pool. The
paper states plainly that it does not account for the cost of selection, and the
learning-percentage signal is not cheap: it requires training the proxy for a
full epoch before any target training can begin. *Held fixed: the cost of the
proxy epoch, and the target's hyperparameters.*

**Proxy fragility.** Running against that optimism, Can Small Training Runs
Reliably Guide Data Curation? (arXiv:2512.24503, ICLR 2026) finds that
proxy-model rankings of *data recipes* are fragile to hyperparameters: using
identical training settings across datasets in the name of fairness produces
misleading conclusions, because the optimal configuration is dataset-specific.
Their remedy is to evaluate proxies at deliberately reduced learning rates, under
which small-scale rankings correlate strongly with fully tuned large-scale runs.
*Held fixed: the object being ranked is a data recipe, not a selection method.*

**The gap.** Put the last two groups side by side. Proxy-based *data selection*
is validated under one fixed hyperparameter configuration. Proxy-based *data
recipe evaluation* is now known to be hyperparameter-fragile. Nobody has asked
whether selection-*method* rankings survive re-tuning the target's learning rate,
and nobody prices the selection step into the efficiency claim it supports. This
paper occupies that seam: it tests the robustness assumption, prices the cost
assumption, and proposes a selection rule that makes the resulting trade-off
explicit rather than implicit.

---

# 3. Methodology

## 3.1 Selection methods and cost classes

We compare five selection methods, grouped by the kind of computation each
requires before target training can start. That grouping — not the methods
themselves — is the axis the paper's argument rides on.

| Method | Signal | Cost class |
|---|---|---|
| Random | none | free |
| Embedding diversity | facility location over MiniLM embeddings | training-free |
| Perplexity | response NLL under a 135M scorer | training-free |
| IFD | conditional vs unconditional response NLL | training-free |
| Learning percentage | loss reduction after one proxy epoch | **training-based** |

*Free* methods need no model evaluation at all. *Training-free* methods need
forward passes only. *Training-based* methods need gradient updates before a
single selected example can be identified. The distinction matters because a
fixed cost paid before training is compute unavailable for training.

Two implementation choices must be stated, because both depart from the sources.

**IFD is computed with an off-the-shelf pretrained scorer**, not Cherry-LLM's
"brief experience" model, which is first fine-tuned on a small slice of the pool.
The original formulation is therefore training-based; ours is training-free. The
two are not interchangeable, and the difference is exactly a cost-class
difference — which is what makes it relevant here rather than incidental.

**IFD and perplexity use the same 135M scorer.** A larger scorer for IFD would
make the two arms differ in two ways at once — signal and scorer capacity — and a
win could not be attributed to either without a third arm to disentangle them.

## 3.2 The total-cost model

Every method is charged for the compute it consumes, in two currencies.

*Analytical FLOPs.* A forward pass costs approximately `2ND` FLOPs for `N`
parameters and `D` tokens. LoRA training costs approximately `4ND`: the backward
pass still propagates activation gradients through every frozen layer, but
computes no weight gradients for them. Full fine-tuning costs approximately
`6ND`. These are approximations and are treated as such.

*Wall-clock*, measured with a stopwatch on stated hardware.

Neither is sufficient alone. Wall-clock conflates method cost with hardware;
FLOPs ignore that a CPU you already own and a rented GPU are not the same
resource. We report both, and we are explicit about a constraint this imposes:
**wall-clock is comparable only within a device.** We observed the same
perplexity selection take 11,943 s on a laptop CPU and 493.7 s on a Tesla T4 — a
24× difference — with analytical FLOPs identical to within 0.01%. Every
cross-method cost claim in this paper therefore uses FLOPs.

## 3.3 Budget-aware selection (proposed)

The literature asks *which selection method is best*. That question is ill-posed.
Methods carry different fixed costs, so under a finite budget the answer changes
with the budget. We make this explicit.

**Quality.** We model held-out loss with a data-scaling law in which each method
is summarised by one scalar, its **effective data multiplier** `e_m`:

```
L(m, r) = L_inf + A · (e_m · r)^(-α),     e_random := 1
```

A method selecting fraction `r` behaves like random selection at fraction
`e_m · r`. So `e_ifd = 2.4` means "IFD's 5% is worth 10% of randomly chosen
data" — the claim the efficiency literature makes implicitly, reduced to one
interpretable, falsifiable number. Pinning `e_random = 1` makes the
parameterisation identifiable; otherwise `e` and `A` trade off freely. The
exponent `α` is shared across methods: it is a property of the dataset and task,
and per-method exponents are not identifiable from two ratios. We report the
fit's residuals so the reader can judge whether that assumption held.

One subtlety: at `r = 1` every method selects the identical set — the whole pool
— so a full-data point carries no information about any method's multiplier and
is attributed to the baseline. Failing to do this lets the ceiling run inflate
whichever method happened to supply it.

**Cost.** `C(m, r) = C_sel(m) + c_train · r`, with `C_sel` measured, not assumed.

**The rule.** Loss decreases monotonically in `r`, so the best ratio for a method
is the largest it can still afford after paying its own selection cost:

```
r*(m) = clip((B - C_sel(m)) / c_train, 0, 1)
choose  argmin_m  L(m, r*(m))
```

Closed-form, no search. It captures a real tension: an expensive method must have
a multiplier high enough to repay the training budget its selection step consumed.

From it we derive the **crossover budget** — the budget below which a method does
not repay its own selection cost. This is the paper's most quotable output,
because it replaces "method X is better" with "method X is better above B."

## 3.4 The robustness protocol

For RQ1 we compute Spearman and Kendall correlations between the method orderings
induced by each learning-rate condition. If selection-method rankings were a
stable property of the methods, these correlations would sit near 1.0 regardless
of learning rate. We adopt the Spearman > 0.95 "stable" threshold used by
arXiv:2512.24503 so that our result is directly comparable to theirs.

---

# 4. Experimental Setup

**Pool and split.** databricks-dolly-15k (15,011 examples), split once with seed 0
into 14,000 training candidates and 1,000 held-out examples. The split is frozen
and committed; the prompt template is hashed (`cb11391ee245`) into the split file,
and the pipeline refuses to run if the template changes afterwards, so silent
template drift cannot invalidate results unnoticed.

**Target model and adaptation.** Qwen2.5-0.5B with LoRA (`r=16`, `α=32`,
dropout 0.05) on all attention and MLP projections, sequence length 512, three
epochs, batch size 4 with gradient accumulation 4 (effective batch 16), AdamW,
cosine schedule with 3% warmup, fp16 autocast with trainable parameters kept in
fp32. Every hyperparameter is identical across conditions except the one variable
a study manipulates.

We report **LoRA, not QLoRA**. 4-bit quantization was dropped: at 0.5B it is
unnecessary on a 15 GB GPU, and it is a confound — with quantized weights, a
difference between selection methods is entangled with how each selected subset
interacts with quantization error.

**Selection ratios.** 5% (700 examples) and 10% (1,400), against a full-data
ceiling. Two seeds per condition.

**Measured selection costs** (Tesla T4, 14,000 examples):

| Method | Wall-clock | FLOPs | Class |
|---|---|---|---|
| Random | 0 s | 0 | free |
| Diversity | 58.0 s | 9.38e13 | training-free |
| Perplexity | 493.7 s | 7.12e14 | training-free |
| IFD | 959.3 s | 1.01e15 | training-free |
| Learning percentage | 1,624.7 s | 3.57e15 | training-based |

The training-based method costs 28× the cheapest training-free method in
wall-clock and 38× in FLOPs, measured on identical hardware. Scores are cached by
scorer configuration, so a method pays this cost once rather than once per
(ratio, seed) — the signal is deterministic and does not depend on the seed.

**Evaluation.** Primary: mean response negative log-likelihood on the frozen
1,000-example held-out split, stored **per example** rather than as a mean, so
that comparisons can be paired. Capability retention: ARC-Easy, 300 examples.

We note the limitation this metric carries: held-out Dolly loss rewards selections
that are Dolly-typical, and so is not a neutral arbiter between methods.

**Statistics.** Differences are tested with a paired bootstrap over held-out
examples (10,000 replicates), which removes example-difficulty variance shared by
both arms. Seeds are aggregated separately, as a distinct variance source; pooling
seeds and examples into one bootstrap would produce intervals that are too narrow.
Every reported difference carries a 95% CI, and differences whose CI straddles
zero are named explicitly as inside noise rather than omitted. Across the main
grid we apply Holm-Bonferroni correction.

**Hardware.** Kaggle Tesla T4 (15 GB). Selection and training both run on GPU;
the laptop-CPU figures reported in §3.2 are a separate device-comparison
measurement and are not mixed into the cost tables.
