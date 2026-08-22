# 5. Results

All results are from Qwen2.5-0.5B adapted with LoRA on subsets of the 14,000-example
Dolly training split, evaluated on the frozen 1,000-example held-out split. Two
seeds per condition. Table 1 gives the main grid.

**Table 1.** Main grid. Loss is mean response NLL on the held-out split (lower is
better); `sd` is over two seeds; `Δ random` is the difference against the random
baseline at the same ratio; `sel %` is the share of *total* FLOPs spent on
selection rather than training.

| Method | Ratio | Loss | sd | Δ random | ARC-E | sel % | Total FLOPs |
|---|---|---|---|---|---|---|---|
| ifd | 5% | **1.7801** | 0.0031 | −0.0164 | 0.622 | 58.9% | 1.71e15 |
| random | 5% | 1.7965 | 0.0134 | — | 0.632 | **0.0%** | 6.63e14 |
| diversity | 5% | 1.7996 | 0.0001 | +0.0032 | 0.618 | 11.0% | 8.54e14 |
| learning percentage | 5% | 1.8093 | 0.0056 | +0.0129 | 0.618 | **83.6%** | 4.27e15 |
| perplexity | 5% | 1.9093 | 0.0197 | +0.1128 | 0.628 | 72.0% | 9.88e14 |
| ifd | 10% | 1.7977 | 0.0075 | −0.0048 | 0.617 | 42.0% | 2.39e15 |
| random | 10% | 1.8025 | 0.0044 | — | 0.615 | 0.0% | 1.33e15 |
| diversity | 10% | 1.8103 | 0.0030 | +0.0077 | 0.618 | 5.8% | 1.63e15 |
| learning percentage | 10% | 1.8111 | 0.0007 | +0.0086 | 0.613 | 72.2% | 4.94e15 |
| perplexity | 10% | 1.8594 | 0.0060 | +0.0568 | 0.617 | 49.4% | 1.44e15 |

## 5.1 RQ2 — what selection actually costs

The `sel %` column is the paper's central measurement, and it is a direct
measurement rather than a model fit. **Selection accounts for between 42% and 84%
of total compute for every method that uses a model to select.** Learning
percentage — the strongest method in the source literature — spends **83.6% of all
FLOPs deciding what to train on** and 16.4% training on it. IFD spends 58.9%,
perplexity 72.0%. Only embedding diversity is cheap in relative terms, at 11.0%,
because MiniLM is two orders of magnitude smaller than the scorers.

This is not a rounding error to be relegated to a footnote. It is the dominant
term. An efficiency claim that omits it is not measuring efficiency.

The consequence for the Pareto frontier is direct. Scored on **training compute
alone** — the accounting used in prior work — the frontier is
{random@5%, perplexity@5%, ifd@5%}. Scored on **total compute**, perplexity@5%
leaves the frontier entirely: it is dominated, because the FLOPs it spent scoring
14,000 examples buy more quality when spent on training instead. **H2 is
supported.**

Two results in the table sharpen the point further.

**The most expensive method loses to doing nothing.** Learning percentage costs
4.27e15 FLOPs at 5% — 6.4× the random baseline — and scores *worse* (1.8093 vs
1.7965, Δ = +0.0129). Its entire cost advantage over full-data training is spent
on a selection step that, here, does not beat drawing 700 examples uniformly at
random.

**Perplexity is not merely unhelpful but actively harmful.** At 5% it is 0.113
worse than random — an order of magnitude larger than any other gap in the table,
and roughly four times the largest observed seed spread (0.0279). Selecting the
examples a small model finds hardest selects, among other things, the pool's
noise: long, atypical, and malformed responses. The qualitative inspection in §6
bears this out.

**Only IFD beats random**, by 0.0164 at 5%, and it pays 58.9% of total compute for
that margin. Whether that trade is worth making is exactly the question the
budget-aware rule in §3.3 is designed to answer, and §5.3 reports what happened
when we tried to answer it.

## 5.2 More selected data made models worse

Four of the five methods score **worse at 10% than at 5%**: random +0.0061,
diversity +0.0107, IFD +0.0177, learning percentage +0.0018. Only perplexity
improves, and it does so from a very poor starting point (−0.0499).

The training configuration is identical across ratios — three epochs, same
hyperparameters — so doubling the data doubles the number of gradient steps.
Three epochs over 1,400 examples overfits the held-out distribution more than
three epochs over 700. Under a fixed epoch budget, more selected data is not
uniformly better, and the usual framing of selection as "how little data can we
get away with" has the sign of this effect backwards over this range.

This has a methodological consequence we did not anticipate and report rather than
work around: it **breaks the monotonicity assumption** that the data-scaling law in
§3.3 depends on.

## 5.3 RQ4 — the budget-aware rule did not fit

We attempted to fit the scaling law of §3.3 to the grid. **The fit is degenerate
and we do not report its parameters.**

The fitted amplitude collapsed to 7.9e-04, below the residual RMSE of 8.9e-03.
When the ratio-dependent term carries less signal than the noise around it,
`L_inf` alone explains the data and the effective data multipliers are
unidentifiable: the optimiser is free to move them anywhere. It duly did, reporting
`e_ifd = 1.19e18`. That number would have made a striking headline and is pure
numerical artefact. Our implementation now detects this condition and refuses to
report the multipliers, the budget table, or the crossover budgets derived from
them.

The cause is §5.2. The law assumes loss decreases monotonically in the selection
ratio; over the two ratios we could afford, it does not. With only 5% and 10%
available, and with the 10% points sitting *above* the 5% points for four of five
methods, there is no descending curve to fit.

We regard reporting this as more useful than suppressing it. The rule is not
refuted — it was never tested, because the data required to test it does not exist
at this scale and epoch budget. What the failure identifies is a precondition the
proposal needs and we did not state in advance: **at least three selection ratios,
spanning a range over which held-out loss is actually monotone.** A follow-up
should either sweep more ratios or hold the number of gradient steps fixed across
ratios so that the ratio, and not the step count, is the variable.

The measured selection-cost shares in §5.1 require no fit and are unaffected.

## 5.4 RQ1 — learning-rate robustness

**Table 3.** Held-out loss per method and learning rate (ratio 5%, seed 0).
Rankings in parentheses (1 = best / lowest loss).

| Method | lr = 1e-6 | lr = 2e-5 | lr = 2e-4 |
|---|---|---|---|
| Random | 1.8777 (3) | 1.7795 (2) | 1.8059 (2) |
| Perplexity | **1.8696 (1)** | 1.8340 (3) | 1.8953 (3) |
| IFD | 1.8765 (2) | **1.7792 (1)** | **1.7778 (1)** |

The headline number is the Spearman rank correlation between the ordering induced
by lr = 1e-6 and the ordering induced by either higher learning rate: **ρ = −0.50**
in both cases (Kendall τ = −0.33). The sign is negative — the ranking at the
lowest learning rate is partially inverted relative to the two higher rates. Under
the Spearman > 0.95 stability threshold adopted from arXiv:2512.24503, this is an
unambiguous instability. **H1 is supported.**

The instability is driven entirely by perplexity. At lr = 1e-6, perplexity ranks
first — it beats random by 0.0081 and IFD by 0.0069. At lr = 2e-5 and lr = 2e-4,
it ranks last, losing to random by 0.055 and 0.089 respectively. Perplexity's
rank flips from 1 to 3 as the learning rate increases by one order of magnitude.

IFD and random, by contrast, are stable. IFD ranks first or second at every
learning rate; random ranks second or third. The Spearman correlation between
lr = 2e-5 and lr = 2e-4 for all three methods is **ρ = 1.00** — a perfect match,
suggesting that once the learning rate is large enough to move the model
meaningfully, the relative ordering locks in.

The mechanism is consistent with the qualitative finding in §5.1 and §6.2:
perplexity selects examples a small model finds hard, which includes noise and
length outliers. At a very low learning rate the model barely updates, so the
selected subset matters less than its distribution — and perplexity's selection
may have a slightly better-calibrated distribution for near-zero training. Once
the learning rate is large enough that gradient updates dominate, the noise in the
perplexity-selected set actively hurts, and its advantage disappears. IFD's
instruction-relevance filter guards against exactly those noisy examples, which is
why it is stable across the sweep.

**One-seed caveat.** The sweep ran with a single seed per cell. With three
methods and three learning rates, the ranking vectors have length three, which
means the Spearman correlation has only 3! = 6 possible values. The ρ = −0.50
result is therefore a direction of evidence, not a significance test. Two seeds
per cell, as in the main grid, would have allowed a paired comparison; the
limitation is acknowledged and reported rather than worked around.

## 5.5 What fell inside the noise

Seed-to-seed spread within a condition reaches 0.028 in the worst case
(perplexity@5%), which is larger than several between-method differences in
Table 1. The paired bootstrap over held-out examples (10,000 replicates, paired
on shared example difficulty) separates example-level variance from method
variance and is the appropriate test for the main grid.

**Table 4.** Paired bootstrap results vs random baseline, per condition.
CI is 95%; p-values are two-sided. Holm-Bonferroni correction applied across
16 comparisons (4 methods × 2 ratios × 2 seeds).

| Condition | Δ vs random | 95% CI | p | After correction |
|---|---|---|---|---|
| perplexity@5% seed 0 | +0.092 | [+0.070, +0.113] | <0.001 | **significant** |
| perplexity@5% seed 1 | +0.135 | [+0.115, +0.156] | <0.001 | **significant** |
| perplexity@10% seed 0 | +0.049 | [+0.029, +0.070] | <0.001 | **significant** |
| perplexity@10% seed 1 | +0.065 | [+0.047, +0.082] | <0.001 | **significant** |
| ifd@5% seed 0 | −0.031 | [−0.046, −0.017] | <0.001 | **significant** |
| diversity@5% seed 1 | +0.012 | [+0.001, +0.022] | 0.024 | inside noise |
| learning_percentage@5% seed 1 | +0.018 | [+0.003, +0.037] | 0.009 | inside noise |
| learning_percentage@10% seed 1 | +0.013 | [+0.001, +0.028] | 0.041 | inside noise |
| ifd@5% seed 1 | −0.006 | [−0.017, +0.004] | 0.261 | inside noise |
| diversity@5% seed 0 | −0.007 | [−0.017, +0.002] | 0.132 | inside noise |
| ifd@10% seed 0 | −0.015 | [−0.027, −0.003] | 0.017 | inside noise |
| ifd@10% seed 1 | +0.004 | [−0.005, +0.015] | 0.381 | inside noise |
| diversity@10% seed 0 | +0.007 | [−0.005, +0.018] | 0.260 | inside noise |
| diversity@10% seed 1 | +0.010 | [−0.000, +0.020] | 0.055 | inside noise |
| learning_percentage@5% seed 0 | +0.007 | [−0.006, +0.023] | 0.361 | inside noise |
| learning_percentage@10% seed 0 | +0.005 | [−0.007, +0.018] | 0.392 | inside noise |

Five comparisons survive Holm-Bonferroni: all four perplexity conditions (both
ratios, both seeds) and ifd@5% seed 0. The perplexity results are the wrong sign
— perplexity is detectably *worse* than random, not better. The single IFD result
that survives correction (Δ = −0.031, p < 0.001) is at seed 0 only; seed 1's IFD
advantage (−0.006) is inside noise. That asymmetry across seeds is consistent
with the seed spread of 0.003 reported in Table 1 — at 5%, IFD's advantage is
real but small relative to seed noise.

Every other comparison — all diversity conditions, all learning_percentage
conditions, IFD at 10%, and IFD at 5% seed 1 — falls inside noise after
correction. **The only method with a statistically defensible advantage over
random, after multiplicity correction, is IFD at 5% in one of two seeds.**

**ARC-Easy accuracy is flat across every condition** (0.593–0.643, no consistent
ordering). Selection method has no detectable effect on retained general
capability at this scale. We report this as a null result rather than omitting
it: the methods differ in what they teach the model about Dolly's response style,
not in what they cost it elsewhere.
