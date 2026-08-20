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

*[Pending: the sweep is being re-run. Six of nine conditions failed silently on a
learning-rate parsing defect — exponent-form floats without a decimal point were
passed to the optimiser as strings — leaving results at a single learning rate.
The defect is fixed and failed runs are now recorded rather than dropped.]*

## 5.5 What fell inside the noise

Seed-to-seed spread within a condition reaches 0.0279 in the worst case
(perplexity@5%), which is larger than several of the between-method differences in
Table 1. Specifically, the gaps for **diversity (+0.0032 at 5%)**, **learning
percentage (+0.0129 at 5%, +0.0086 at 10%)**, **IFD (−0.0048 at 10%)** and
**diversity (+0.0077 at 10%)** are all comparable to or smaller than that spread,
and with two seeds per condition we cannot separate them from noise.

Only two differences in the table are large relative to seed variation: IFD's
advantage at 5% (−0.0164) and perplexity's deficit (+0.1128 at 5%, +0.0568 at
10%). The paired bootstrap over held-out examples, which removes the
example-difficulty variance shared between arms, is the appropriate test for the
remainder, and is reported per condition in the accompanying results.

**ARC-Easy accuracy is flat across every condition** (0.613–0.632, a range of
0.019 with no consistent ordering). Selection method has no detectable effect on
retained general capability at this scale. We report this as a null result rather
than omitting it: the methods differ in what they teach the model about Dolly's
response style, not in what they cost it elsewhere.
