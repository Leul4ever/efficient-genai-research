# 6. Ablations and Analysis

## 6.1 Do these methods select the same data?

Selection methods are usually compared on downstream quality alone, which leaves
open a prior question: do they even disagree? We measure Jaccard overlap between
the 700 examples each method selects at 5%. Two independent draws of 700 from
14,000 overlap at 0.0256 by chance.

| | random | perplexity | diversity | ifd | learn % |
|---|---|---|---|---|---|
| **random** | — | 0.030 | 0.026 | 0.023 | 0.027 |
| **perplexity** | 0.030 | — | 0.021 | **0.107** | 0.020 |
| **diversity** | 0.026 | 0.021 | — | 0.028 | 0.027 |
| **ifd** | 0.023 | **0.107** | 0.028 | — | **0.306** |
| **learning %** | 0.027 | 0.020 | 0.027 | **0.306** | — |

Almost every pair sits at chance. Methods that are all described as identifying
"difficult" or "informative" examples are, with two exceptions, selecting
essentially disjoint subsets of the pool. Whatever they measure, it is not a shared
underlying quantity.

The two exceptions are informative. **IFD and learning percentage overlap at 0.306
— twelve times chance** — despite one being training-free and the other requiring a
full proxy epoch. If a training-free signal recovers a third of what an
epoch-of-training signal finds, the case for paying 83.6% of total compute for the
latter weakens considerably. **Perplexity and IFD overlap at 0.107**, four times
chance, which is unsurprising: they share a scorer and IFD's numerator is the
conditional perplexity.

## 6.2 What perplexity actually selected

Perplexity was the worst method by a wide margin (+0.113 against random, §5.1).
Inspecting its selections explains why, and the explanation is not subtle.

Selecting the examples a small model finds *hardest* overwhelmingly selects
**examples with very short responses**:

| | mean response length | median | ≤20 chars |
|---|---|---|---|
| whole pool | 359 | 187 | **6%** |
| perplexity | **51** | **18** | **55%** |
| ifd | 611 | 124 | 23% |
| random | 351 | 183 | 6% |
| diversity | 368 | 171 | 6% |
| learning % | 401 | 204 | 6% |

Perplexity's median selected response is **18 characters**. A sample of what that
means in practice:

| instruction | selected response |
|---|---|
| In what key do most car horns honk? | `F` |
| How many books are there in the Harry Potter series? | `7` |
| How many member states does the European Union have? | `27` |
| how many limbs are in yoga | `8` |

The mechanism is now clear. A short response offers few tokens over which to
amortise the model's uncertainty, so mean per-token NLL is high almost by
construction. "Select the highest-perplexity examples" therefore reduces, on this
pool, to "select the one-token factoid answers" — a **9× enrichment** of ≤20-character
responses relative to the pool. Fine-tuning on that teaches the model to answer
tersely, which mismatches a held-out distribution whose median response is 187
characters, and the held-out loss duly rises.

This is worth stating plainly because it is a property of the *metric*, not of the
pool: response-length confounding will afflict any length-normalised perplexity
criterion applied to a corpus with mixed response lengths. A practitioner adopting
perplexity selection should either length-stratify or normalise for it. We did
neither, deliberately, because our object of study was the method as commonly
described.

IFD is partially protected by construction. Its ratio of conditional to
unconditional perplexity cancels much of the length effect, and its selections show
23% short responses rather than 55%. It is also the only method that beat random.
That is a coherent story — but with two seeds it is a hypothesis this study
suggests rather than establishes.

## 6.3 The cost-class question, restated

Combining §5.1 and §6.1: the training-based method costs 83.6% of total compute,
scores worse than random, and shares 31% of its selections with a training-free
method costing 59%. Nothing in these results identifies what the proxy epoch buys.
That is a negative result about a specific method at a specific scale, not a general
claim — but it is the result, and the accounting that produced it is the paper's
contribution.

---

# 8. Conclusion

We asked whether the reported efficiency of instruction-tuning data selection
survives two assumptions the literature has left untested: that the cost of
selecting data is negligible, and that method rankings measured at one
hyperparameter setting generalise.

On the first, the answer is no. **Selection consumed between 42% and 84% of total
compute** for every method that uses a model to select. It is the dominant term, not
a footnote, and pricing it in removes perplexity from the quality/compute Pareto
frontier. The strongest method in the source literature — learning percentage, which
requires a full proxy training epoch — spent **83.6% of all FLOPs deciding what to
train on and then scored worse than choosing 700 examples uniformly at random.** At
this scale, on this pool, its selection step did not pay for itself.

Two further results were not anticipated. **More selected data made models worse**:
four of five methods scored worse at 10% than at 5% under a fixed epoch budget,
because doubling the data doubles the gradient steps. And **methods described in
similar terms select almost disjoint data** — pairwise overlap sits at chance for
most pairs, and perplexity's criterion turns out to select one-token factoid
answers at nine times their rate in the pool, which fully accounts for its poor
performance.

Our proposed contribution, a budget-aware selection rule summarising each method by
an effective data multiplier, **could not be tested**. The scaling law it rests on
requires held-out loss to fall monotonically with the selection ratio, and over the
two ratios we could afford it does not. We report the degenerate fit rather than its
parameters, and identify the precondition we should have stated in advance: at least
three ratios spanning a monotone range, or a step count held fixed across ratios.

The learning-rate robustness question (RQ1) was not completed within the compute
budget. It remains open, and it is the natural next step.

What we can defend is narrower than we set out to show and, we think, more useful
than another benchmark table. Efficiency claims for data selection are currently
made against an accounting that omits their largest cost term. When that term is
measured — on identical hardware, in hardware-independent units — the ordering
changes, and the cheapest possible baseline becomes very hard to beat. Any future
claim that a selection method improves efficiency should be required to state the
budget at which it does so.
