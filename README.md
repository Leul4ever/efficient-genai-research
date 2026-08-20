# Does Proxy-Based Data Selection Survive Contact With Reality?

Robustness and net-cost accounting for efficient instruction tuning.
Research plan: [docs/revised_research_plan.md](docs/revised_research_plan.md).
Kaggle operations: [docs/kaggle.md](docs/kaggle.md).

## The one architectural idea

**Selection and training never run in the same process.**

Selection runs on a laptop CPU, writes a JSON artifact (chosen indices + measured
cost + full selector config), and that artifact is committed to git. Training runs
on a Kaggle GPU and reads the artifact. It never selects anything itself.

Three things fall out of that split, and all three are load-bearing for the paper:

1. A method's selection cost is **measured once** and becomes a fixed, citable
   number. RQ2's whole argument is about that number, so it must not drift per run.
2. GPU quota — 30 h/week, non-bankable — is spent only on training.
3. A run is reproducible from `(config, selection artifact)` alone.

## Pipeline

GitHub is the only thing the two machines share.

```
                          +--------------------+
                          |       GitHub       |
                          |  code / configs /  |
                          |  results / paper   |
                          +---------+----------+
                                    | clone / pull / push
                +-------------------+-------------------+
                |                                       |
        +-------v--------+                     +--------v-------+
        |       PC       |                     |     KAGGLE     |
        |  selection     |                     |  QLoRA train   |
        |  analysis      |                     |  evaluation    |
        |  figures       |                     |  learning_pct  |
        |  paper         |                     |  selection     |
        +-------+--------+                     +--------+-------+
                |                                       |
                +------------------+--------------------+
                                   v
                        results/runs.jsonl      <- union-merged, append-only
                                   v
                     scripts/analyze.py   -> statistics
                     scripts/figures.py   -> results/figures/*.pdf
                                   v
                             paper/paper.md
```

| Stage | Command | Where | Output |
|---|---|---|---|
| Split | `scripts/make_split.py` | PC, once | `results/split.json` **[commit]** |
| Select | `scripts/run_selection.py` | PC (CPU) | `results/selections/*.json` **[commit]** |
| Train | `src/runner.py` → `src/train.py` | Kaggle GPU | `results/runs.jsonl` **[commit]** |
| Analyse | `scripts/analyze.py` | PC | the three RQ sections |
| Figures | `scripts/figures.py` | PC | `results/figures/*.{pdf,png}` |
| Write | `paper/paper.md` | PC | the deliverable |

`results/runs.jsonl` is the convergence point, and that is exactly where it can go
wrong: two sharded Kaggle notebooks both appending at EOF and both pushing produce
a rebase conflict on **every** push. `.gitattributes` gives the file git's
`merge=union` driver so both sides' lines survive, and `registry.load_runs()`
de-duplicates by `run_id` on read. Without that pair, the natural way to resolve
the conflict under time pressure is to pick one side — silently destroying a
completed run and the GPU-hours that produced it.

## Quickstart

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                    # 32 checks, no downloads, seconds
python scripts/make_split.py                    # once; then commit results/split.json

# Selection: laptop CPU. Time these -- the numbers are primary results, not footnotes.
# MEASURED on a laptop CPU: ~300 ms/example, so ~70 min per full pass over 14,000
# with a 135M scorer. Scores are cached by scorer config, so a method pays that
# once and every (ratio, seed) after it is instant.
for m in random perplexity diversity hybrid ifd; do
  python scripts/run_selection.py --method $m --ratio 0.05 --seed 0 1 2
  python scripts/run_selection.py --method $m --ratio 0.10 --seed 0 1 2
done
# learning_percentage needs a GPU -- run it in the Kaggle notebook with --device cuda

git add results/ && git commit -m "frozen split + selection artifacts"

# Training: Kaggle. See docs/kaggle.md; paste notebooks/kaggle_cell.py into a cell.
python src/runner.py --config configs/study1_main_grid.yaml --dry-run
python src/runner.py --config configs/study1_main_grid.yaml --budget-hours 8

# Analysis and figures: laptop.
python scripts/analyze.py --all
python scripts/figures.py            # real results
python scripts/figures.py --demo     # synthetic + watermarked, to check layout now
```

## Studies

| Config | Runs | RQ | Notes |
|---|---|---|---|
| `study1_main_grid.yaml` | 36 | — | 6 methods × 2 ratios × 3 seeds. The only study running the pairwise judge. |
| `study1_anchors.yaml` | 3 | — | Full-data ceiling. Expensive: ~20× a 5% run. |
| `study2_lr_sweep.yaml` | 24 | RQ1 | **Run first in Week 3.** Carries the paper. |
| `study3_scale_transfer.yaml` | 16 | RQ3 | Reuses Study 1's selections unchanged — that is the point. |
| `study4_cross_family.yaml` | 6 | RQ3/RQ4 | Held-out test of the rule. Underpowered for significance **by design**. |

85 training runs total. Drop Study 4 first if the quota bites, then Study 3;
Studies 1 + 2 alone are a complete paper.

## Design rules the code enforces

- **The split is frozen.** `data.py` hashes the prompt template into `split.json`
  and refuses to load if the template changed afterwards. Silent template drift
  would invalidate every prior run without raising anything.
- **A run's identity is its config hash.** `registry.run_id()` hashes only the
  fields that define the experiment, so re-running is idempotent and changing a
  hyperparameter can never overwrite an old result.
- **Held-out loss is stored per example, not as a mean.** Paired bootstrap needs
  the vector; storing the mean throws away statistical power irreversibly.
- **Every difference gets a CI.** `analyze.py` prints `INSIDE NOISE` for
  differences whose CI straddles zero, and applies Holm-Bonferroni across the grid.
- **A score is computed once, not once per condition.** Score-based selectors are
  deterministic -- `seed` changes nothing -- so `results/scores/` caches by scorer
  config. Without it the main grid recomputes each score six times, which at
  measured CPU rates turns 8.5 hours of IFD scoring into 51.
- **No method is "best" without a budget.** `policy.py` fits an effective data
  multiplier per method and a crossover budget below which the method does not
  repay its own selection cost. Fitted on Study 1, validated on Studies 3 and 4,
  at zero extra GPU cost.
- **Failures are logged, not dropped.** A run that diverges to `nan` at one
  learning rate and not another is RQ1's signal, not an inconvenience.

## Two things to disclose in the paper

1. **IFD here is not Cherry-LLM's IFD.** The original uses a briefly fine-tuned
   "experience" model, which makes it training-based. This implementation uses an
   off-the-shelf pretrained scorer, keeping it training-free. Deliberate, and it
   changes which cost class the method falls into — say so in §3.
2. **Wall-clock is not comparable across cost classes.** `learning_percentage`
   runs on a GPU; the training-free methods run on a CPU. Use analytical FLOPs for
   all cross-method cost claims. See `cost.py`.

## Layout

```
configs/       frozen hyperparameters; base.yaml + one file per study
src/
  data.py      frozen split, prompt template
  cost.py      FLOP + wall-clock ledger (RQ2's foundation)
  registry.py  content-addressed run IDs, append-only JSONL
  stats.py     paired bootstrap, ranking stability, Pareto, Holm-Bonferroni
  policy.py    the budget-aware selection rule -- the constructive contribution
  train.py     one condition -> one line in runs.jsonl
  evaluate.py  held-out loss (per example), lm-eval harness, generations
  runner.py    sweep driver: subprocess per run, resumable, shardable
  selection/   the six methods + the submodular solver
scripts/       make_split, run_selection, analyze, figures, smoke_test
notebooks/     kaggle_cell.py
paper/         paper.md -- section skeleton wired to the command producing each number
docs/kaggle.md quota, sharding, session survival, failure modes
```
