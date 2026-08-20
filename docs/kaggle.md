# Running on Kaggle

Everything in this document exists because of one number: **~30 GPU-hours per week,
reset weekly, not bankable.** Unused hours in Week 1 do not roll into Week 3. The
plan's "120 h over four weeks with 68 h headroom" is true as a total and misleading
as a schedule — the headroom only exists if the draw is spread evenly.

## The split that shapes everything

| Stage | Where | Why |
|---|---|---|
| Frozen split | laptop, once | `scripts/make_split.py`, committed |
| Selection (5 of 6 methods) | **laptop CPU** | forward passes only; costs zero GPU quota |
| Selection (`learning_percentage`) | **Kaggle GPU** | needs a full proxy training epoch |
| Training + eval | **Kaggle GPU** | 85 runs |
| Analysis + figures | laptop | reads `results/runs.jsonl` |

Selection artifacts are committed to git and are the *only* thing the GPU stage
reads. Training never re-runs selection. That is what makes a method's measured
cost a fixed, citable number rather than something that drifts per run.

### The cost-comparison wrinkle you must disclose

`learning_percentage` runs on a GPU while the training-free methods run on a CPU.
Their wall-clock times are therefore **not comparable**. Use analytical FLOPs
(`cost.py`) for every cross-method cost claim, and report wall-clock only within a
device. Say this explicitly in §4 — a reader who notices it before you do will
assume the whole ledger is sloppy.

## One-time setup

1. **Notebook settings** → Accelerator **GPU T4 x2**, Internet **On**.
   Internet is off by default; without it, HF downloads and `git push` both fail.
2. **Add-ons → Secrets**:
   - `GITHUB_TOKEN` — fine-grained PAT, `contents: write`, scoped to this repo only.
   - `HF_TOKEN` — needed for **Llama-3.2-1B, which is gated**. Request access on the
     model page **in Week 1**, not in Week 3. Approval is not instant, and Study 4
     cannot start without it.
3. Edit `REPO` at the top of `notebooks/kaggle_cell.py`, paste the file into one
   cell, run.

## Use both GPUs

"T4 x2" is two separate 16 GB cards, not one 32 GB card. A 0.5B or 1.5B model in
4-bit fits on one comfortably, so model parallelism buys nothing. Run **two
notebooks** instead:

```
notebook A:  SHARD=0, N_SHARDS=2   # CUDA_VISIBLE_DEVICES=0
notebook B:  SHARD=1, N_SHARDS=2   # CUDA_VISIBLE_DEVICES=1
```

Sharding is on the `run_id` hash, not list position, so a resumed session picks up
the same subset it started with. Both notebooks push to the same branch, so the
push step rebases first.

Note the quota consequence: two concurrent sessions burn quota at **twice the rate**.
Half the wall-clock, identical GPU-hours.

## Rebalanced weekly budget

The plan's original schedule back-loads Studies 2, 3, and 4 into Week 3, which
needs well over 30 h in a single week. Recommended redistribution:

| Week | Work | Approx GPU-h |
|---|---|---|
| 1 | Pipeline, `random` + `full` on 0.5B × 3 seeds, one timed run to calibrate | ~10 |
| 2 | Study 1 main grid (36 runs) + Study 3 (16 runs) | ~26 |
| 3 | **Study 2 first** (24 runs, carries the paper), then Study 4 (6 runs) | ~22 |
| 4 | Writing. Reserve ~5 h for reruns; **no new experiments** | ~5 |

Keep `--budget-hours` at or below 8 per session, and check the sidebar's quota
readout before starting a second concurrent notebook.

## Session survival

- **Sessions are killed at ~12 h and can be pre-empted sooner without warning.**
- `/kaggle/working` is **deleted** when the session ends. Results that were not
  pushed did not happen, and the GPU-hours that produced them are gone from the
  weekly quota permanently. The cell pushes at the end; if you interrupt a sweep
  manually, run the push block before closing.
- `runner.py` launches each run as a subprocess, so a CUDA OOM kills one condition
  and the sweep continues. `train.py` appends to `runs.jsonl` with `fsync` the
  moment a run finishes, so a kill costs at most one run.
- Resuming is just re-running the same command: completed `run_id`s are skipped.
- **Stop the session when you are done.** An idle GPU session keeps burning quota.

## Calibrate before you trust the budget

Do this in Week 1, before committing to 85 runs:

```bash
python src/runner.py --config configs/study1_main_grid.yaml --dry-run
python src/train.py --config configs/study1_main_grid.yaml \
    --set selection_method=random ratio=0.05 seed=0
```

Read `cost.training_wall_clock_s` out of `results/runs.jsonl` and multiply. Then do
the same for one `ratio=1.0` ceiling run — the full-data runs are ~20× the 5% runs
and dominate the budget. If the projection exceeds ~26 h in any week, cut Study 4
first, then Study 3, exactly as the plan's risk table says. Studies 1 + 2 alone are
a complete paper.

## Known failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `401` on Llama-3.2 | gated model, no/unapproved `HF_TOKEN` | request access early; verify in Week 1 |
| `bitsandbytes` import error | version drift against torch | keep the pins in `kaggle_cell.py` |
| OOM on the 1.5B runs | both shards landed on one GPU | confirm `CUDA_VISIBLE_DEVICES` differs |
| `git push` rejected | the other shard pushed first | the cell rebases; re-run the push block |
| Loss is `nan` from step 1 | fp16 overflow at high LR | expected at `lr=2e-4` on some models — **this is a data point for RQ1, log it, do not silently drop it** |
| Harness metrics missing | `lm-eval` threw | run is still valid for held-out loss; `evaluate.py` catches this deliberately |

The `nan` row deserves emphasis. A run that diverges at one learning rate and not
another is precisely the fragility RQ1 is measuring. Record it as a failed run with
its config, and report the failure rate per method per LR in the paper.
