"""Sweep driver. Expands a study's grid and runs each condition as a subprocess.

Subprocess, not an in-process loop, for three reasons that all bite on Kaggle:
  1. a CUDA OOM in one condition kills that process, not the sweep;
  2. GPU memory is reclaimed completely between runs, so run N+1 is not affected
     by fragmentation from run N -- which would otherwise make wall-clock, a
     primary result here, depend on run order;
  3. the sweep is resumable, because train.py logs each run the moment it finishes.

    python src/runner.py --config configs/study2_lr_sweep.yaml --budget-hours 20
    python src/runner.py --config configs/study1_main_grid.yaml --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import yaml

from registry import append_run, completed_ids, run_id


def expand_grid(cfg: dict) -> list[dict]:
    """Cartesian product over any key whose value is a list under `grid:`."""
    grid = cfg.get("grid") or {}
    base = {k: v for k, v in cfg.items() if k != "grid"}
    keys = sorted(grid)
    conditions = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cond = {**base, **dict(zip(keys, combo))}
        # The full-data ceiling has no selection method; collapse the duplicates
        # the product would otherwise generate across method names.
        if cond.get("ratio", 0) >= 1.0:
            cond["selection_method"] = "full"
        conditions.append(cond)

    seen, unique = set(), []
    for cond in conditions:
        key = tuple(sorted((k, str(v)) for k, v in cond.items()))
        if key not in seen:
            seen.add(key)
            unique.append(cond)
    return unique


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="stop launching new runs once this much wall-clock is spent. "
                         "Kaggle's GPU quota is weekly and non-bankable, so guard it.")
    ap.add_argument("--shard", type=int, default=0,
                    help="which shard this process runs. Kaggle's 'T4 x2' is two "
                         "separate 16GB GPUs; two sharded processes, one per GPU, "
                         "roughly halve wall-clock for a sweep of small models.")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--force", action="store_true",
                    help="re-run conditions already in runs.jsonl. Needed to "
                         "regenerate per-example loss artifacts when a previous "
                         "run's files were lost with the machine that produced "
                         "them -- the metrics survive in runs.jsonl, the .npy "
                         "vectors the paired bootstrap needs do not.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard must be in [0, {args.n_shards})")

    cfg = yaml.safe_load(args.config.read_text())
    conditions = expand_grid(cfg)
    done = completed_ids()

    # template_hash is injected by train.py, but run_id needs it to match. Read it here.
    from data import load_split

    template_hash = load_split()["template_hash"]

    pending = ([c for c in conditions
                if run_id({**c, "template_hash": template_hash}) not in done]
               if not args.force else list(conditions))

    if args.n_shards > 1:
        # Shard on the run_id hash, not on list position. Position-based sharding
        # would reshuffle every time a run completes and the pending list shrinks,
        # so a resumed session would pick up a different subset than it started with.
        pending = [c for c in pending
                   if int(run_id({**c, "template_hash": template_hash}), 16) % args.n_shards
                   == args.shard]

    print(f"{args.config.name}: {len(conditions)} conditions, {len(done)} already logged, "
          f"{len(pending)} pending"
          + (f" on shard {args.shard}/{args.n_shards}" if args.n_shards > 1 else ""))

    if args.dry_run:
        for c in pending:
            print(f"  {c['selection_method']:>20}  r={c['ratio']:<6} lr={c['learning_rate']:<8} "
                  f"seed={c['seed']}  model={c['target_model']}")
        return

    t_start = time.perf_counter()
    failures = []

    for i, cond in enumerate(pending, 1):
        elapsed_h = (time.perf_counter() - t_start) / 3600
        if args.budget_hours and elapsed_h >= args.budget_hours:
            print(f"BUDGET STOP: {elapsed_h:.2f}h spent, {len(pending) - i + 1} runs left. "
                  f"Re-run this command next session; completed runs will be skipped.")
            break

        overrides = [f"{k}={v}" for k, v in cond.items()
                     if k not in ("study",) and not isinstance(v, (list, dict))]
        cmd = [sys.executable, str(Path(__file__).parent / "train.py"),
               "--config", str(args.config), "--set", *overrides]
        if args.force:
            cmd.append("--force")
        print(f"\n[{i}/{len(pending)}] {' '.join(overrides)}")

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
        if result.returncode != 0:
            # Keep going -- one dead condition should not cost the remaining sweep --
            # but RECORD it. A failure that only prints to stdout vanishes with the
            # session, and a sweep that quietly drops two thirds of its conditions
            # still looks like a completed sweep in results/runs.jsonl.
            print(f"  FAILED (exit {result.returncode}) -- continuing")
            failures.append(cond)
            try:
                append_run({
                    "run_id": run_id({**cond, "template_hash": template_hash}),
                    "status": "failed",
                    "study": cond.get("study", "unknown"),
                    "config": cond,
                    "exit_code": result.returncode,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  (could not log the failure: {exc})")

    print(f"\nsweep finished in {(time.perf_counter() - t_start) / 3600:.2f}h, "
          f"{len(failures)} failure(s)")
    for f in failures:
        print(f"  failed: {f['selection_method']} r={f['ratio']} "
              f"lr={f['learning_rate']} seed={f['seed']}")


if __name__ == "__main__":
    main()
