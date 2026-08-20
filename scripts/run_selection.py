"""Stage 1: run a selection method and write its artifact.

Runs on the laptop CPU for every method except learning_percentage, which needs a
GPU (see methods.LearningPercentageSelector). The artifact it writes -- indices +
measured cost + full selector config -- is committed to git and is the ONLY thing
the Kaggle training stage reads. Training never re-runs selection, so a method's
cost is measured once and a run is never contaminated by a re-selection.

    python scripts/run_selection.py --method perplexity --ratio 0.05 --seed 0
    python scripts/run_selection.py --method random --ratio 0.05 --seed 0 1 2
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import selection  # noqa: E402,F401  (registers selectors)
from data import load_split, load_train_examples  # noqa: E402
from paths import SELECTIONS, selection_path  # noqa: E402
from selection.base import REGISTRY, build  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    split = load_split()
    examples = load_train_examples()
    k = int(round(args.ratio * len(examples)))
    print(f"pool={len(examples)}  ratio={args.ratio}  k={k}  method={args.method}")

    SELECTIONS.mkdir(parents=True, exist_ok=True)

    for seed in args.seed:
        out = selection_path(args.method, args.ratio, seed)
        if out.exists() and not args.overwrite:
            print(f"skip (exists): {out.name}")
            continue

        kwargs = {} if args.method == "random" else {"device": args.device}
        selector = build(args.method, **kwargs)
        idx, cost = selector.select(examples, k, seed)

        if len(set(idx.tolist())) != k:
            raise RuntimeError(f"{args.method} returned {len(set(idx.tolist()))} unique of {k}")

        out.write_text(json.dumps({
            "method": args.method,
            "cost_class": selector.cost_class,
            "ratio": args.ratio,
            "seed": seed,
            "k": k,
            "pool_size": len(examples),
            # Indices are positions in split["train_idx"], not raw Dolly rows.
            "local_idx": [int(i) for i in idx],
            "pool_idx": [int(split["train_idx"][i]) for i in idx],
            "selector_config": selector.config(),
            "cost": cost.as_dict(),
            "env": {"git_sha": git_sha(), "python": platform.python_version(),
                    "platform": platform.platform()},
        }, indent=2))

        print(f"wrote {out.name}  "
              f"wall={cost.wall_clock_s:.1f}s  flops={cost.flops:.3e}  "
              f"class={selector.cost_class}")


if __name__ == "__main__":
    main()
