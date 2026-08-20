#!/usr/bin/env bash
# Order matters: perplexity first, because ifd and hybrid reuse its cached scores.
set -u
log() { echo "[$(date +%H:%M:%S)] $*"; }
for m in perplexity diversity ifd hybrid; do
  log "=== $m: 5% ==="
  python scripts/run_selection.py --method "$m" --ratio 0.05 --seed 0 1 2 2>&1 | grep -vE "it/s\]|^Warning|torch_dtype"
  log "=== $m: 10% ==="
  python scripts/run_selection.py --method "$m" --ratio 0.10 --seed 0 1 2 2>&1 | grep -vE "it/s\]|^Warning|torch_dtype"
done
log "ALL SELECTIONS COMPLETE"
ls -1 results/selections/ | wc -l
