"""Run once. Creates results/split.json. Commit the result."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import make_split  # noqa: E402
from paths import SPLIT  # noqa: E402

if __name__ == "__main__":
    if SPLIT.exists():
        raise SystemExit(f"{SPLIT} already exists. Refusing to overwrite a frozen split.")
    SPLIT.parent.mkdir(parents=True, exist_ok=True)
    split = make_split()
    SPLIT.write_text(json.dumps(split, indent=2))
    print(f"Wrote {SPLIT}: {len(split['train_idx'])} train / {len(split['held_out_idx'])} held-out")
    print(f"template_hash={split['template_hash']}  -- commit this file now.")
