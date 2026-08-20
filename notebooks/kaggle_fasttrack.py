"""FAST TRACK -- one Kaggle cell, 24-hour deadline.

HOW TO RUN IT. Do not paste this file into Kaggle. Paste these two lines instead:

    !git clone https://github.com/Leul4ever/efficient-genai-research.git /kaggle/working/efficient-genai-research 2>/dev/null || git -C /kaggle/working/efficient-genai-research pull
    %run /kaggle/working/efficient-genai-research/notebooks/kaggle_fasttrack.py

The repo is public, so that clone needs no credentials. The reason to bootstrap
rather than paste: train.py and evaluate.py have never executed anywhere, so
fixes are likely tonight, and `git pull` picks them up on the next run. Re-pasting
a 180-line cell at 3am is how a transcription typo eats an hour.


Do NOT run the four-week workflow in notebooks/kaggle_cell.py. This file does the
whole job in one pass and, critically, FAILS EARLY: it proves a single training run
works before committing hours to a sweep. train.py and evaluate.py have never
executed anywhere, so a first-run bug is likely -- find it in minute four, not at
three in the morning.

Order is deliberate:
    1. setup and smoke test          ~3 min
    2. selection, on GPU             ~25 min   (hours on a laptop CPU; not tonight)
    3. ONE calibration run           ~8 min    <- STOP HERE if it fails
    4. the grid, then the LR sweep   ~4 h
    5. push after every stage        so a dead session never costs everything

Set STAGE below to re-enter at any step after a crash.
"""

# ---------------------------------------------------------------- configuration
REPO = "Leul4ever/efficient-genai-research"
BRANCH = "main"
STAGE = "all"          # "setup" | "select" | "calibrate" | "sweep" | "all"
BUDGET_HOURS = 7.0

# Leave empty to use the Kaggle secret (Add-ons -> Secrets). Kaggle requires a
# secret to be ATTACHED to each notebook, not merely created -- an unattached
# secret raises BackendError "No user secrets exist for kernel id ...".
# If attaching is fighting you, paste the token here instead: this notebook is
# private, and a deadline beats tidiness. Rotate the token once you have submitted,
# and never make a notebook containing it public.
INLINE_GITHUB_TOKEN = ""

# ------------------------------------------------------------------- environment
import json
import os
import subprocess
import sys
import time

T_START = time.time()


def sh(cmd, check=False, **kw):
    """Run a command, streaming output so a killed session still shows progress."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=check, **kw)


def elapsed():
    return f"[{(time.time() - T_START) / 60:6.1f} min]"


def resolve_token() -> str:
    """A token is OPTIONAL now that the repo is public: cloning needs no auth, and
    only pushing results back does. Missing it must therefore not stop the run --
    the GPU work is the scarce thing, and results can still be downloaded from the
    notebook's Output tab if the push never works."""
    if INLINE_GITHUB_TOKEN.strip():
        print("auth: INLINE_GITHUB_TOKEN")
        return INLINE_GITHUB_TOKEN.strip()
    try:
        from kaggle_secrets import UserSecretsClient

        tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
        print("auth: Kaggle secret GITHUB_TOKEN")
        return tok
    except Exception as exc:
        print(chr(10).join([
            "",
            f"NOTE: no GitHub token ({type(exc).__name__}). Continuing anyway.",
            "The repo is public, so the clone below works unauthenticated and every",
            "experiment will run. Only pushing results back needs a token.",
            "",
            "To enable pushing, either:",
            "  A) Add-ons -> Secrets -> create GITHUB_TOKEN, then tick ATTACH for",
            "     THIS notebook. Creating the secret is not enough on its own.",
            "  B) Set INLINE_GITHUB_TOKEN at the top of this cell.",
            "Without one, download results/ from the Output tab BEFORE closing the",
            "session -- /kaggle/working is deleted when it ends.",
            "",
        ]))
        return ""


GITHUB_TOKEN = resolve_token()

# Pinned. An unpinned install that upgrades peft or transformers midway makes the
# early runs incomparable with the late ones, and you would not notice until the
# numbers refuse to line up at 6am.
sh([sys.executable, "-m", "pip", "install", "-q",
    "peft==0.13.2", "bitsandbytes==0.44.1", "accelerate==1.0.1",
    "transformers==4.46.2", "datasets==3.1.0", "sentence-transformers==3.2.1",
    "lm-eval==0.4.5"])

WORK = "/kaggle/working/efficient-genai-research"
# Unauthenticated clone: the repo is public. Keeping the token out of the remote
# URL also keeps it out of `git remote -v`, the reflog, and any traceback that
# prints the command -- which matters more now that the repo is public.
if not os.path.exists(WORK):
    sh(["git", "clone", "--branch", BRANCH,
        f"https://github.com/{REPO}.git", WORK], check=True)
else:
    sh(["git", "-C", WORK, "pull", "--rebase"])

os.chdir(WORK)
sh(["git", "config", "user.email", "kaggle@runner.local"])
sh(["git", "config", "user.name", "kaggle-runner"])

os.environ["HF_HOME"] = "/kaggle/temp/hf"          # not /kaggle/working: 20GB output cap
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)


def rescue_note():
    print(f"Results are NOT lost: they are in {WORK}/results/. Before closing "
          "this session, download results/ from the notebook's Output tab. "
          "/kaggle/working is deleted when the session ends.")


def push(msg):
    """Commit always, push only if a token exists.

    Committing even without a token is deliberate: it timestamps the work in the
    local history, so a token added mid-session pushes everything accumulated so
    far rather than only what came after."""
    sh(["git", "add", "results/"])
    status = subprocess.run(["git", "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    if not status.strip():
        print("nothing new to commit")
        return
    sh(["git", "commit", "-m", msg])

    if not GITHUB_TOKEN:
        print("COMMITTED LOCALLY -- no token, so not pushed.")
        rescue_note()
        return

    # Set the authenticated URL only at push time, so the token never lives in the
    # stored remote between calls.
    sh(["git", "remote", "set-url", "origin",
        f"https://{GITHUB_TOKEN}@github.com/{REPO}.git"])
    sh(["git", "pull", "--rebase"])
    r = sh(["git", "push"])
    sh(["git", "remote", "set-url", "origin", f"https://github.com/{REPO}.git"])

    if r.returncode == 0:
        print("PUSHED")
    else:
        print("PUSH FAILED.")
        rescue_note()


# ------------------------------------------------------------ 1. setup + smoke
if STAGE in ("setup", "all"):
    print(f"\n{elapsed()} ===== STAGE 1: smoke test =====")
    r = sh([sys.executable, "scripts/smoke_test.py"])
    assert r.returncode == 0, "smoke test failed -- stop and fix before burning GPU time"

# --------------------------------------------------------------- 2. selection
# On GPU because the deadline does not fit CPU selection: measured at ~300 ms per
# example on a laptop, one perplexity pass over 14,000 examples is ~70 minutes and
# learning_percentage needs a full proxy training epoch on top.
#
# CONSEQUENCE FOR THE PAPER: wall-clock for these methods is now GPU wall-clock and
# is NOT comparable to the laptop-CPU figures. Use analytical FLOPs for every
# cross-method cost claim -- cost.py already reports both -- and state the change.
if STAGE in ("select", "all"):
    print(f"\n{elapsed()} ===== STAGE 2: selection on GPU =====")
    for method in ["perplexity", "diversity", "ifd", "learning_percentage"]:
        for ratio in ["0.05", "0.10"]:
            sh([sys.executable, "scripts/run_selection.py", "--method", method,
                "--ratio", ratio, "--seed", "0", "1", "--device", "cuda"])
    sh(["ls", "-1", "results/selections/"])
    push("kaggle: selection artifacts (GPU)")

# ------------------------------------------------------------- 3. calibration
# The whole point of this stage: prove one run works, and measure it, BEFORE
# committing the night to 27 of them.
if STAGE in ("calibrate", "all"):
    print(f"\n{elapsed()} ===== STAGE 3: one calibration run =====")
    r = sh([sys.executable, "src/train.py", "--config", "configs/fast_grid.yaml",
            "--set", "selection_method=random", "ratio=0.05", "seed=0"])
    assert r.returncode == 0, "calibration run FAILED -- fix this before the sweep"

    runs = [json.loads(l) for l in open("results/runs.jsonl") if l.strip()]
    last = runs[-1]
    per_run_min = last["cost"]["training_wall_clock_s"] / 60
    eval_min = last["cost"]["evaluation_wall_clock_s"] / 60
    print(f"\n{elapsed()} MEASURED: {per_run_min:.1f} min train + {eval_min:.1f} min eval")
    print(f"  loss = {last['metrics']['held_out_loss']:.4f}")
    print(f"  29 remaining runs project to "
          f"{(per_run_min + eval_min) * 29 / 60:.1f} h")
    if (per_run_min + eval_min) * 29 / 60 > BUDGET_HOURS:
        print("  OVER BUDGET -- drop the LR sweep to 2 methods, or ratios to 0.05 only")
    push("kaggle: calibration run")

# -------------------------------------------------------------------- 4. sweep
# Grid first (it is the paper's spine), LR sweep second (it is the novel core),
# full-data anchor last (one run, but 10-20x the cost of any other).
if STAGE in ("sweep", "all"):
    for cfg, label in [("configs/fast_grid.yaml", "grid"),
                       ("configs/fast_lr_sweep.yaml", "lr-sweep"),
                       ("configs/fast_anchor.yaml", "full-data anchor")]:
        left = BUDGET_HOURS - (time.time() - T_START) / 3600
        if left <= 0.3:
            print(f"\n{elapsed()} BUDGET EXHAUSTED before {label}. "
                  f"Re-run this cell with STAGE='sweep' -- finished runs are skipped.")
            break
        print(f"\n{elapsed()} ===== SWEEP: {label} ({left:.1f} h left) =====")
        sh([sys.executable, "src/runner.py", "--config", cfg,
            "--budget-hours", str(round(left, 2))])
        push(f"kaggle: {label}")

# ------------------------------------------------------------------------ done
n = sum(1 for l in open("results/runs.jsonl") if l.strip()) if os.path.exists("results/runs.jsonl") else 0
print(f"\n{elapsed()} ===== DONE: {n} runs logged =====")
print("Confirm the push landed on GitHub, then STOP the session -- an idle GPU")
print("session keeps burning the weekly quota.")
