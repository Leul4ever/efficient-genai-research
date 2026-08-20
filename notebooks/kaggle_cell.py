"""Paste this into a single Kaggle notebook cell. It is the whole GPU-side workflow.

Kaggle constraints this code is shaped around:
  * ~30 GPU-hours per WEEK, reset weekly and NOT bankable. Unused Week 1 hours do
    not roll into Week 3. --budget-hours enforces the per-session share.
  * Sessions are killed at ~12h, and can be pre-empted sooner without warning.
    Everything must be resumable, and results must leave the container before the
    session dies -- /kaggle/working is lost unless it is saved or pushed.
  * "T4 x2" is two separate 16GB GPUs, not one 32GB GPU. A 1.5B model in 4-bit fits
    on one comfortably, so the right use of the second GPU is a SECOND PROCESS, not
    model parallelism. See SHARD below: it roughly halves wall-clock for free.
  * Internet must be switched ON in the notebook sidebar or HF downloads and the
    git push both fail. This is off by default on new notebooks.

Setup, once:
  1. Notebook settings -> Accelerator: GPU T4 x2, Internet: On.
  2. Add-ons -> Secrets: add GITHUB_TOKEN (a fine-grained PAT with contents:write
     on this repo only) and HF_TOKEN (needed for Llama-3.2, which is gated --
     request access on the model page BEFORE Week 3 or Study 4 fails at 3am).
  3. Set REPO below to your fork.
"""

# ---------------------------------------------------------------- configuration
REPO = "YOUR_GITHUB_USERNAME/efficient-genai-research"
BRANCH = "main"
CONFIG = "configs/study1_main_grid.yaml"
BUDGET_HOURS = 8.0     # per session; keep the weekly total under ~28h
SHARD, N_SHARDS = 0, 1  # run a second notebook with SHARD=1, N_SHARDS=2 to use both GPUs

# ------------------------------------------------------------------- environment
import os, subprocess, sys, textwrap

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")
try:
    os.environ["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
except Exception:
    print("WARN: no HF_TOKEN secret; gated models (Llama-3.2) will 401")

# Pin versions. An unpinned install that silently upgrades peft or transformers
# mid-project makes Week 1 runs incomparable with Week 3 runs, and you will not
# notice until the numbers refuse to line up.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "peft==0.13.2", "bitsandbytes==0.44.1", "accelerate==1.0.1",
                "transformers==4.46.2", "datasets==3.1.0", "sentence-transformers==3.2.1",
                "lm-eval==0.4.5"], check=True)

WORK = "/kaggle/working/efficient-genai-research"
if not os.path.exists(WORK):
    subprocess.run(["git", "clone", "--branch", BRANCH,
                    f"https://{GITHUB_TOKEN}@github.com/{REPO}.git", WORK], check=True)
else:
    subprocess.run(["git", "-C", WORK, "pull", "--rebase"], check=True)

os.chdir(WORK)
subprocess.run(["git", "config", "user.email", "kaggle@runner.local"], check=True)
subprocess.run(["git", "config", "user.name", "kaggle-runner"], check=True)

# HF caches to /kaggle/working by default, which counts against the 20GB output
# limit AND gets re-downloaded every session. /kaggle/temp is bigger and faster.
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(SHARD) if N_SHARDS > 1 else "0"

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)

# ------------------------------------------------------------------------ verify
# The selection artifacts must already be in the repo. If they are not, someone
# skipped the CPU stage and the sweep would fail one run at a time over hours.
import glob

n_sel = len(glob.glob("results/selections/*.json"))
assert os.path.exists("results/split.json"), "results/split.json missing -- commit the frozen split"
assert n_sel > 0, "no selection artifacts -- run scripts/run_selection.py locally and push"
print(f"{n_sel} selection artifacts present")

subprocess.run([sys.executable, "scripts/smoke_test.py"], check=True)

# --------------------------------------------------------------------- the sweep
cmd = [sys.executable, "src/runner.py", "--config", CONFIG,
       "--budget-hours", str(BUDGET_HOURS)]
if N_SHARDS > 1:
    cmd += ["--shard", str(SHARD), "--n-shards", str(N_SHARDS)]

# Streamed, not captured: if the session is killed you still want the log visible
# in the notebook output to see how far it got.
subprocess.run(cmd, check=False)

# ------------------------------------------------------------------- push results
# This is the step people forget. /kaggle/working is DELETED when the session ends.
# Results that are not pushed did not happen, and the GPU-hours that produced them
# are gone from the weekly quota permanently.
subprocess.run(["git", "add", "results/runs.jsonl", "results/runs"], check=False)
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
if status.strip():
    subprocess.run(["git", "commit", "-m", f"kaggle: {os.path.basename(CONFIG)} shard {SHARD}"],
                   check=True)
    # Rebase first: the other shard's notebook is pushing to the same branch.
    subprocess.run(["git", "pull", "--rebase"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("results pushed")
else:
    print("nothing new to push")

print(textwrap.dedent("""
    Session done. Before closing:
      * check the run count:  wc -l results/runs.jsonl
      * check remaining quota in the Kaggle sidebar, and STOP the session --
        an idle GPU session keeps burning the weekly allowance.
"""))
