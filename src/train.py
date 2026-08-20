"""Stage 2: QLoRA fine-tune one condition, evaluate it, append one JSONL line.

One process = one run = one line in results/runs.jsonl. No sweeps here; the sweep
lives in runner.py so that a killed Kaggle session loses at most one run.

Adaptation hyperparameters are IDENTICAL across every condition except the one
variable a study manipulates. That is the whole design: if two runs differ in more
than one place, neither tells you anything.

    python src/train.py --config configs/study1_main_grid.yaml \
        --set selection_method=perplexity ratio=0.05 seed=0
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from cost import CostRecord, total_cost, train_flops
from data import build_full, build_prompt, load_split, load_train_examples
from paths import RUN_ARTIFACTS, selection_path
from registry import append_run, completed_ids, run_id


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {item!r}")
        cfg[key] = yaml.safe_load(raw)  # infers int/float/bool/str
    cfg["template_hash"] = load_split()["template_hash"]
    return cfg


def load_selection(cfg: dict) -> tuple[list[int], dict | None, str]:
    """Returns (local indices into the train split, selection cost record, cost class).

    ratio == 1.0 is the full-data ceiling: no selection artifact, no selection cost.
    """
    if cfg["ratio"] >= 1.0:
        return list(range(len(load_split()["train_idx"]))), None, "none"

    path = selection_path(cfg["selection_method"], cfg["ratio"], cfg["seed"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run scripts/run_selection.py for this condition first. "
            "Training must never perform its own selection."
        )
    art = json.loads(path.read_text())
    return art["local_idx"], art["cost"], art["cost_class"]


def disable_bnb_in_peft() -> bool:
    """Stop peft from importing bitsandbytes when we are not quantizing.

    peft decides via `is_bnb_available()`, which is
    `importlib.util.find_spec("bitsandbytes") is not None` -- a check that answers
    "is it installed", NOT "does it import". Kaggle ships a bitsandbytes that is
    installed and broken: it imports `triton.ops`, removed in Triton 3.x. So the
    probe says yes, `_create_new_module` does `from .bnb import dispatch_bnb_8bit`,
    and get_peft_model dies on the fp16 path that never wanted bitsandbytes at all.

    Uninstalling the package fixes it too, but that is a property of the machine and
    has to be re-done on every fresh session. Overriding the probe is a property of
    this code, so it holds wherever the code runs. Only ever called when
    load_in_4bit is false, so nothing that genuinely needs bnb is affected.
    """
    patched = []
    for mod_name in ("peft.import_utils", "peft.tuners.lora.model",
                     "peft.tuners.lora.__init__", "peft.mapping"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for fn in ("is_bnb_available", "is_bnb_4bit_available"):
            if hasattr(mod, fn):
                setattr(mod, fn, lambda: False)
                patched.append(f"{mod_name}.{fn}")
    if patched:
        print(f"bnb probe disabled in peft ({len(patched)} hooks)")
    return bool(patched)


def build_model(cfg: dict):
    """LoRA on an fp16 base by default; 4-bit only if explicitly asked for.

    `load_in_4bit` defaults to FALSE. At the scales this project actually runs,
    4-bit buys nothing: Qwen2.5-0.5B in fp16 is roughly 1 GB and Qwen2.5-1.5B about
    3 GB, against 15 GB of T4. It also drags in bitsandbytes, which is the most
    fragile dependency in the stack -- 0.44.1 imports `triton.ops`, removed in
    Triton 3.x, and ships no CUDA 12.8 binary, so it fails outright on current
    Kaggle images.

    Dropping it is also cleaner science. Quantization is a confound: with 4-bit
    weights, a difference between selection methods is entangled with how each
    selected subset interacts with quantization error. fp16 removes that.

    Report it as LoRA, not QLoRA. The distinction matters and is cheap to state.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # After the peft import (its submodules must exist to be patched) and before
    # get_peft_model (which is where the bnb probe is consulted).
    if not cfg.get("load_in_4bit", False):
        disable_bnb_in_peft()

    tok = AutoTokenizer.from_pretrained(cfg["target_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    load_kwargs = {"device_map": {"": 0}, "torch_dtype": torch.float16}
    if cfg.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # T4 has no bf16; float16 is the only compute dtype available here.
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(cfg["target_model"], **load_kwargs)

    if cfg.get("load_in_4bit", False):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        # Gradient checkpointing needs the inputs to require grad, otherwise the
        # recomputed graph is detached from the frozen embedding and no LoRA
        # gradient ever arrives. prepare_model_for_kbit_training does this for the
        # 4-bit path; on the fp16 path it must be done explicitly.
        model.enable_input_require_grads()
        model.config.use_cache = False  # incompatible with checkpointing

    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=cfg.get("lora_targets",
                               ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]),
    ))

    # Trainable params in fp32. Adam's moments on fp16 parameters underflow at these
    # learning rates -- and the LR sweep deliberately visits 1e-6, where fp16 updates
    # would round to zero and silently turn RQ1 into a measurement of nothing.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    return tok, model


def tokenize(examples: list[dict], tok, max_len: int):
    """Prompt tokens are masked out: only the response contributes to the loss.

    Training on prompt tokens as well is a defensible alternative, but it changes
    the effective objective. Fix one choice in Week 1 and never revisit it.
    """
    from datasets import Dataset

    rows = []
    for ex in examples:
        prompt_ids = tok(build_prompt(ex), add_special_tokens=True)["input_ids"]
        full_ids = tok(build_full(ex, eos=tok.eos_token or ""),
                       add_special_tokens=True)["input_ids"][:max_len]
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100
        if all(l == -100 for l in labels):
            continue  # prompt filled the window; nothing to learn from
        rows.append({"input_ids": full_ids, "labels": labels,
                     "attention_mask": [1] * len(full_ids)})
    return Dataset.from_list(rows)


def collate(batch, pad_id: int):
    width = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = width - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--force", action="store_true", help="re-run even if already logged")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    rid = run_id(cfg)
    if rid in completed_ids() and not args.force:
        print(f"skip {rid}: already in results/runs.jsonl")
        return

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    local_idx, sel_cost, sel_class = load_selection(cfg)
    pool = load_train_examples()
    train_examples = [pool[i] for i in local_idx]
    print(f"run {rid}: {cfg['selection_method']} @ {cfg['ratio']} "
          f"-> {len(train_examples)} examples, lr={cfg['learning_rate']}, seed={cfg['seed']}")

    tok, model = build_model(cfg)
    ds = tokenize(train_examples, tok, cfg["max_seq_len"])

    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=lambda b: collate(b, tok.pad_token_id))
    steps_per_epoch = max(1, len(loader) // cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0.0),
    )
    sched = get_cosine_schedule_with_warmup(
        opt, int(cfg.get("warmup_ratio", 0.03) * total_steps), total_steps
    )
    scaler = torch.cuda.amp.GradScaler()

    train_cost = CostRecord(stage="training", device=torch.cuda.get_device_name(0))
    t0 = time.perf_counter()
    trained_tokens = 0
    model.train()

    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = model(**batch).loss / cfg["grad_accum"]
            scaler.scale(loss).backward()
            trained_tokens += int(batch["attention_mask"].sum())
            if (step + 1) % cfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.get("max_grad_norm", 1.0),
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
        print(f"  epoch {epoch + 1}/{cfg['epochs']} done")

    train_cost.wall_clock_s = time.perf_counter() - t0
    train_cost.flops = train_flops(cfg["target_model"], trained_tokens, epochs=1, lora=True)
    train_cost.notes = {"trained_tokens": trained_tokens, "total_steps": total_steps,
                        "n_examples": len(ds)}

    from evaluate import evaluate_all

    art_dir = RUN_ARTIFACTS / rid
    art_dir.mkdir(parents=True, exist_ok=True)
    metrics, eval_cost = evaluate_all(model, tok, cfg, art_dir)

    sel_record = CostRecord(**sel_cost) if sel_cost else None
    append_run({
        "run_id": rid,
        "status": "ok",
        "study": cfg["study"],
        "config": cfg,
        "metrics": metrics,
        "cost": {
            **total_cost(sel_record, train_cost),
            "evaluation_wall_clock_s": eval_cost.wall_clock_s,
            "selection_cost_class": sel_class,
        },
        "n_train_examples": len(ds),
    })
    print(f"logged {rid}: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
