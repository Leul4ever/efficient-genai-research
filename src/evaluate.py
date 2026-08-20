"""Stage 2b: evaluation. Frozen in Week 1, never changed afterwards.

The critical design point is that held-out loss is saved PER EXAMPLE, not as a
mean. Paired bootstrap needs the per-example vector: pairing on examples removes
example-difficulty variance and is far more sensitive than comparing two means
with independent CIs. Storing only the mean throws that power away irreversibly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cost import CostRecord
from data import build_prompt, load_held_out_examples


@torch.no_grad()
def held_out_loss(model, tok, cfg) -> tuple[dict, np.ndarray]:
    """Mean response NLL on the frozen 1,000-example split.

    Note for the paper: this metric rewards matching Dolly's response style, so a
    selection method that happens to pick Dolly-typical examples is flattered by it.
    Keep it as the cheap, always-available primary, but let IFEval and the pairwise
    judgement carry the headline claims.
    """
    examples = load_held_out_examples()
    model.eval()
    per_example = np.full(len(examples), np.nan, dtype=np.float64)

    for i, ex in enumerate(tqdm(examples, desc="held-out loss", leave=False)):
        response = ex["response"].strip()
        if not response:
            continue
        prompt_ids = tok(build_prompt(ex), add_special_tokens=True)["input_ids"]
        resp_ids = tok(response, add_special_tokens=False)["input_ids"]
        budget = cfg["max_seq_len"] - len(resp_ids)
        if budget < 1:
            resp_ids = resp_ids[: cfg["max_seq_len"] - 1]
            budget = 1
        prompt_ids = prompt_ids[-budget:]

        ids = torch.tensor([prompt_ids + resp_ids]).cuda()
        logits = model(input_ids=ids).logits[0, :-1, :].float()
        labels = ids[0, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[len(prompt_ids) - 1 :] = True
        if mask.sum() == 0:
            continue
        per_example[i] = float(torch.nn.functional.cross_entropy(
            logits[mask], labels[mask], reduction="mean"
        ))

    valid = np.isfinite(per_example)
    return {
        "held_out_loss": float(per_example[valid].mean()),
        "held_out_ppl": float(np.exp(per_example[valid].mean())),
        "held_out_n_valid": int(valid.sum()),
    }, per_example


@torch.no_grad()
def generate_responses(model, tok, cfg, n: int) -> list[dict]:
    """Greedy generations on the first n held-out prompts, for the pairwise judge.

    Greedy, not sampled: sampling adds a variance source that would have to be
    averaged over, and the compute budget does not stretch to that.
    """
    examples = load_held_out_examples()[:n]
    model.eval()
    out = []
    for ex in tqdm(examples, desc="generate", leave=False):
        ids = tok(build_prompt(ex), return_tensors="pt", truncation=True,
                  max_length=cfg["max_seq_len"]).to("cuda")
        gen = model.generate(
            **ids, max_new_tokens=cfg.get("gen_max_new_tokens", 256),
            do_sample=False, pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"instruction": ex["instruction"], "context": ex.get("context", ""),
                    "reference": ex["response"], "generation": text.strip()})
    return out


def lm_eval_harness(model, tok, cfg) -> dict:
    """IFEval / ARC-Easy / HellaSwag through lm-eval-harness.

    Wrapped in a try/except on purpose: the harness is the most fragile dependency
    in the stack, and a harness failure must not destroy an otherwise-good training
    run. A run whose harness metrics are missing is still a usable data point for
    held-out loss.
    """
    tasks = cfg.get("harness_tasks") or []
    if not tasks:
        return {}
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        results = lm_eval.simple_evaluate(
            model=HFLM(pretrained=model, tokenizer=tok, batch_size=cfg.get("eval_batch_size", 8)),
            tasks=tasks,
            limit=cfg.get("harness_limit"),
            bootstrap_iters=0,  # we do our own bootstrap; the harness's is redundant
        )["results"]
        flat = {}
        for task, scores in results.items():
            for metric, value in scores.items():
                if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
                    flat[f"{task}.{metric.split(',')[0]}"] = float(value)
        return flat
    except Exception as exc:  # noqa: BLE001
        print(f"WARN: lm-eval failed ({type(exc).__name__}: {exc}); continuing without it")
        return {"harness_error": str(exc)[:200]}


def evaluate_all(model, tok, cfg, artifact_dir: Path) -> tuple[dict, CostRecord]:
    cost = CostRecord(stage="evaluation", device="cuda")
    t0 = time.perf_counter()

    metrics, per_example = held_out_loss(model, tok, cfg)
    np.save(artifact_dir / "held_out_per_example_loss.npy", per_example)

    metrics.update(lm_eval_harness(model, tok, cfg))

    n_gen = cfg.get("n_judge_prompts", 0)
    if n_gen:
        gens = generate_responses(model, tok, cfg, n_gen)
        (artifact_dir / "generations.json").write_text(json.dumps(gens, indent=2))
        metrics["n_generations"] = len(gens)

    cost.wall_clock_s = time.perf_counter() - t0
    return metrics, cost
