"""Shared forward-pass machinery for the training-free scorers.

All of this runs on a laptop CPU. It is slow but bounded: 14k examples through a
135M model at seq 512 is minutes, not hours, and that measured wall-clock IS a
result (see cost.py).
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from data import build_prompt


def load_scorer(model_id: str, device: str = "cpu"):
    # Imported lazily so the pure-numpy parts of this package (the submodular
    # solver, the statistics) stay importable on a machine without transformers.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def response_nll(
    examples: list[dict],
    tok,
    model,
    *,
    device: str = "cpu",
    max_len: int = 512,
    condition_on_prompt: bool = True,
    desc: str = "scoring",
) -> tuple[np.ndarray, int]:
    """Mean negative log-likelihood of each response token.

    condition_on_prompt=True  -> NLL(response | instruction)   [conditional]
    condition_on_prompt=False -> NLL(response)                 [unconditional]

    The pair is exactly what IFD needs. Returns (nll_per_example, total_tokens_processed)
    where the token count feeds the FLOP ledger.

    Batch size is 1 on purpose: padding wastes CPU FLOPs and would corrupt the
    measured cost, which is a primary result here rather than an implementation detail.
    """
    out = np.empty(len(examples), dtype=np.float64)
    total_tokens = 0

    for i, ex in enumerate(tqdm(examples, desc=desc, leave=False)):
        response = ex["response"].strip()
        if not response:
            out[i] = np.nan
            continue

        if condition_on_prompt:
            prefix_ids = tok(build_prompt(ex), add_special_tokens=True)["input_ids"]
        else:
            prefix_ids = tok("", add_special_tokens=True)["input_ids"]
        resp_ids = tok(response, add_special_tokens=False)["input_ids"]

        # Truncate the PREFIX from the left, never the response: the quantity being
        # measured is the response's likelihood, so response tokens must survive.
        budget = max_len - len(resp_ids)
        if budget < 1:
            resp_ids = resp_ids[: max_len - 1]
            budget = 1
        prefix_ids = prefix_ids[-budget:]

        input_ids = torch.tensor([prefix_ids + resp_ids], device=device)
        labels = input_ids.clone()
        labels[0, : len(prefix_ids)] = -100  # score the response only

        logits = model(input_ids=input_ids).logits
        # Standard causal shift: logits at position t predict token t+1.
        shift_logits = logits[0, :-1, :]
        shift_labels = labels[0, 1:]
        mask = shift_labels != -100
        if mask.sum() == 0:
            out[i] = np.nan
            continue
        loss = torch.nn.functional.cross_entropy(
            shift_logits[mask].float(), shift_labels[mask], reduction="mean"
        )
        out[i] = float(loss)
        total_tokens += int(input_ids.numel())

    return out, total_tokens
