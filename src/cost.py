"""Total-cost accounting: the ledger RQ2 rides on.

Two currencies, both reported:
  * wall-clock seconds, measured with a stopwatch on stated hardware;
  * analytical FLOPs, hardware-independent and therefore comparable.

Neither alone is honest. Wall-clock conflates method cost with hardware; FLOPs
ignore that selection runs on a CPU you already own while training rents a GPU.
Report both and let the reader pick.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

# Forward pass ~= 2 FLOPs per parameter per token (multiply-accumulate).
FWD_FLOPS_PER_PARAM_TOKEN = 2.0

# Full fine-tuning backward ~= 2x forward (activation grads + weight grads) -> 6 total.
# LoRA freezes the base weights, so weight-gradient computation for them disappears,
# but activation gradients must still flow through every frozen layer. That leaves
# ~2x forward for the backward pass -> 4 total. This is an approximation and is
# stated as such in the paper's Limitations section.
FULL_FT_TOTAL_MULTIPLIER = 6.0
LORA_TOTAL_MULTIPLIER = 4.0

# Model sizes in parameters. Used for analytical FLOPs only.
PARAM_COUNTS = {
    "HuggingFaceTB/SmolLM-135M": 135e6,
    "Qwen/Qwen2.5-0.5B": 494e6,
    "Qwen/Qwen2.5-1.5B": 1.54e9,
    "meta-llama/Llama-3.2-1B": 1.24e9,
    "sentence-transformers/all-MiniLM-L6-v2": 22.7e6,
}


def resolve_params(model_id: str) -> float:
    if model_id not in PARAM_COUNTS:
        raise KeyError(
            f"Unknown model {model_id!r}. Add its parameter count to cost.PARAM_COUNTS "
            "rather than guessing at analysis time."
        )
    return PARAM_COUNTS[model_id]


def forward_flops(model_id: str, n_tokens: int) -> float:
    return FWD_FLOPS_PER_PARAM_TOKEN * resolve_params(model_id) * n_tokens


def train_flops(model_id: str, n_tokens: int, epochs: int = 1, lora: bool = True) -> float:
    mult = LORA_TOTAL_MULTIPLIER if lora else FULL_FT_TOTAL_MULTIPLIER
    return mult * resolve_params(model_id) * n_tokens * epochs


@dataclass
class CostRecord:
    """Everything needed to place one method on the Pareto plot."""

    stage: str  # "selection" | "training" | "evaluation"
    wall_clock_s: float = 0.0
    flops: float = 0.0
    device: str = "unknown"
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@contextmanager
def stopwatch(record: CostRecord):
    """Wall-clock timing that survives exceptions."""
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record.wall_clock_s = time.perf_counter() - t0


def total_cost(selection: CostRecord | None, training: CostRecord) -> dict:
    """The number the paper argues about: selection is NOT free.

    Prior work reports `training` alone. RQ2 asks what happens to the ranking
    when the selection column is added back in.
    """
    sel_flops = selection.flops if selection else 0.0
    sel_wall = selection.wall_clock_s if selection else 0.0
    return {
        "selection_flops": sel_flops,
        "training_flops": training.flops,
        "total_flops": sel_flops + training.flops,
        "selection_wall_clock_s": sel_wall,
        "training_wall_clock_s": training.wall_clock_s,
        "total_wall_clock_s": sel_wall + training.wall_clock_s,
        "selection_share_of_total_flops": (
            sel_flops / (sel_flops + training.flops)
            if (sel_flops + training.flops) > 0
            else 0.0
        ),
    }
