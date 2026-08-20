"""The six selection methods compared in the paper.

Cost classes (the axis the RQ2 argument rides on):
  free           - no model forward passes at all
  training-free  - forward passes only
  training-based - requires gradient updates before selection can happen
"""
from __future__ import annotations

import heapq
import time

import numpy as np
import torch

from cost import CostRecord, forward_flops, train_flops
from data import build_full
from selection.base import ScoreSelector, Selector, register, scores_with_cache
from selection.scoring import load_scorer, response_nll


@register("random")
class RandomSelector(Selector):
    cost_class = "free"

    def select(self, examples, k, seed):
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(examples), size=k, replace=False))
        return idx, CostRecord(stage="selection", flops=0.0, device="none")


@register("perplexity")
class PerplexitySelector(ScoreSelector):
    """Response perplexity under a small pretrained scorer.

    direction="high" keeps the examples the scorer finds hardest. This is the
    standard reading of "informative", but it is a hypothesis, not a fact:
    high perplexity also selects noise, which the qualitative section inspects.
    """

    cost_class = "training-free"
    direction = "high"

    def __init__(self, scorer="HuggingFaceTB/SmolLM-135M", device="cpu", max_len=512):
        self.scorer, self.device, self.max_len = scorer, device, max_len

    def score(self, examples):
        cost = CostRecord(stage="selection", device=self.device)
        tok, model = load_scorer(self.scorer, self.device)
        t0 = time.perf_counter()
        nll, ntok = response_nll(
            examples, tok, model, device=self.device, max_len=self.max_len,
            condition_on_prompt=True, desc="perplexity",
        )
        cost.wall_clock_s = time.perf_counter() - t0
        cost.flops = forward_flops(self.scorer, ntok)
        cost.notes = {"scorer": self.scorer, "tokens_processed": ntok, "passes": 1}
        return nll, cost

    def config(self):
        return {**super().config(), "scorer": self.scorer, "max_len": self.max_len}


@register("ifd")
class IFDSelector(ScoreSelector):
    """Instruction-Following Difficulty: PPL(response|instruction) / PPL(response).

    NOTE - deviation from Cherry-LLM (Li et al.): the original computes IFD with a
    "brief experience" model first fine-tuned on ~1% of the pool, which makes it
    training-based. This implementation uses an off-the-shelf pretrained scorer,
    keeping it training-free. That is a deliberate change and must be stated in the
    paper; the two variants are not interchangeable, and the cost-class difference
    is precisely what RQ2 is measuring.

    Ratio < 1 means the instruction HELPED, so the example is easy; Cherry-LLM
    discards those. We keep the filter and report how many are dropped.

    SCORER CHOICE: the same 135M model that the perplexity selector uses, so the
    difference between the two methods is the SIGNAL -- conditional versus
    unconditional loss -- and nothing else. Scoring IFD with a larger model would
    confound the signal with scorer capacity, and a win could not be attributed to
    either. It is also ~4x cheaper on CPU, but that is the smaller reason.
    """

    cost_class = "training-free"
    direction = "high"

    def __init__(self, scorer="HuggingFaceTB/SmolLM-135M", device="cpu", max_len=512,
                 filter_below_one=True):
        self.scorer, self.device, self.max_len = scorer, device, max_len
        self.filter_below_one = filter_below_one

    def score(self, examples):
        cost = CostRecord(stage="selection", device=self.device)
        tok, model = load_scorer(self.scorer, self.device)
        t0 = time.perf_counter()
        cond, tok_c = response_nll(
            examples, tok, model, device=self.device, max_len=self.max_len,
            condition_on_prompt=True, desc="ifd/conditional",
        )
        uncond, tok_u = response_nll(
            examples, tok, model, device=self.device, max_len=self.max_len,
            condition_on_prompt=False, desc="ifd/unconditional",
        )
        cost.wall_clock_s = time.perf_counter() - t0
        cost.flops = forward_flops(self.scorer, tok_c + tok_u)

        # Ratio of mean NLLs equals ratio of perplexities; computed in log space.
        with np.errstate(divide="ignore", invalid="ignore"):
            ifd = np.exp(cond - uncond)
        n_filtered = 0
        if self.filter_below_one:
            keep = np.isfinite(ifd) & (ifd >= 1.0)
            n_filtered = int(len(ifd) - keep.sum())
            ifd = np.where(keep, ifd, -np.inf)  # drop the "instruction helped" cases
        cost.notes = {
            "scorer": self.scorer,
            "tokens_processed": tok_c + tok_u,
            "passes": 2,
            "n_dropped_ifd_lt_1": n_filtered,
            "deviation": "pretrained scorer, not Cherry-LLM brief-experience model",
        }
        return ifd, cost

    def config(self):
        return {
            **super().config(),
            "scorer": self.scorer,
            "max_len": self.max_len,
            "filter_below_one": self.filter_below_one,
        }


@register("diversity")
class DiversitySelector(Selector):
    """Facility-location maximisation over sentence embeddings (lazy greedy).

    Facility location f(S) = sum_i max_{j in S} sim(i, j) is monotone submodular,
    so greedy carries a (1 - 1/e) guarantee. Lazy greedy is exact, just faster.
    """

    cost_class = "training-free"

    def __init__(self, embedder="sentence-transformers/all-MiniLM-L6-v2", device="cpu",
                 batch_size=64, max_pool=20_000):
        self.embedder, self.device, self.batch_size = embedder, device, batch_size
        self.max_pool = max_pool

    def embed(self, examples):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.embedder, device=self.device)
        texts = [build_full(ex) for ex in examples]
        emb = model.encode(
            texts, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        )
        # Approximate token count for the FLOP ledger; MiniLM truncates at 256 pieces.
        approx_tokens = sum(min(256, len(t.split()) * 4 // 3) for t in texts)
        return emb.astype(np.float32), approx_tokens

    def select(self, examples, k, seed, prior=None):
        n = len(examples)
        if n > self.max_pool:
            raise ValueError(
                f"Pool of {n} exceeds max_pool={self.max_pool}; an {n}x{n} similarity "
                "matrix would not fit in laptop RAM. Raise max_pool deliberately."
            )
        cost = CostRecord(stage="selection", device=self.device)
        t0 = time.perf_counter()
        emb, approx_tokens = self.embed(examples)
        sim = emb @ emb.T  # cosine, since the embeddings are normalised
        idx = lazy_greedy_facility_location(sim, k, prior=prior)
        cost.wall_clock_s = time.perf_counter() - t0
        cost.flops = forward_flops(self.embedder, approx_tokens)
        cost.notes = {
            "embedder": self.embedder,
            "approx_tokens": approx_tokens,
            "objective": "facility_location" + ("_weighted" if prior is not None else ""),
        }
        return np.sort(idx), cost

    def config(self):
        return {**super().config(), "embedder": self.embedder}


@register("hybrid")
class HybridSelector(DiversitySelector):
    """Perplexity-weighted facility location: quality signal x coverage signal.

    Weighting the facility-location objective by a per-example quality prior is the
    cheapest way to combine the two training-free signals, and it keeps the whole
    method inside the training-free cost class.
    """

    cost_class = "training-free"

    def __init__(self, scorer="HuggingFaceTB/SmolLM-135M", **kwargs):
        super().__init__(**kwargs)
        self.scorer = scorer

    def select(self, examples, k, seed, prior=None):
        ppl = PerplexitySelector(scorer=self.scorer, device=self.device)
        # Through the cache: if plain perplexity already ran with this scorer, its
        # scores are on disk and this is instant instead of another full pass.
        scores, ppl_cost = scores_with_cache(ppl, examples)
        # Rank-normalise to [0, 1] so the prior is scale-free and outlier-robust.
        ranks = np.zeros(len(scores), dtype=np.float32)
        order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf), kind="stable")
        ranks[order] = np.linspace(0.0, 1.0, len(scores), dtype=np.float32)
        weights = 0.5 + ranks  # in [0.5, 1.5]: never zero an example out entirely

        idx, div_cost = super().select(examples, k, seed, prior=weights)
        div_cost.flops += ppl_cost.flops
        div_cost.wall_clock_s += ppl_cost.wall_clock_s
        div_cost.notes = {**div_cost.notes, "scorer": self.scorer, "prior": "perplexity_rank"}
        return idx, div_cost

    def config(self):
        return {**super().config(), "scorer": self.scorer}


@register("learning_percentage")
class LearningPercentageSelector(ScoreSelector):
    """Zhang et al. (arXiv:2402.10430): how much does one proxy epoch reduce each
    example's loss?  LP_i = (loss_before_i - loss_after_i) / loss_before_i.

    This is the only training-based method here, and that is the entire point: it
    needs a full proxy epoch before a single target step can run. On a laptop CPU
    that epoch is impractical, so this selector runs on the Kaggle GPU while the
    training-free selectors run on CPU. Wall-clock is therefore NOT comparable
    across cost classes: use analytical FLOPs for cross-method cost claims and
    report wall-clock only within a device. State this explicitly in the paper.

    direction="low" keeps the examples the proxy learned LEAST from in one epoch,
    i.e. the hard ones, following the source paper.
    """

    cost_class = "training-based"
    direction = "low"

    def __init__(self, proxy="HuggingFaceTB/SmolLM-135M", device="cuda", max_len=512,
                 epochs=1, lr=2e-5, batch_size=8):
        self.proxy, self.device, self.max_len = proxy, device, max_len
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size

    def score(self, examples):
        cost = CostRecord(stage="selection", device=self.device)
        tok, model = load_scorer(self.proxy, self.device)
        t0 = time.perf_counter()

        before, ntok = response_nll(
            examples, tok, model, device=self.device, max_len=self.max_len, desc="lp/before"
        )
        trained_tokens = self._train_one_epoch(examples, tok, model)
        after, _ = response_nll(
            examples, tok, model, device=self.device, max_len=self.max_len, desc="lp/after"
        )
        cost.wall_clock_s = time.perf_counter() - t0

        cost.flops = (
            forward_flops(self.proxy, 2 * ntok)
            + train_flops(self.proxy, trained_tokens, epochs=self.epochs, lora=False)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = (before - after) / before
        cost.notes = {
            "proxy": self.proxy,
            "epochs": self.epochs,
            "lr": self.lr,
            "scoring_tokens": 2 * ntok,
            "training_tokens": trained_tokens,
            "passes": "2 forward + 1 training epoch",
        }
        return lp, cost

    def _train_one_epoch(self, examples, tok, model) -> int:
        """Plain full fine-tune of the proxy. Deliberately not LoRA: the source
        method trains the proxy outright, and swapping in LoRA here would understate
        the very cost RQ2 sets out to expose."""
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr)
        texts = [build_full(ex, eos=tok.eos_token or "") for ex in examples]
        order = np.random.RandomState(0).permutation(len(texts))
        total_tokens = 0

        for _ in range(self.epochs):
            for start in range(0, len(order), self.batch_size):
                batch = [texts[i] for i in order[start : start + self.batch_size]]
                enc = tok(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_len,
                ).to(self.device)
                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100
                loss = model(**enc, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                total_tokens += int(enc["attention_mask"].sum())

        model.eval()
        return total_tokens

    def config(self):
        return {
            **super().config(),
            "proxy": self.proxy,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
        }


def lazy_greedy_facility_location(sim: np.ndarray, k: int, prior=None) -> np.ndarray:
    """Exact greedy for facility location, accelerated by the lazy-evaluation trick.

    Marginal gains are non-increasing as the set grows (submodularity), so a stale
    gain that still tops the heap after recomputation is provably the true argmax.
    That turns O(nk) gain evaluations into roughly O(n log n + k log n).
    """
    n = sim.shape[0]
    if k >= n:
        return np.arange(n)
    w = np.ones(n, dtype=np.float32) if prior is None else prior.astype(np.float32)

    # Shift similarities to be non-negative before running greedy. Cosine lives in
    # [-1, 1], and leaving it there breaks the objective: with cur_max initialised
    # to 0, a candidate from a cluster that is dissimilar to everything selected so
    # far has sim < 0 < cur_max and scores ZERO marginal gain. Greedy then keeps
    # drawing from the first cluster it touched -- the exact opposite of diversity,
    # and a failure that produces plausible-looking output rather than an error.
    # An affine shift preserves the argmax ordering and keeps f monotone submodular.
    sim = sim - sim.min()

    cur_max = np.zeros(n, dtype=np.float32)
    # Gain of the singleton {j} is sum_i w_i * sim(i, j), since cur_max starts at zero.
    heap = [(-float(np.dot(w, sim[:, j])), j) for j in range(n)]
    heapq.heapify(heap)

    selected: list[int] = []
    while len(selected) < k:
        _stale, j = heapq.heappop(heap)
        gain = float(np.dot(w, np.maximum(sim[:, j] - cur_max, 0.0)))
        if not heap or gain >= -heap[0][0] - 1e-9:
            selected.append(j)
            np.maximum(cur_max, sim[:, j], out=cur_max)
        else:
            heapq.heappush(heap, (-gain, j))
    return np.array(selected)
