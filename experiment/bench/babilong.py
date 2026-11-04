"""BABILong loader — multi-hop reasoning task.

Source: https://huggingface.co/datasets/RMT-team/babilong
Paper: Kuratov et al. 2024 (NeurIPS) — bAbI facts scattered in PG-19 prose at
controlled token-budget lengths.

We use qa2 (Two Supporting Facts):
  input  = long PG-19 narrative with bAbI facts about people moving/picking objects
  question = "Where is the football?" (or similar location/possession query)
  target = short location/object name (e.g. "kitchen")

Why qa2 specifically (and not qa1/qa3/qa5):
  - qa1 (single fact) is too close to NIAH retrieval — duplicates that axis
  - qa3 (three facts) collapses to 0% accuracy at 16K for most models, making
    f(N) curves uninformative
  - qa5 (relations) tests a different aggregation pattern but has narrower
    answer distribution
  - qa2 is the canonical multi-hop primitive: two related facts must be
    retrieved AND combined

Length-config mapping (verified empirically against the actual dataset):
  config "16k"   → ~10K words p50, fits our 12K tier
  config "32k"   → ~20K words p50, fits our 24K tier
  config "64k"   → ~44K words p50, fits our 64K tier
  config "128k"  → ~89K words p50, fits any >64K tier (not used in v6)

For 6K and 32K (our exact ladders), we pick the closest BABILong config and
filter by actual word count.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class BabilongEpisode:
    """Mirrors NiahEpisode/HotPotQAEpisode shape so orchestrator+procedures work."""
    prompt: str          # haystack + question + "Answer:" — for legacy A0 path
    haystack_text: str   # just the input (no question appended) — for RAG/ReAct/MemGPT
    question: str
    gold_answer: str     # the single target word
    target_words: int    # config-level requested length
    actual_words: int    # actual word count of haystack
    seed: int
    episode_idx: int
    haystack_hash: str
    bin_idx: Optional[int] = None  # not used for BABILong (no fixed Kamradt depths)
    extra: dict = None


# Map our target_words ladder to BABILong's token-config bucket.
# BABILong configs are nominal token sizes; actual word counts vary.
#
# v6.2 (Codex P0.4): bump short tier to 9000 because at 6000 the 16k config
# has only ~2-12 examples within tolerance (BABILong shortest example is ~8K
# words). At 9000 the same 16k config gives pool=95 with full diversity for
# n=30. The 6000 mapping is retained for backward compat with main run data
# but should not be used for new runs.
_TARGET_TO_CONFIG = {
    6000:  "16k",   # LEGACY: was used in main run; gives pool=2, n=20 effective n=2
    9000:  "16k",   # NEW v6.2: gives pool=95, n=30 OK
    12000: "16k",
    24000: "32k",
    32000: "32k",
    64000: "64k",
    128000: "128k",
}


@lru_cache(maxsize=8)
def _load_qa(config: str, qa_task: str = "qa2"):
    """Cached HF download. Returns list of {input, question, target} dicts."""
    from datasets import load_dataset
    ds = load_dataset("RMT-team/babilong", config, split=qa_task)
    return [{"input": ex["input"], "question": ex["question"], "target": ex["target"]}
            for ex in ds]


def _filter_by_length(records: list[dict], target_words: int, tol: float = 0.30,
                       min_pool: int = 10) -> list[dict]:
    """Pick records whose input word count is within ±tol of target.

    v6.2: widen tolerance progressively if pool is too small (min_pool=10).
    Previous version with fixed tol=0.30 gave pool=2 at target=6000 (16k
    config has p10=10K words, so only the shortest 2 examples are within
    ±30% of 6000). With auto-widening, pool grows to 20+ at the cost of
    actual_words drifting further from target — analysis uses actual_words
    as the x-axis anyway, so this is fine.
    """
    pool = []
    cur_tol = tol
    while len(pool) < min_pool and cur_tol < 1.0:
        lo = int(target_words * (1 - cur_tol))
        hi = int(target_words * (1 + cur_tol))
        pool = [r for r in records if lo <= len(r["input"].split()) <= hi]
        if len(pool) >= min_pool:
            break
        cur_tol += 0.10
    # If even at 100% tolerance we have <min_pool, return whatever we have.
    return pool


def _multi_config_pool(target_words: int, tol: float = 0.30,
                        qa_task: str = "qa2", min_pool: int = 30):
    """Pool records from ALL configs and filter by target word count.

    Fix for v6.3: original code picked ONE config and filtered, giving pool=4 at
    target=4000 (only 4 of 100 16k records happen to be ~4000 words). Combining
    all configs (4k+8k+16k+32k) gives ~400 candidate records and richer pool.
    """
    all_records: list[dict] = []
    for cfg in ("4k", "8k", "16k", "32k"):
        try:
            all_records.extend(_load_qa(cfg, qa_task=qa_task))
        except Exception:
            pass
    return _filter_by_length(all_records, target_words, tol=tol, min_pool=min_pool)


def make_episode(seed: int, episode_idx: int, target_words: int) -> Optional[BabilongEpisode]:
    """Build a BABILong qa2 episode by sampling from a multi-config pool.

    Returns None if no records match the target length (caller should skip).
    """
    pool = _multi_config_pool(target_words, tol=0.30, qa_task="qa2", min_pool=30)
    if not pool:
        return None

    idx = (seed * 1_000_003 ^ episode_idx) % len(pool)
    rec = pool[idx]
    haystack = rec["input"]
    question = rec["question"].strip()
    target = rec["target"].strip()

    # Canonical BABILong prompt (Kuratov 2024, NeurIPS) — explicitly tells the
    # model that facts about character positions/actions are hidden in narrative
    # prose and that the answer must be based ONLY on those facts. Without this
    # framing, models tend to "answer the most recent sentence containing the
    # entity" instead of tracking state across multiple facts.
    prompt = (
        "I will give you context with the facts about positions and actions of "
        "different persons hidden among irrelevant text. I will then ask you a "
        "question and you need to give an answer based ONLY on the given facts. "
        "Output a single word — the location or object name.\n\n"
        f"Context:\n{haystack}\n\n"
        f"Question: {question}\nAnswer:"
    )

    return BabilongEpisode(
        prompt=prompt,
        haystack_text=haystack,
        question=question,
        gold_answer=target,
        target_words=target_words,
        actual_words=len(haystack.split()),
        seed=seed,
        episode_idx=episode_idx,
        haystack_hash=hashlib.sha256(haystack.encode("utf-8")).hexdigest()[:16],
        bin_idx=None,
        extra={"babilong_config": "multi", "qa_task": "qa2", "n_pool": len(pool)},
    )


def scorer(response: str, gold: str) -> float:
    """Normalized substring match: 1.0 if gold word appears in response.

    bAbI answers are single tokens (location/object name) like 'kitchen',
    'football'. We normalize to lowercase and check substring presence.
    """
    if not response or not gold:
        return 0.0
    return 1.0 if gold.strip().lower() in response.strip().lower() else 0.0
