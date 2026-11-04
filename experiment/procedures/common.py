"""Shared helpers for A_rag / A_singleagent / A_subagents on RULER / HotPotQA / NoLiMa.

Each procedure has its own NIAH-style runner already. The dataset-agnostic logic
lives here: convert any episode into (haystack_text, question, gold), then
dispatch to the procedure's main runner. Scoring uses the right scorer per
dataset.

This keeps procedure files small and consistent.
"""
from __future__ import annotations

from typing import Optional

from bench.utils import token_f1


def get_dataset_inputs(episode) -> tuple[str, str, object]:
    """Extract (haystack_text, question, gold) from any episode dataclass.

    Gold may be a string (HotPotQA, BABILong, Banking77, NIAH-value), a list of
    strings (RULER), or a needle sentence (NIAH-full). The scorer dispatch
    below normalizes each case to its proper metric.
    """
    haystack = getattr(episode, "haystack_text", None) or getattr(episode, "context", "")
    question = getattr(episode, "question", "")
    # Gold candidates in priority order.
    # NIAH: use `value` directly (the 7-digit number), not `needle` (full sentence).
    # The scorer matches `value` substring in response.
    if hasattr(episode, "value") and type(episode).__name__ == "NiahEpisode":
        return haystack, question, getattr(episode, "value")
    for attr in ("needle", "gold_answer"):
        if hasattr(episode, attr):
            return haystack, question, getattr(episode, attr)
    raise ValueError(f"Episode {type(episode).__name__} has no needle/gold_answer attribute")


def score_response(response: str, gold, episode) -> tuple[float, str]:
    """Pick the right scorer per dataset. Returns (metric_value, metric_name)."""
    type_name = type(episode).__name__
    if type_name == "NiahEpisode":
        from bench.niah import scorer

        return scorer(response, gold), "exact_match"
    if type_name == "NoLiMaEpisode":
        from bench.nolima import scorer

        return scorer(response, gold), "exact_match"
    if type_name == "RulerEpisode":
        from bench.ruler import scorer

        return scorer(response, gold, episode.task_subtype), (
            "exact_match" if episode.task_subtype == "multikey" else "partial_match"
        )
    if type_name == "HotPotQAEpisode":
        return token_f1(response, gold), "f1"
    if type_name == "BabilongEpisode":
        from bench.babilong import scorer

        return scorer(response, gold), "exact_match"
    if type_name == "Banking77IclEpisode":
        from bench.banking77_icl import scorer

        return scorer(response, gold), "exact_match"
    # Fallback — fuzzy substring
    if isinstance(gold, str):
        return (1.0 if gold and gold in response else 0.0), "exact_match"
    return 0.0, "unknown"
