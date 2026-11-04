"""Bench-side utilities shared by NIAH / RULER / LongBench / NoLiMa loaders."""
from __future__ import annotations

import re
import string


_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """Standard QA normalization: lowercase, drop articles, drop punctuation, collapse whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLE_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 (SQuAD-style) over normalized strings.

    For HotPotQA-style short golds, we score against the FIRST LINE of pred to
    avoid penalizing verbose continuations. This matches LongBench's official
    behavior on chat-model outputs.
    """
    # Take only first non-empty line — matches LongBench scorer convention
    first_line = (pred or "").strip().split("\n", 1)[0]
    pred_norm = normalize_answer(first_line).split()
    gold_norm = normalize_answer(gold).split()
    if not pred_norm and not gold_norm:
        return 1.0
    if not pred_norm or not gold_norm:
        return 0.0
    common: dict[str, int] = {}
    for tok in pred_norm:
        common[tok] = min(pred_norm.count(tok), gold_norm.count(tok))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_norm)
    recall = n_same / len(gold_norm)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def truncate_middle(text: str, max_words: int, head_frac: float = 0.5) -> str:
    """Truncate by words: keep head_frac at start, rest at tail. Inserts '... [TRUNCATED] ...'."""
    words = text.split()
    if len(words) <= max_words:
        return text
    n_head = int(max_words * head_frac)
    n_tail = max_words - n_head
    return " ".join(words[:n_head]) + " ... [TRUNCATED] ... " + " ".join(words[-n_tail:])
