"""BANKING77 Many-shot In-Context Learning loader — ICL task.

Source: https://huggingface.co/datasets/banking77 (Casanueva et al. 2020)
  10003 train + 3080 test, 77 intent classes for banking customer-support queries.

Why ICL is the right 3rd axis (per HELMET, Yen et al. 2024 ICLR):
  - HELMET shows ICL has the LOWEST correlation with retrieval, multi-hop, RAG,
    summarization tasks among all long-context categories tested.
  - It directly probes a different capability: can the model USE thousands of
    in-context examples to learn label semantics?
  - Length is perfectly controllable (just add more demonstrations).
  - Output is a single label name → automatic exact-match scoring with strong
    signal (random baseline = 1/77 ≈ 1.3%).

Episode structure:
  - 1 test query (held out from train pool, deterministic by seed+episode_idx)
  - K stratified few-shot demonstrations from train (balanced across 77 classes
    where possible, then shuffled deterministically)
  - K chosen so that resulting prompt is ≈ target_words

System + user prompts follow the canonical many-shot ICL template (instruction
+ demonstrations + final query) used in HELMET ICL evaluations.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


@dataclass
class Banking77IclEpisode:
    """Mirrors NiahEpisode shape — orchestrator dispatches on type name."""
    prompt: str            # full prompt including all demos + test query
    haystack_text: str     # just the demonstrations block (no instruction/test) — for RAG/ReAct/MemGPT
    question: str          # the test query text only
    gold_answer: str       # label name (e.g. 'card_arrival')
    target_words: int
    actual_words: int
    seed: int
    episode_idx: int
    haystack_hash: str
    bin_idx: Optional[int] = None
    extra: dict = None


# Empirical avg from actual prompt building (2026-05-16 calibration):
# 12 words (BANKING77 avg query) + 4 words (Label/marker) + newlines = ~14 words.
# Tested: with this constant, target=6K hits ~6K actual words within ±3%.
_WORDS_PER_DEMO = 14


@lru_cache(maxsize=1)
def _load_banking77():
    """Load BANKING77 train + test from HF. Returns (train_records, test_records, label_names)."""
    from datasets import load_dataset
    ds_train = load_dataset("banking77", split="train")
    ds_test = load_dataset("banking77", split="test")
    label_names = ds_train.features["label"].names  # 77 names

    train = [{"text": ex["text"].strip(), "label": int(ex["label"]),
              "label_name": label_names[int(ex["label"])]} for ex in ds_train]
    test = [{"text": ex["text"].strip(), "label": int(ex["label"]),
             "label_name": label_names[int(ex["label"])]} for ex in ds_test]
    return train, test, label_names


def _select_demos(
    train_pool: list[dict],
    n_demos: int,
    rng: random.Random,
    label_names: list[str],
) -> list[dict]:
    """Stratified selection: ~n_demos/77 per class, then shuffle.

    If n_demos < 77 we take 1 per class for the first 77 (no class missed),
    then random fill. If n_demos > train pool, we just take min available.
    """
    by_label: dict[int, list[dict]] = {}
    for rec in train_pool:
        by_label.setdefault(rec["label"], []).append(rec)
    # Shuffle each class bucket deterministically
    for lab in by_label:
        rng.shuffle(by_label[lab])

    selected: list[dict] = []
    # Round-robin draw across classes until we have n_demos or pool exhausted
    pos_by_label = {lab: 0 for lab in by_label}
    n_classes = len(by_label)
    label_keys = sorted(by_label.keys())
    while len(selected) < n_demos:
        progressed = False
        for lab in label_keys:
            if pos_by_label[lab] < len(by_label[lab]):
                selected.append(by_label[lab][pos_by_label[lab]])
                pos_by_label[lab] += 1
                progressed = True
                if len(selected) >= n_demos:
                    break
        if not progressed:
            break  # pool exhausted

    rng.shuffle(selected)
    return selected[:n_demos]


def _format_prompt(demos: list[dict], test_query: str, label_names: list[str]) -> tuple[str, str]:
    """Build (haystack_block, full_prompt). haystack_block = demos only."""
    demo_lines = []
    for d in demos:
        demo_lines.append(f"Query: {d['text']}\nLabel: {d['label_name']}")
    haystack = "\n\n".join(demo_lines)

    # We show the label vocabulary explicitly so the model knows the legal set.
    # This is important at small K where coverage of all 77 classes is partial.
    labels_listed = ", ".join(label_names)

    full_prompt = (
        "You are a banking customer-support intent classifier. Below are query/label "
        "examples. After them, classify the FINAL query. Reply with ONLY the label "
        "name, exactly as shown in the examples (lowercase with underscores, e.g. "
        "card_arrival). Do not add any other text.\n\n"
        f"Legal labels (77 total): {labels_listed}\n\n"
        f"Examples:\n{haystack}\n\n"
        f"Now classify:\nQuery: {test_query}\nLabel:"
    )
    return haystack, full_prompt


def make_episode(seed: int, episode_idx: int, target_words: int) -> Banking77IclEpisode:
    """Build a many-shot ICL episode at approximately target_words length."""
    train, test, label_names = _load_banking77()

    # Per-episode RNG — deterministic, no cross-episode contamination.
    # IMPORTANT: use proper RNG via Random() for test_idx selection.
    # Previous version used `(seed * 7919 ^ episode_idx) % len(test)` which
    # produced indices in a narrow contiguous window (e.g. 3020-3047 for
    # seed=42, ep_idx 0..19) because XOR with small ep_idx only toggles low
    # bits. Since BANKING77 test split happens to be grouped by label, this
    # gave us only 2 unique labels per 20 episodes (instead of 15-20).
    # Fix v6.2: use rng.randrange to get truly diverse indices.
    rng = random.Random((seed * 1_000_003) ^ episode_idx ^ 0xDEAD)
    # Use a SEPARATE RNG instance for test_idx so it doesn't consume RNG state
    # affecting demo selection — keeps demo sampling stable.
    test_rng = random.Random((seed * 2_654_435_761) ^ (episode_idx * 1_009) ^ 0xBEEF)
    test_idx = test_rng.randrange(len(test))
    test_rec = test[test_idx]

    # Number of demos to hit target_words. Subtract overhead of instruction + final query.
    overhead_words = 180  # instruction header + labels list + test query + format
    test_words = len(test_rec["text"].split()) + 5
    demo_budget = max(50, target_words - overhead_words - test_words)
    n_demos = demo_budget // _WORDS_PER_DEMO

    demos = _select_demos(train, n_demos, rng, label_names)

    haystack, prompt = _format_prompt(demos, test_rec["text"], label_names)
    actual_words = len(prompt.split())

    return Banking77IclEpisode(
        prompt=prompt,
        haystack_text=haystack,
        question=test_rec["text"],
        gold_answer=test_rec["label_name"],
        target_words=target_words,
        actual_words=actual_words,
        seed=seed,
        episode_idx=episode_idx,
        haystack_hash=hashlib.sha256(haystack.encode("utf-8")).hexdigest()[:16],
        bin_idx=None,
        extra={"n_demos": len(demos), "test_idx": test_idx,
               "test_label": test_rec["label_name"], "n_classes": 77},
    )


def scorer(response: str, gold_label_name: str) -> float:
    """Match the first non-empty token-like chunk in response against gold label.

    Models tend to reply either:
      - "card_arrival"
      - "Label: card_arrival"
      - "card_arrival\\nSome explanation..."
      - " The label is: card_arrival"

    Strategy: scan for an exact occurrence of the gold label name in the response
    (case-insensitive), bounded by non-alphanumeric or start/end. This is more
    permissive than strict-equal but rejects partial matches like 'card' for
    'card_arrival'.
    """
    if not response or not gold_label_name:
        return 0.0
    import re
    pat = r"(?<![a-zA-Z0-9_])" + re.escape(gold_label_name) + r"(?![a-zA-Z0-9_])"
    return 1.0 if re.search(pat, response, re.IGNORECASE) else 0.0
