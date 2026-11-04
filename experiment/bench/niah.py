"""NIAH (Needle-in-a-Haystack) — RULER niah-single template + Kamradt depths.

Implementation follows two published benchmarks:

  * RULER (Hsieh et al., COLM 2024) niah-single — random key/value pair embedded in
    a "noise" haystack of repeated filler sentences. The model must extract the
    value given the key.
  * Kamradt NIAH (2023) — 11 fixed insertion depths {0%, 10%, ..., 100%} aligned
    to sentence breaks.

Why these specific choices:
  - Random key+value: nothing memorizable in training data (true extractive test).
  - Noise haystack: model can't predict the value from semantics; must locate
    the literal needle sentence.
  - 11 Kamradt depths: published baseline; episode_idx → depth_idx = idx % 11
    distributes coverage evenly across positions (H6 anisotropy hypothesis).
  - Substring scoring on `value`: deterministic, cheap, exactly matches RULER's
    canonical scorer.
"""
from __future__ import annotations

import hashlib
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# RULER niah-single noise template — 5 short sentences, ~22 words total.
NOISE_TEMPLATE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)

# Word bank for random key generation. Avoid politically/culturally loaded words.
KEY_WORD_BANK = [
    "lemon", "grove", "thunder", "harbor", "cobalt", "saffron", "willow", "anchor",
    "marble", "cypress", "amber", "garnet", "ember", "frost", "horizon", "lantern",
    "meadow", "nimbus", "obsidian", "petal", "quartz", "river", "sapphire", "tundra",
    "umbra", "valley", "windrose", "xenon", "yarrow", "zephyr", "azure", "bronze",
    "coral", "dune", "echo", "flint", "glade", "haven", "ivy", "juniper",
]

N_POSITION_BINS = 11  # Kamradt depths {0%, 10%, ..., 100%}


@dataclass
class NiahEpisode:
    """RULER niah-single episode.

    Fields:
      prompt: haystack + question (the user-message content).
      haystack_text: just the haystack with the needle inserted, no question
        appended. Used by procedures that re-shard or search the document.
      question: the question text only.
      needle: full needle sentence (logging / debug).
      key: random key word (e.g. "thunder-saffron") — used as semantic anchor.
      value: random 7-digit number (the answer the model must produce).
      bin_idx: 0..10 — depth bucket (episode_idx % 11).
      depth_percent: 0.0..1.0 — actual insertion depth (= bin_idx * 0.1).
      position_char: char offset where the needle was inserted.
      haystack_hash: sha256[:16] for cache-key stability.
      target_words: target haystack length in words.
      seed: RNG seed.
      episode_idx: episode index.
    """
    prompt: str
    haystack_text: str
    question: str
    needle: str
    key: str
    value: str
    bin_idx: int
    depth_percent: float
    position_char: int
    haystack_hash: str
    target_words: int
    seed: int
    episode_idx: int


def _make_key(rng: random.Random) -> str:
    """Random 2-word hyphenated key, e.g. 'thunder-saffron'."""
    w1, w2 = rng.sample(KEY_WORD_BANK, 2)
    return f"{w1}-{w2}"


def _make_value(rng: random.Random) -> str:
    """Random 7-digit number string."""
    return "".join(rng.choices(string.digits, k=7))


def base_haystack(seed: int, target_words: int) -> str:
    """Build a noise haystack of approximately `target_words` words.

    Deterministic per (seed, target_words) — same inputs → same haystack, which
    enables llama-server prefix-cache reuse across all episodes of a cell.
    """
    # Number of noise blocks needed (one block ≈ 22 words).
    words_per_block = len(NOISE_TEMPLATE.split())
    n_blocks = (target_words + words_per_block - 1) // words_per_block
    text = NOISE_TEMPLATE * n_blocks
    # Trim words to exactly target_words.
    words = text.split()
    if len(words) > target_words:
        text = " ".join(words[:target_words])
    return text


def make_episode(
    seed: int,
    episode_idx: int,
    target_words: int,
    bin_idx: Optional[int] = None,
) -> NiahEpisode:
    """Create one RULER niah-single episode with Kamradt depth binning.

    Insertion is aligned to a sentence break (`. `) at or below the target
    char offset, matching Kamradt's approach to keep the needle in a natural
    syntactic context.

    `bin_idx` defaults to `episode_idx % 11`. With n_examples=30 per cell, each
    of the 11 depths gets ~2-3 episodes.
    """
    if bin_idx is None:
        bin_idx = episode_idx % N_POSITION_BINS
    depth_percent = bin_idx * (1.0 / (N_POSITION_BINS - 1))  # 0/10, 1/10, ..., 10/10
    haystack = base_haystack(seed, target_words)

    # Per-episode RNG — deterministic for (seed, episode_idx).
    rng = random.Random((seed * 1_000_003) ^ episode_idx)
    key = _make_key(rng)
    value = _make_value(rng)
    needle_sentence = f"One of the special magic numbers for {key} is: {value}."

    # Insertion point: target depth in chars, snapped to nearest `. ` boundary.
    n_chars = len(haystack)
    raw_pos = int(n_chars * depth_percent)
    # Find the next `. ` at or after raw_pos so needle goes at a sentence start.
    if raw_pos >= n_chars:
        insert_pos = n_chars
    else:
        snap = haystack.find(". ", raw_pos)
        if snap == -1:
            # Past the last `. ` — append at end.
            insert_pos = n_chars
        else:
            insert_pos = snap + 2  # after the `. `

    haystack_with_needle = (
        haystack[:insert_pos] + needle_sentence + " " + haystack[insert_pos:]
    )

    question = f"What is the special magic number for {key}?"
    # The prompt is what A0_baseline sends to the model as the user message
    # (procedures with system_prompt support pass document + question separately).
    prompt = haystack_with_needle + "\n\nQuestion: " + question + "\nAnswer:"
    haystack_hash = hashlib.sha256(haystack_with_needle.encode("utf-8")).hexdigest()[:16]

    return NiahEpisode(
        prompt=prompt,
        haystack_text=haystack_with_needle,
        question=question,
        needle=needle_sentence,
        key=key,
        value=value,
        bin_idx=bin_idx,
        depth_percent=depth_percent,
        position_char=insert_pos,
        haystack_hash=haystack_hash,
        target_words=target_words,
        seed=seed,
        episode_idx=episode_idx,
    )


def scorer(response: str, needle_or_value: str) -> float:
    """Substring match: 1.0 if value is found in response, else 0.0.

    Accepts either the full needle sentence or just the value — extracts the
    last whitespace-bounded token assuming it's the answer. For NiahEpisode
    callers pass `episode.value` directly.
    """
    if not response:
        return 0.0
    # If caller passed full needle sentence, extract value (last 7 digits).
    import re
    m = re.search(r"\b(\d{7})\b", needle_or_value)
    target = m.group(1) if m else needle_or_value
    return 1.0 if target in response else 0.0


# Legacy alias for code that imported `extract_password`. Now we just return
# the first 7-digit number in the response (the value pattern).
def extract_password(response: str) -> str:
    """Extract the first 7-digit number from response. Returns '' if not found."""
    import re
    m = re.search(r"\b\d{7}\b", response)
    return m.group(0) if m else ""
