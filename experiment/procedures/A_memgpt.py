"""Memory-style archival search implementation.

Uses the MemGPT document prompt with project-specific task instructions,
adaptive BM25 retrieval, deduplication, and a final-answer action.
This is not a complete reproduction of the MemGPT architecture.
See docs/systems.md and this snapshot's README for provenance.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:
    BM25Okapi = None  # type: ignore

from bench.niah import NiahEpisode
from llm.base import LLMClient
from procedures.common import get_dataset_inputs, score_response
from runner.episode import IntermediateMessage


PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "memgpt_packer2023" / "memgpt_doc.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")

MAX_STEPS = 6  # cap the loop similar to ReAct K=6

# Task-aware addenda appended to the canonical Packer 2023 MemGPT prompt for
# datasets where the generic "document analysis" framing leads the model astray.
ADDENDUM_BABILONG = (
    "TASK SPECIFICATION (BABILong multi-hop fact tracking):\n"
    "The archival memory holds a long narrative document where short facts "
    "about character positions and object movements ('John picked up the apple', "
    "'Mary went to the kitchen') are scattered among irrelevant prose. To "
    "answer 'Where is X?':\n"
    "  1. Use archival_memory_search to find facts about X (who has X, who "
    "interacted with X).\n"
    "  2. Once you know WHO has X, search for that person's most recent "
    "LOCATION (e.g. 'John went to ...').\n"
    "  3. Once you have the final location, immediately call "
    "send_message(message=\"<single word location>\").\n\n"
    "IMPORTANT: send_message MUST contain ONLY ONE WORD — the location or "
    "object name (e.g. 'kitchen', 'garden', 'bathroom'). NO sentences, NO "
    "explanations.\n"
)

ADDENDUM_ICL = (
    "TASK SPECIFICATION (Banking77 intent classification):\n"
    "The archival memory holds a long list of example queries paired with "
    "their correct intent labels (e.g. 'Query: I forgot my PIN. Label: "
    "pin_blocked'). Your job is INTENT CLASSIFICATION — not answering the "
    "user's question, but choosing which LABEL fits the user's query.\n\n"
    "Process:\n"
    "  1. Use archival_memory_search to find example queries similar to the "
    "user's query.\n"
    "  2. Read which Label is attached to those similar examples.\n"
    "  3. Call send_message(message=\"<label_name>\") with the exact label "
    "name (lowercase with underscores, e.g. 'pin_blocked', 'verify_my_identity').\n\n"
    "IMPORTANT: send_message MUST contain ONLY the label name — one token, no "
    "explanations, no full sentences. DO NOT try to answer the user's question; "
    "just classify it."
)

_ACTION_RE = re.compile(
    r"(archival_memory_search|send_message)\s*\(\s*(?:query|message)?\s*=?\s*[\"']?(.+?)[\"']?\s*\)",
    re.IGNORECASE | re.DOTALL,
)


def _chunk_text(text: str, chunk_words: int = 380, overlap_words: int = 48) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks


def _adaptive_top_k(n_chunks: int, max_steps: int = 6) -> int:
    """top_k such that max_steps queries can cover ~60% of haystack chunks.

    For 9-chunk haystack (3K words): top_k=2 (cap 3) → 6×2=12, de-duped ~7-8.
    For 70-chunk haystack (24K words): top_k=7 → 6×7=42, de-duped ~30 (43%).
    Capped at 10 to keep token budget bounded.
    """
    target_total = int(n_chunks * 0.6)
    per_query = max(3, (target_total + max_steps - 1) // max_steps)
    return min(per_query, 10)


def _archival_memory_search(
    haystack: str,
    query: str,
    seen_indices: Optional[set] = None,
    max_steps: int = 6,
) -> tuple[str, list[int]]:
    """BM25 over the haystack with adaptive top_k + de-dup across queries.

    Returns (formatted_result, newly_selected_indices). De-dup excludes chunks
    already shown to the model in previous archival_memory_search calls.
    """
    chunks = _chunk_text(haystack)
    n_chunks = len(chunks)
    seen = seen_indices or set()
    top_k = _adaptive_top_k(n_chunks, max_steps=max_steps)
    # Display FULL chunk content (no truncation) — chunks are 380 words ≈ 2700
    # chars. Truncating to 1200 chars cut off needles at depth >10% within each
    # chunk, causing MemGPT to repeatedly search without seeing the answer.
    if BM25Okapi is None or n_chunks <= top_k:
        new_idxs = [i for i in range(n_chunks) if i not in seen][:top_k]
        result = "\n".join(f"[{i+1}] {chunks[idx]}" for i, idx in enumerate(new_idxs))
        return result, new_idxs
    tokenised = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenised)
    scores = np.array(bm25.get_scores(query.lower().split()))
    order = np.argsort(-scores)
    new_idxs = [int(i) for i in order if int(i) not in seen][:top_k]
    selected = [chunks[i] for i in new_idxs]
    result = "\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(selected))
    return result, new_idxs


def _parse_action(raw: str) -> tuple[str, str] | None:
    """Extract an MemGPT function call from the model's inner-monologue+action."""
    m = _ACTION_RE.search(raw)
    if m:
        return m.group(1).lower(), m.group(2).strip().strip("'\"")
    # Heuristic fallback: line starting with "send_message:" or similar
    m2 = re.search(r"(?:Final\s*answer|Answer|Reply)\s*:\s*(.+?)$", raw,
                   re.IGNORECASE | re.MULTILINE)
    if m2:
        return "send_message", m2.group(1).strip()
    return None


def run(episode, llm: LLMClient, max_steps: int = MAX_STEPS, max_tokens: int = 256) -> dict:
    """Dataset-agnostic MemGPT runner with system/user split + adaptive top_k."""
    # Reset slot KV state for clean trajectory (no contamination from prior episode).
    # Local LlamaCpp implements as POST /slots/N?action=erase; API clients no-op.
    if hasattr(llm, "reset_slot"):
        llm.reset_slot()

    haystack, question, gold = get_dataset_inputs(episode)

    intermediate: list[dict] = []
    final_answer = ""
    finish_reason = "length"
    seen_chunks: set[int] = set()

    # System prompt is Packer 2023 MemGPT-style + our terse format request +
    # task-specific addendum based on episode type. The addendum tells the
    # model HOW the canonical "archival_memory_search/send_message" tools map
    # to this particular task (multi-hop tracking, classification, etc.).
    ep_type = type(episode).__name__
    if ep_type == "BabilongEpisode":
        task_addendum = "\n\n" + ADDENDUM_BABILONG
    elif ep_type == "Banking77IclEpisode":
        task_addendum = "\n\n" + ADDENDUM_ICL
    else:
        task_addendum = ""

    system_prompt = (
        f"{SYSTEM_PROMPT.strip()}\n\n"
        f"You have a document loaded in archival memory. Use these tools:\n"
        f"  - archival_memory_search(query=\"...\") — retrieve top BM25 chunks\n"
        f"  - send_message(message=\"...\") — give your final answer (ends conversation)\n\n"
        f"Each step format:\n"
        f"Inner monologue: <one-sentence reasoning>\n"
        f"<one tool call>\n\n"
        f"IMPORTANT: As soon as a TOOL RESULT contains the information you need, "
        f"immediately call send_message(message=\"<answer>\") on the next step. "
        f"Do NOT search again if the answer is already visible in retrieved chunks.\n\n"
        f"Always finish with send_message(message=\"<your answer>\")."
        f"{task_addendum}"
    )

    trajectory = f"Question: {question}\n\n--- Step 1 ---\nInner monologue:"

    for step in range(1, max_steps + 1):
        prompt = trajectory if step == 1 else trajectory + f"\n\n--- Step {step} ---\nInner monologue:"
        r = llm.generate(
            prompt,
            max_tokens=max_tokens,
            stop=[f"--- Step {step+1} ---", "Question:", "TOOL RESULT:"],
            temperature=0.0,
            system_prompt=system_prompt,
        )
        intermediate.append(
            IntermediateMessage(
                role=f"memgpt/step#{step}",
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                duration_ms=r.wall_ms,
                summary=r.text[:200],
                # Per-call finish_reason for multi-step provider-failure detection
                # (Codex V6.2 P1.6).
                extra={
                    "finish_reason": r.finish_reason,
                    "error": r.extra.get("error", "") if r.extra else "",
                },
            ).__dict__
        )
        if step == 1:
            trajectory = f"Question: {question}\n\n--- Step 1 ---\nInner monologue:{r.text}"
        else:
            trajectory += f"\n\n--- Step {step} ---\nInner monologue:{r.text}"

        parsed = _parse_action(r.text)
        if parsed is None:
            trajectory += "\nTOOL RESULT: Invalid format. Use archival_memory_search(query=\"...\") or send_message(message=\"...\")."
            continue

        func, arg = parsed
        if func == "send_message":
            final_answer = arg
            finish_reason = "stop"
            break
        elif func == "archival_memory_search":
            result, new_idxs = _archival_memory_search(
                haystack, arg, seen_indices=seen_chunks, max_steps=max_steps
            )
            seen_chunks.update(new_idxs)
            trajectory += f"\nTOOL RESULT:\n{result}"
            intermediate.append(
                IntermediateMessage(
                    role=f"memgpt/archival_search#{step}",
                    summary=f"query={arg!r} new_chunks={new_idxs} total_seen={len(seen_chunks)}",
                ).__dict__
            )

    # Forced-final fallback (mirror of ReAct's pattern): if model never called
    # send_message in max_steps, do ONE extra LLM call with a terse-extractor
    # system prompt asking for just the final answer. This catches cases where
    # the model emitted the answer as a bare word ('river', 'bedroom') in the
    # inner_monologue but never wrapped it in send_message(). Flag as
    # finish_reason='forced_final' so analysis can filter.
    if not final_answer:
        forced = llm.generate(
            trajectory + "\n\nBased on the trajectory above, what is your final answer? Reply with ONLY the value, no preamble.",
            max_tokens=32,
            stop=["\n\n"],
            temperature=0.0,
            system_prompt="Read the trajectory and reply with ONLY the final answer value (one word or short phrase), no preamble.",
        )
        intermediate.append(
            IntermediateMessage(
                role="memgpt/forced_final",
                input_tokens=forced.input_tokens,
                output_tokens=forced.output_tokens,
                duration_ms=forced.wall_ms,
                summary=forced.text[:200],
                extra={
                    "finish_reason": forced.finish_reason,
                    "error": forced.extra.get("error", "") if forced.extra else "",
                },
            ).__dict__
        )
        final_answer = forced.text.strip()
        finish_reason = "forced_final" if final_answer else "no_send_message"
    metric_value, metric_name = score_response(final_answer, gold, episode)
    total_in = sum(m["input_tokens"] for m in intermediate)
    total_out = sum(m["output_tokens"] for m in intermediate)
    total_dur = sum(m["duration_ms"] for m in intermediate)

    return {
        "output": final_answer,
        "metric_value": metric_value,
        "metric_name": metric_name,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "wall_clock_ms": total_dur,
        "intermediate_messages": intermediate,
        "finish_reason": finish_reason,
        "n_steps": len(intermediate),
    }
