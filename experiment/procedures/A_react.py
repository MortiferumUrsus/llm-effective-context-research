"""Project ReAct-style loop over a fixed source document.

Uses task-specific instructions and Search, Lookup, and Finish actions.
The upstream prompt examples are retained as reference material.
The available code includes a November lookup-continuation correction.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from bench.niah import NiahEpisode
from bench.utils import token_f1
from llm.base import LLMClient
from procedures.common import get_dataset_inputs, score_response
from runner.episode import IntermediateMessage


PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "react_yao2023"
WEBTHINK_PROMPT = (PROMPTS_DIR / "webthink_simple6.txt").read_text(encoding="utf-8")

MAX_STEPS = 6  # per v4 §1.1: "up to K=6 steps"

# Define document tools and the action format with a short example.
SYSTEM_REACT = (
    "You answer questions about a specific document by taking interleaved "
    "Thoughts and Actions. Available actions:\n"
    "- Search[<term>]: find a passage in the document containing <term>\n"
    "- Lookup[<term>]: find the next sentence containing <term>\n"
    "- Finish[<answer>]: submit your final answer (ends the trajectory)\n\n"
    "Reply in this exact format each turn:\n"
    "Thought N: <one-sentence reasoning>\n"
    "Action N: <one action>\n\n"
    "Example:\n"
    "Question: What is the secret code mentioned in the document?\n"
    "Thought 1: I need to search for the secret code.\n"
    "Action 1: Search[secret code]\n"
    "Observation 1: ...the secret code is X-93421...\n"
    "Thought 2: Found it. The code is X-93421.\n"
    "Action 2: Finish[X-93421]\n\n"
    "Always end with Action N: Finish[<answer>]."
)

# Used only as last-resort fallback if model fails to emit Finish[...] in
# max_steps. Sends the trajectory + a direct ask for the final answer with a
# different system prompt (terse-extractor mode).
SYSTEM_FORCED_FINAL = (
    "Read the trajectory below. Reply with ONLY the final answer value, "
    "no preamble, no explanation."
)

# Task-aware ReAct system prompt for BABILong qa2 (two supporting facts).
# Tells the model the document has bAbI facts hidden in narrative prose and
# multi-hop chaining is required.
SYSTEM_REACT_BABILONG = (
    "You answer questions about character positions and object locations "
    "hidden in a long narrative document. Short facts like 'John picked up "
    "the apple', 'Mary went to the kitchen', 'Sandra dropped the football' "
    "are scattered among irrelevant prose. To answer 'Where is X?' you must:\n"
    "  (1) Find WHO last interacted with X (picked up / dropped / took).\n"
    "  (2) Find WHERE that person went MOST RECENTLY.\n"
    "  (3) Answer with that ONE location word.\n\n"
    "Available actions:\n"
    "- Search[<term>]: find a passage containing <term>\n"
    "- Lookup[<term>]: find the next sentence containing <term>\n"
    "- Finish[<single_word_answer>]: terminate with one-word answer\n\n"
    "Format each turn:\n"
    "Thought N: <one-sentence reasoning>\n"
    "Action N: <one action>\n\n"
    "Example:\n"
    "Question: Where is the apple?\n"
    "Thought 1: Find who interacted with the apple last.\n"
    "Action 1: Search[apple]\n"
    "Observation 1: ...John picked up the apple. Then he went to the garden.\n"
    "Thought 2: John has the apple and is in the garden.\n"
    "Action 2: Finish[garden]\n\n"
    "Always end with Action N: Finish[<one word>]."
)

# Task-aware ReAct system prompt for Banking77 ICL.
# Reframes the agentic loop as classification over a label set.
SYSTEM_REACT_ICL = (
    "You are an intent classifier for banking customer-support queries. The "
    "'document' is a list of example queries paired with their correct intent "
    "labels (formatted as 'Query: ...' / 'Label: <label_name>'). Your task: "
    "find which label is most appropriate for the final query.\n\n"
    "Available actions:\n"
    "- Search[<term>]: find example queries containing <term>\n"
    "- Lookup[<term>]: find the next example mentioning <term>\n"
    "- Finish[<label_name>]: terminate with the chosen label (lowercase with "
    "underscores, exactly as shown in examples)\n\n"
    "Format each turn:\n"
    "Thought N: <one-sentence reasoning>\n"
    "Action N: <one action>\n\n"
    "Example:\n"
    "Question: How do I reset my PIN?\n"
    "Thought 1: Search for example queries about PIN.\n"
    "Action 1: Search[reset PIN]\n"
    "Observation 1: ...Query: I forgot my pin number. Label: pin_blocked...\n"
    "Thought 2: That example matches — the label is pin_blocked.\n"
    "Action 2: Finish[pin_blocked]\n\n"
    "Always end with Action N: Finish[<label_name>]."
)

_ACTION_RE = re.compile(
    r"Action\s*\d+\s*:\s*(Search|Lookup|Finish)\s*\[(.+?)\]",
    re.IGNORECASE | re.DOTALL,
)
_FINISH_RE = re.compile(
    r"(?:Action\s*\d+\s*:\s*)?Finish\s*\[(.+?)\]",
    re.IGNORECASE | re.DOTALL,
)


def _backend_search(haystack: str, query: str, max_chars: int = 600) -> str:
    """Substring search → return a window around the first match.

    Tries each whitespace-separated term of the query in order. Returns a
    ±max_chars/2 window around the FIRST term that hits in the haystack.
    Falls back to "Could not find" if none of the terms appear.

    Designed to work with any haystack (RULER noise single-paragraph or
    Wikipedia-style multi-paragraph). Original Yao 2023 paragraph-split was
    inadequate because RULER's noise haystack has no `\\n\\n` breaks → the
    whole haystack was matched and only the first 400 chars returned.
    """
    q_terms = [t.strip() for t in re.split(r"\s+", query.strip()) if t.strip()]
    if not q_terms:
        return f"Could not find [{query}]."
    hay_lower = haystack.lower()
    for term in q_terms:
        pos = hay_lower.find(term.lower())
        if pos < 0:
            continue
        # Window centered on the match.
        half = max_chars // 2
        start = max(0, pos - half)
        end = min(len(haystack), pos + len(term) + half)
        snippet = haystack[start:end]
        # Add ellipsis markers when truncated for clarity.
        if start > 0:
            snippet = "..." + snippet
        if end < len(haystack):
            snippet = snippet + "..."
        return snippet
    return f"Could not find [{query}]."


def _backend_lookup(haystack: str, term: str, max_chars: int = 400,
                    cursors: Optional[dict[str, int]] = None) -> str:
    """Return successive matches when an episode-local cursor is provided.

    Calls without a cursor retain the historical first-hit behavior for
    diagnostic comparisons. The normal revised runner supplies a cursor.
    """
    key = term.strip().lower()
    if not key:
        return "No more results."
    matches = list(re.finditer(re.escape(key), haystack, re.IGNORECASE))
    index = cursors.get(key, 0) if cursors is not None else 0
    if index >= len(matches):
        return "No more results."
    pos = matches[index].start()
    if cursors is not None:
        cursors[key] = index + 1
    start = max(0, pos - 80)
    end = min(len(haystack), pos + max_chars)
    return f"(Result {index + 1} / {len(matches)}; character {pos}) {haystack[start:end]}"


def _parse_action(text: str) -> tuple[str, str] | None:
    """Pull the first Action line from the model's reply."""
    m = _ACTION_RE.search(text)
    if m:
        return m.group(1).capitalize(), m.group(2).strip()
    # Free-form "Finish: <answer>" without Action prefix
    m = _FINISH_RE.search(text)
    if m:
        return "Finish", m.group(1).strip()
    return None


def _pick_system_prompt(episode) -> str:
    """Choose task-aware system prompt by episode type (v6.1)."""
    type_name = type(episode).__name__
    if type_name == "BabilongEpisode":
        return SYSTEM_REACT_BABILONG
    if type_name == "Banking77IclEpisode":
        return SYSTEM_REACT_ICL
    return SYSTEM_REACT  # NIAH / HotPotQA / RULER / NoLiMa — generic


def run(
    episode,
    llm: LLMClient,
    max_steps: int = MAX_STEPS,
    max_tokens_per_step: int = 256,
) -> dict:
    """Dataset-agnostic ReAct runner with task-aware system prompt switching."""
    haystack, question, gold = get_dataset_inputs(episode)
    system_prompt = _pick_system_prompt(episode)

    intermediate: list[dict] = []
    # Trajectory is the user-message content. System prompt holds the ReAct
    # format spec; user holds the question + accumulated Thought/Action/Observation.
    trajectory = f"Question: {question}\nThought 1:"
    final_answer = ""
    finish_reason = "length"
    lookup_cursors: dict[str, int] = {}

    for step in range(1, max_steps + 1):
        thought_action_prompt = trajectory if step == 1 else trajectory + f"\nThought {step}:"
        r = llm.generate(
            thought_action_prompt,
            max_tokens=max_tokens_per_step,
            stop=[f"\nObservation {step}:", f"\nThought {step+1}:", "\nQuestion:"],
            temperature=0.0,
            system_prompt=system_prompt,
        )
        thought_action = r.text
        intermediate.append(
            IntermediateMessage(
                role=f"react/thought_action#{step}",
                input_tokens=r.input_tokens,
                output_tokens=r.output_tokens,
                duration_ms=r.wall_ms,
                summary=thought_action[:200],
                # Per-call finish_reason + error for provider-failure detection
                # in multi-step methods (Codex V6.2 P1.6).
                extra={
                    "finish_reason": r.finish_reason,
                    "error": r.extra.get("error", "") if r.extra else "",
                },
            ).__dict__
        )

        # Append model's output to trajectory
        if step == 1:
            trajectory = f"Question: {question}\nThought 1:{thought_action}"
        else:
            trajectory += f"\nThought {step}:{thought_action}"

        parsed = _parse_action(thought_action)
        if parsed is None:
            obs = "Invalid action. Use Search[term], Lookup[term], or Finish[answer]."
            trajectory += f"\nObservation {step}: {obs}"
            continue
        act, arg = parsed
        if act == "Finish":
            final_answer = arg
            finish_reason = "stop"
            break
        elif act == "Search":
            obs = _backend_search(haystack, arg)
        elif act == "Lookup":
            obs = _backend_lookup(haystack, arg, cursors=lookup_cursors)
        else:
            obs = f"Unknown action {act}"
        trajectory += f"\nObservation {step}: {obs}"
        intermediate.append(
            IntermediateMessage(
                role=f"react/observation#{step}",
                summary=f"{act}[{arg}] -> {obs[:150]}",
            ).__dict__
        )

    # Forced-final fallback: if model never emitted Finish[...] in max_steps,
    # do ONE extra generate call with a terse-extractor system prompt asking
    # for ONLY the final answer. This counts toward scoring but is flagged as
    # `finish_reason="forced_final"` so analysis can filter.
    if not final_answer:
        forced = llm.generate(
            trajectory + "\n\nBased on the trajectory above, what is your final answer?",
            max_tokens=64,
            stop=["\n\n"],
            temperature=0.0,
            system_prompt=SYSTEM_FORCED_FINAL,
        )
        intermediate.append(
            IntermediateMessage(
                role="react/forced_final",
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
        finish_reason = "forced_final"
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
