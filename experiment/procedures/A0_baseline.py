"""A_0 — single-LLM baseline with proper system/user split.

Sends the document + question in the `user` role and a brief format-instruction
system prompt. The system message guarantees a parseable `Answer: <X>` response
for substring scoring; without it, chat-tuned models add filler that breaks
exact-match metrics. No agents, no retrieval, no grammar.

Context-fit policy (v6.1, replaces v5's silent head+tail truncation):
  * If prompt + output_budget + safety fits in model_ctx → use as-is.
  * If overflow ≤ FIT_OVERFLOW_THRESHOLD_PCT (default 20%) → soft middle-trim
    to fit. This handles the case where a 28K document + question marginally
    exceeds a 32K model — we keep the experiment going with a small,
    transparent trim.
  * If overflow > FIT_OVERFLOW_THRESHOLD_PCT → return ctx_overflow with
    metric=0 and NO API call. This is the honest signal: the model simply
    cannot ingest this length. Logged with finish_reason="ctx_overflow".

All three decisions are recorded in the episode `extra` dict for downstream
analysis (so we can split f(N) curves by trim status).
"""
from __future__ import annotations

import re
from typing import Optional

from bench.niah import NiahEpisode, scorer as niah_scorer, extract_password
from bench.utils import truncate_middle, token_f1
from llm.base import GenerateResult, LLMClient
from runner.episode import IntermediateMessage


# Minimal stops — system prompt + Answer: format do the heavy lifting.
DEFAULT_STOPS = ["\n\n"]

# Maximum overflow tolerated for soft trim. Above this, ctx_overflow is returned.
FIT_OVERFLOW_THRESHOLD_PCT = 0.20

# Tokens reserved on top of `max_tokens` for system prompt + format overhead.
RESERVED_OVERHEAD_TOKENS = 200

# Unified system prompt for extractive QA tasks (NIAH, NoLiMa, RULER, BABILong).
SYSTEM_EXTRACTIVE = (
    "You are a precise reading assistant. Read the document and answer the "
    "question using only information from the document. Respond ONLY in this "
    "format:\nAnswer: <your answer>"
)

# Slightly different system prompt for HotPotQA (free-form, may need short phrase).
SYSTEM_FREEFORM = (
    "You answer questions based on the given passages. Reply with just the "
    "final answer (a short phrase or a few words) in this format:\nAnswer: <your answer>"
)

# BANKING77 ICL system prompt — ICL classifier mode. Inline (no Answer: marker)
# because the ICL prompt itself has 'Label:' as a position-cue for the model.
SYSTEM_ICL_CLASSIFY = (
    "You are a banking customer-support intent classifier. Given a query, "
    "output exactly one intent label from the set demonstrated in the "
    "examples. Output ONLY the label (lowercase with underscores). Do not "
    "add any other text."
)

# BABILong-specific system prompt — canonical Kuratov 2024 framing.
# Tells the model that facts about positions are hidden among narrative prose.
# Without this, models default to "last sentence containing entity" rather than
# tracking state.
SYSTEM_BABILONG = (
    "You are reading a long document where facts about character positions and "
    "movements (e.g. 'John went to the kitchen', 'Mary picked up the apple') "
    "are hidden among irrelevant narrative prose. Your job is to track WHERE "
    "each entity ends up by chaining the facts together. Output exactly one "
    "word — the final location or object name. Respond ONLY in this format:\n"
    "Answer: <one word>"
)

# Extract "Answer: <text>" from model output. Falls back to full text.
_ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _extract_answer(text: str) -> str:
    """Find 'Answer: <X>' in model output. If not present, return full text stripped."""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _check_and_fit(
    prompt: str,
    model_ctx_tokens: int,
    llm: LLMClient,
    output_budget: int,
) -> tuple[Optional[str], str, int, int]:
    """Decide if prompt fits in model context.

    Returns (maybe_trimmed_prompt, decision, n_tokens_pre, overflow_tokens).
      * decision = "fits"        → prompt unchanged, no trim, OK to call LLM
      * decision = "trimmed"     → small overflow, middle-trimmed, OK to call LLM
      * decision = "ctx_overflow" → large overflow, returns None — caller must
                                    skip LLM call and record metric=0
    """
    try:
        n_in = llm.count_tokens(prompt)
    except Exception:
        n_in = len(prompt) // 4  # rough char/4 fallback

    doc_budget = model_ctx_tokens - output_budget - RESERVED_OVERHEAD_TOKENS
    doc_budget = max(256, doc_budget)

    if n_in <= doc_budget:
        return prompt, "fits", n_in, 0

    overflow = n_in - doc_budget
    overflow_pct = overflow / max(1, n_in)

    if overflow_pct > FIT_OVERFLOW_THRESHOLD_PCT:
        return None, "ctx_overflow", n_in, overflow

    # Soft trim: middle-out, minimum needed.
    target_words = max(64, int(doc_budget / 1.3))
    trimmed = truncate_middle(prompt, target_words)
    return trimmed, "trimmed", n_in, overflow


def _ctx_overflow_result(out_text: str, n_tokens_pre: int, overflow: int,
                          decision: str, metric_name: str) -> dict:
    """Build a stub result for ctx_overflow cases — no API call was made."""
    return {
        "output": out_text,
        "extracted": "",
        "metric_value": 0.0,
        "metric_name": metric_name,
        "input_tokens": 0,
        "output_tokens": 0,
        "wall_clock_ms": 0.0,
        "intermediate_messages": [],
        "finish_reason": "ctx_overflow",
        "n_tokens_pre_call": n_tokens_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }


def _build_user_message(document: str, question: str) -> str:
    """User message = document + question. System holds format instructions."""
    return f"{document}\n\nQuestion: {question}"


def run_niah(
    episode: NiahEpisode,
    llm: LLMClient,
    model_ctx_tokens: int = 8192,
    max_tokens: int = 64,
) -> dict:
    """Run A0 on a single NIAH (RULER niah-single) episode."""
    user_msg = _build_user_message(episode.haystack_text, episode.question)
    fitted, decision, n_pre, overflow = _check_and_fit(
        user_msg, model_ctx_tokens, llm, output_budget=max_tokens
    )
    if decision == "ctx_overflow":
        return _ctx_overflow_result("", n_pre, overflow, decision, "exact_match")

    result: GenerateResult = llm.generate(
        fitted,
        max_tokens=max_tokens,
        stop=DEFAULT_STOPS,
        temperature=0.0,
        system_prompt=SYSTEM_EXTRACTIVE,
    )
    extracted = _extract_answer(result.text)
    # Score against the value (7-digit number). Substring match.
    metric = niah_scorer(extracted + " " + result.text, episode.value)
    intermediate = [
        IntermediateMessage(
            role="A0/single_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.wall_ms,
            summary=result.text[:200],
            extra={
                "trim_decision": decision,
                "extracted": extracted,
                "stopping_word": result.extra.get("stopping_word", ""),
            },
        ).__dict__
    ]
    return {
        "output": result.text,
        "extracted": extracted,
        "metric_value": metric,
        "metric_name": "exact_match",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_ms": result.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": result.finish_reason,
        "n_tokens_pre_call": n_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }


def run_freeform(
    prompt_or_document: str,
    gold: str,
    llm: LLMClient,
    model_ctx_tokens: int = 8192,
    max_tokens: int = 128,
    question: Optional[str] = None,
) -> dict:
    """Generic free-form A0 runner. If question is None, treats prompt as legacy
    full-prompt string (back-compat with old callers)."""
    if question is None:
        user_msg = prompt_or_document
    else:
        user_msg = _build_user_message(prompt_or_document, question)
    fitted, decision, n_pre, overflow = _check_and_fit(
        user_msg, model_ctx_tokens, llm, output_budget=max_tokens
    )
    if decision == "ctx_overflow":
        return _ctx_overflow_result("", n_pre, overflow, decision, "f1")

    result: GenerateResult = llm.generate(
        fitted,
        max_tokens=max_tokens,
        stop=DEFAULT_STOPS,
        temperature=0.0,
        system_prompt=SYSTEM_FREEFORM,
    )
    extracted = _extract_answer(result.text)
    metric = token_f1(extracted, gold)
    intermediate = [
        IntermediateMessage(
            role="A0/single_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.wall_ms,
            summary=result.text[:200],
            extra={"trim_decision": decision, "extracted": extracted, "gold": gold[:120]},
        ).__dict__
    ]
    return {
        "output": result.text,
        "extracted": extracted,
        "metric_value": metric,
        "metric_name": "f1",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_ms": result.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": result.finish_reason,
        "n_tokens_pre_call": n_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }


def run_ruler(
    episode,
    llm: LLMClient,
    model_ctx_tokens: int = 8192,
    max_tokens: int = 96,
) -> dict:
    """A0 on a RULER episode (multikey/multivalue). Substring match scorer."""
    from bench import ruler as ruler_mod

    user_msg = _build_user_message(getattr(episode, "haystack_text", ""), episode.question)
    fitted, decision, n_pre, overflow = _check_and_fit(
        user_msg, model_ctx_tokens, llm, output_budget=max_tokens
    )
    if decision == "ctx_overflow":
        metric_name = "exact_match" if episode.task_subtype == "multikey" else "partial_match"
        return _ctx_overflow_result("", n_pre, overflow, decision, metric_name)

    result: GenerateResult = llm.generate(
        fitted,
        max_tokens=max_tokens,
        stop=DEFAULT_STOPS,
        temperature=0.0,
        system_prompt=SYSTEM_EXTRACTIVE,
    )
    extracted = _extract_answer(result.text)
    metric = ruler_mod.scorer(extracted, episode.gold_answer, episode.task_subtype)
    intermediate = [
        IntermediateMessage(
            role="A0/single_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.wall_ms,
            summary=result.text[:200],
            extra={"trim_decision": decision, "extracted": extracted},
        ).__dict__
    ]
    return {
        "output": result.text,
        "extracted": extracted,
        "metric_value": metric,
        "metric_name": "exact_match" if episode.task_subtype == "multikey" else "partial_match",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_ms": result.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": result.finish_reason,
        "n_tokens_pre_call": n_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }


def run_hotpotqa(
    episode,
    llm: LLMClient,
    model_ctx_tokens: int = 8192,
    max_tokens: int = 128,
) -> dict:
    """A0 on a HotPotQA episode. F1 scorer."""
    return run_freeform(
        getattr(episode, "haystack_text", "") or getattr(episode, "context", ""),
        episode.gold_answer,
        llm,
        model_ctx_tokens=model_ctx_tokens,
        max_tokens=max_tokens,
        question=episode.question,
    )


def run_babilong(
    episode,
    llm: LLMClient,
    model_ctx_tokens: int = 32768,
    max_tokens: int = 32,
) -> dict:
    """A0 on a BABILong qa2 episode. Substring match on target word."""
    from bench import babilong as bab_mod

    user_msg = _build_user_message(episode.haystack_text, episode.question)
    fitted, decision, n_pre, overflow = _check_and_fit(
        user_msg, model_ctx_tokens, llm, output_budget=max_tokens
    )
    if decision == "ctx_overflow":
        return _ctx_overflow_result("", n_pre, overflow, decision, "exact_match")

    result: GenerateResult = llm.generate(
        fitted,
        max_tokens=max_tokens,
        stop=DEFAULT_STOPS,
        temperature=0.0,
        system_prompt=SYSTEM_BABILONG,
    )
    extracted = _extract_answer(result.text)
    metric = bab_mod.scorer(extracted + " " + result.text, episode.gold_answer)
    intermediate = [
        IntermediateMessage(
            role="A0/single_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.wall_ms,
            summary=result.text[:200],
            extra={"trim_decision": decision, "extracted": extracted,
                   "gold": episode.gold_answer},
        ).__dict__
    ]
    return {
        "output": result.text,
        "extracted": extracted,
        "metric_value": metric,
        "metric_name": "exact_match",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_ms": result.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": result.finish_reason,
        "n_tokens_pre_call": n_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }


def run_banking77_icl(
    episode,
    llm: LLMClient,
    model_ctx_tokens: int = 32768,
    max_tokens: int = 16,
) -> dict:
    """A0 on a BANKING77 ICL episode. Label-name match scoring.

    For ICL we use a DIFFERENT system prompt — no 'Answer:' marker is added by
    the model, because the prompt already ends with 'Label:' as the position
    cue. The model emits just the label name.
    """
    from bench import banking77_icl as bk_mod

    # ICL prompt already contains full structure (instruction + demos + 'Label:').
    # We send it as-is in user message; system is the classify role.
    fitted, decision, n_pre, overflow = _check_and_fit(
        episode.prompt, model_ctx_tokens, llm, output_budget=max_tokens
    )
    if decision == "ctx_overflow":
        return _ctx_overflow_result("", n_pre, overflow, decision, "exact_match")

    result: GenerateResult = llm.generate(
        fitted,
        max_tokens=max_tokens,
        stop=["\n", "Query:", "Label:"],  # one-line label only
        temperature=0.0,
        system_prompt=SYSTEM_ICL_CLASSIFY,
    )
    metric = bk_mod.scorer(result.text, episode.gold_answer)
    intermediate = [
        IntermediateMessage(
            role="A0/single_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.wall_ms,
            summary=result.text[:200],
            extra={"trim_decision": decision, "gold": episode.gold_answer,
                   "n_demos": episode.extra.get("n_demos") if episode.extra else None},
        ).__dict__
    ]
    return {
        "output": result.text,
        "extracted": result.text.strip(),
        "metric_value": metric,
        "metric_name": "exact_match",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_ms": result.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": result.finish_reason,
        "n_tokens_pre_call": n_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }


def run_nolima(
    episode,
    llm: LLMClient,
    model_ctx_tokens: int = 8192,
    max_tokens: int = 64,
) -> dict:
    """A0 on a NoLiMa episode (paraphrased needle). Substring match.

    Legacy — not used in v6.1 grid but kept for orchestrator compatibility.
    """
    from bench import nolima as nolima_mod

    user_msg = _build_user_message(getattr(episode, "haystack_text", ""), episode.question)
    fitted, decision, n_pre, overflow = _check_and_fit(
        user_msg, model_ctx_tokens, llm, output_budget=max_tokens
    )
    if decision == "ctx_overflow":
        return _ctx_overflow_result("", n_pre, overflow, decision, "exact_match")

    result: GenerateResult = llm.generate(
        fitted,
        max_tokens=max_tokens,
        stop=DEFAULT_STOPS,
        temperature=0.0,
        system_prompt=SYSTEM_EXTRACTIVE,
    )
    extracted = _extract_answer(result.text)
    metric = nolima_mod.scorer(extracted, episode.needle)
    intermediate = [
        IntermediateMessage(
            role="A0/single_call",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            duration_ms=result.wall_ms,
            summary=result.text[:200],
            extra={"trim_decision": decision, "extracted": extracted,
                   "paraphrase_idx": getattr(episode, "paraphrase_idx", None)},
        ).__dict__
    ]
    return {
        "output": result.text,
        "extracted": extracted,
        "metric_value": metric,
        "metric_name": "exact_match",
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "wall_clock_ms": result.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": result.finish_reason,
        "n_tokens_pre_call": n_pre,
        "trim_decision": decision,
        "overflow_tokens": overflow,
    }
