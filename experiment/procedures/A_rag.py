"""A_RAG — classical BM25 + bge-small + RRF top-k retrieval, then single LLM pass.

Pipeline:
  1. Split haystack into 380-word chunks with 48-word overlap.
  2. BM25 score each chunk against the question.
  3. bge-small dense score each chunk against the question.
  4. Reciprocal Rank Fusion (RRF) combine top-k from each → final top-5.
  5. Concatenate top-5 chunks in rank order, or explicitly selected source order.

Chunk lengths are words, not measured tokens. Source order is a distinct
experimental condition; historical results used rank order.
The embedder is loaded once and cached at module level.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except ImportError:
    BM25Okapi = None  # type: ignore

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from bench.niah import NiahEpisode, extract_password, scorer as niah_scorer
from bench.utils import token_f1
from llm.base import LLMClient
from procedures.common import get_dataset_inputs, score_response
from runner.episode import IntermediateMessage


_EMBEDDER: Optional["SentenceTransformer"] = None
_EMBEDDER_NAME = "BAAI/bge-small-en-v1.5"


_EMBEDDER_LOAD_FAILED = False


def _get_embedder():
    """Load bge-small once; on crash/segfault path, return None and downgrade
    to BM25-only retrieval (still a valid RAG baseline)."""
    global _EMBEDDER, _EMBEDDER_LOAD_FAILED
    if _EMBEDDER_LOAD_FAILED:
        return None
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Force CPU — local Qwen-7B holds all GTX 1660 Ti VRAM during Exp1.
            # Concurrent CUDA from bge-small caused 0xC0000005 access violations
            # in the api_runner_v4 subprocess.
            _EMBEDDER = SentenceTransformer(_EMBEDDER_NAME, device="cpu")
        except Exception:
            _EMBEDDER_LOAD_FAILED = True
            return None
    return _EMBEDDER


def _chunk_text(text: str, chunk_words: int = 380, overlap_words: int = 48) -> list[str]:
    """Approximate 512-token chunks via word count (1.3 tok/word heuristic).

    With chunk_words=380 we get ~494 tokens per chunk, overlap_words=48 → ~62 tokens.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks


def _rrf_fusion(ranks_a: list[int], ranks_b: list[int], k: int = 60) -> list[float]:
    """Compute RRF scores: 1/(k+rank_a) + 1/(k+rank_b). Equal length."""
    n = max(len(ranks_a), len(ranks_b))
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        scores[i] = 1.0 / (k + ranks_a[i] + 1) + 1.0 / (k + ranks_b[i] + 1)
    return scores.tolist()


def retrieve_chunks(
    haystack: str,
    question: str,
    top_k: int = 5,
    chunk_words: int = 380,
    overlap_words: int = 48,
    source_order: bool = False,
) -> tuple[list[str], dict]:
    """Run BM25 + bge-small + RRF and return top-k chunk texts."""
    chunks = _chunk_text(haystack, chunk_words=chunk_words, overlap_words=overlap_words)
    if len(chunks) <= top_k:
        return chunks, {"n_chunks": len(chunks), "fallback": "few_chunks",
                        "top_idx": list(range(len(chunks))),
                        "source_order": source_order, "retrieval_mode": "all_chunks"}

    # BM25 ranks (descending score)
    if BM25Okapi is None:
        bm25_scores = np.zeros(len(chunks))
    else:
        tokenised = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenised)
        bm25_scores = np.array(bm25.get_scores(question.lower().split()))
    bm25_rank = np.argsort(-bm25_scores)
    bm25_rank_pos = {int(i): r for r, i in enumerate(bm25_rank)}

    # Dense ranks
    embedder = _get_embedder()
    if embedder is None:
        dense_scores = np.zeros(len(chunks))
    else:
        emb_chunks = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
        emb_q = embedder.encode([question], normalize_embeddings=True, show_progress_bar=False)
        dense_scores = emb_chunks @ emb_q.T
        dense_scores = dense_scores.flatten()
    dense_rank = np.argsort(-dense_scores)
    dense_rank_pos = {int(i): r for r, i in enumerate(dense_rank)}

    if BM25Okapi is None and embedder is None:
        raise RuntimeError("Neither BM25 nor the dense retriever is available")
    if embedder is None:
        top_idx = bm25_rank[:top_k]
        retrieval_mode = "bm25_only"
    elif BM25Okapi is None:
        top_idx = dense_rank[:top_k]
        retrieval_mode = "dense_only"
    else:
        rrf = np.array(_rrf_fusion(
            [bm25_rank_pos[i] for i in range(len(chunks))],
            [dense_rank_pos[i] for i in range(len(chunks))],
        ))
        top_idx = np.argsort(-rrf)[:top_k]
        retrieval_mode = "bm25_bge_rrf"
    rank_order_idx = [int(i) for i in top_idx]
    if source_order:
        top_idx = np.sort(top_idx)
    selected = [chunks[i] for i in top_idx]
    return selected, {
        "n_chunks": len(chunks),
        "top_idx": [int(i) for i in top_idx],
        "rank_order_idx": rank_order_idx,
        "source_order": source_order,
        "retrieval_mode": retrieval_mode,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "source_word_offsets": [int(i)*(chunk_words-overlap_words) for i in top_idx],
        "bm25_top": [int(i) for i in bm25_rank[:top_k]],
        "dense_top": [int(i) for i in dense_rank[:top_k]],
    }


DEFAULT_STOPS = ["\n\n"]

SYSTEM_RAG = (
    "You are a precise reading assistant. The user gives you retrieved "
    "document fragments and a question. Answer using only those fragments. "
    "Respond ONLY in this format:\nAnswer: <your answer>"
)

# Variant for BABILong qa2 — same fact-tracking framing as A0 uses.
SYSTEM_RAG_BABILONG = (
    "You are reading retrieved fragments of a longer document where facts "
    "about character positions and movements are hidden among narrative prose. "
    "Track where each entity ends up by chaining facts. Output one word — the "
    "final location or object name. Respond ONLY in this format:\n"
    "Answer: <one word>"
)

# Variant for Banking77 ICL — classifier mode (no Answer: marker, the prompt
# already ends with 'Label:').
SYSTEM_RAG_ICL = (
    "You are a banking customer-support intent classifier. The user gives you "
    "retrieved query/label examples and a final query. Output ONLY the matching "
    "intent label (lowercase with underscores). No other text."
)

import re
_ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def run(
    episode,
    llm: LLMClient,
    top_k: int = 5,
    max_tokens: int = 96,
    source_order: bool = False,
) -> dict:
    """Dataset-agnostic RAG runner with system/user split."""
    haystack, question, gold = get_dataset_inputs(episode)
    chunks, meta = retrieve_chunks(haystack, question, top_k=top_k,
                                 source_order=source_order)
    retrieval_summary = " | ".join(c[:80].replace("\n", " ") for c in chunks)

    # Pick system prompt + user-message style based on dataset type.
    ep_type = type(episode).__name__
    if ep_type == "BabilongEpisode":
        system_prompt = SYSTEM_RAG_BABILONG
        user_msg = (
            "Document fragments (facts hidden in narrative):\n"
            + "\n---\n".join(chunks)
            + f"\n\nQuestion: {question}"
        )
    elif ep_type == "Banking77IclEpisode":
        system_prompt = SYSTEM_RAG_ICL
        # For ICL, the chunks ARE labelled examples. Format as ICL block.
        user_msg = (
            "Retrieved query/label examples:\n"
            + "\n---\n".join(chunks)
            + f"\n\nNow classify:\nQuery: {question}\nLabel:"
        )
        # ICL needs different stops — model emits just the label, no '\n\n'.
        stops = ["\n", "Query:"]
    else:
        system_prompt = SYSTEM_RAG
        user_msg = (
            "Document fragments:\n"
            + "\n---\n".join(chunks)
            + f"\n\nQuestion: {question}"
        )
    stops = locals().get("stops", DEFAULT_STOPS)
    r = llm.generate(
        user_msg,
        max_tokens=max_tokens,
        stop=stops,
        temperature=0.0,
        system_prompt=system_prompt,
    )
    extracted = _extract_answer(r.text)
    metric_value, metric_name = score_response(extracted + " " + r.text, gold, episode)
    intermediate = [
        IntermediateMessage(
            role="rag/retrieval",
            summary=retrieval_summary[:400],
            extra=meta,
        ).__dict__,
        IntermediateMessage(
            role="rag/llm",
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            duration_ms=r.wall_ms,
            summary=r.text[:200],
        ).__dict__,
    ]
    return {
        "output": r.text,
        "metric_value": metric_value,
        "metric_name": metric_name,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "wall_clock_ms": r.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": r.finish_reason,
    }


def run_niah(
    episode: NiahEpisode,
    llm: LLMClient,
    top_k: int = 5,
    max_tokens: int = 64,
) -> dict:
    chunks, meta = retrieve_chunks(episode.haystack_text, episode.question, top_k=top_k)
    retrieval_summary = " | ".join(c[:80].replace("\n", " ") for c in chunks)

    user_msg = (
        "Document fragments:\n"
        + "\n---\n".join(chunks)
        + f"\n\nQuestion: {episode.question}"
    )
    r = llm.generate(
        user_msg,
        max_tokens=max_tokens,
        stop=DEFAULT_STOPS,
        temperature=0.0,
        system_prompt=SYSTEM_RAG,
    )
    extracted = _extract_answer(r.text)
    metric = niah_scorer(r.text + " " + extracted, episode.value)

    intermediate = [
        IntermediateMessage(
            role="rag/retrieval",
            summary=retrieval_summary[:400],
            extra=meta,
        ).__dict__,
        IntermediateMessage(
            role="rag/llm",
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            duration_ms=r.wall_ms,
            summary=r.text[:200],
        ).__dict__,
    ]
    return {
        "output": r.text,
        "extracted": extracted,
        "metric_value": metric,
        "metric_name": "exact_match",
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "wall_clock_ms": r.wall_ms,
        "intermediate_messages": intermediate,
        "finish_reason": r.finish_reason,
    }
