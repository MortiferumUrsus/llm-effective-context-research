"""Episode logger + dataclass — JSONL append-only per the PROMPT §7 schema."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class IntermediateMessage:
    role: str
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    summary: str = ""
    relevance: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class Episode:
    """Schema mirrors §7 of PROMPT_FOR_CLAUDE_CODE.md."""

    run_id: str  # uuid4 hex
    config_name: str
    seed: int
    model: str  # "Qwen2.5-7B-Instruct"
    model_version: str  # "Q4_K_M-bartowski-2026-05-11" + sha256[:8]
    quantization: str  # "Q4_K_M"
    procedure_A: str  # "A0_baseline" | "A_rag" | "A_singleagent" | "A_subagents"
    dataset_t: str  # "NIAH" | "RULER_multikey" | "HotPotQA" | "NoLiMa"
    length_N: int
    input_text_hash: str  # sha256[:16] of haystack
    output: str
    input_tokens: int
    output_tokens: int
    intermediate_messages: list[dict]
    metric_value: float
    metric_name: str  # "exact_match" | "f1"
    wall_clock_ms: float
    needle_position_bin: Optional[int] = None
    finish_reason: str = "stop"
    git_commit: str = ""
    timestamp: str = ""
    extra: dict = field(default_factory=dict)


def get_git_commit() -> str:
    """Return current git HEAD short hash. Falls back to 'no-git' on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip()
    except Exception:
        return "no-git"


def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_text(text: str, n_hex: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n_hex]


class JsonlWriter:
    """Append-only JSONL writer with resume-skip support.

    On construction it indexes existing `(run_id)` tuples in the file so the
    orchestrator can skip already-done episodes on restart.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._done_keys: set[tuple] = set()
        self._scan_existing()

    def _scan_existing(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Key by target_words (canonical config-level length) for stable
                # resume across changes to _length_tokens_for_ctx mapping.
                target_words = int(
                    rec.get("extra", {}).get("target_words")
                    or rec.get("length_N", -1)
                )
                key = (
                    rec.get("config_name"),
                    rec.get("procedure_A"),
                    rec.get("dataset_t"),
                    target_words,
                    int(rec.get("seed", -1)),
                    int(rec.get("extra", {}).get("episode_idx", -1)),
                )
                self._done_keys.add(key)
            except Exception:
                continue

    def is_done(
        self,
        config_name: str,
        procedure: str,
        dataset: str,
        target_words: int,
        seed: int,
        episode_idx: int,
    ) -> bool:
        return (config_name, procedure, dataset, target_words, seed, episode_idx) in self._done_keys

    def write(self, ep: Episode) -> None:
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(asdict(ep), ensure_ascii=False) + "\n")
        self._fh.flush()
        # Record in done-set for resume — key by target_words (in extra), not length_N
        target_words = int(ep.extra.get("target_words", ep.length_N))
        key = (
            ep.config_name,
            ep.procedure_A,
            ep.dataset_t,
            target_words,
            ep.seed,
            int(ep.extra.get("episode_idx", -1)),
        )
        self._done_keys.add(key)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def new_run_id() -> str:
    return uuid.uuid4().hex
