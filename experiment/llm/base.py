"""Abstract LLM client interface.

All concrete clients (llama.cpp, Gemini, ...) must implement `LLMClient` and
return `GenerateResult` objects. The orchestrator never calls a client directly
without going through this interface.

Grammar parameter is intentionally NOT exposed here — measurement runs forbid
GBNF (see locked design decisions). Internal Reader→controller calls that need
structured metadata can construct prompts that elicit `KEY: value` lines and
parse them with regex on the calling side.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional


@dataclass
class GenerateResult:
    text: str
    input_tokens: int
    output_tokens: int
    wall_ms: float
    finish_reason: str  # "stop", "length", "timeout", "parse_error"
    extra: dict = field(default_factory=dict)


class LLMClient(Protocol):
    """All measurement-time LLM access goes through this interface."""

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerateResult:
        ...

    def count_tokens(self, text: str) -> int:
        ...
