"""OpenRouter API adapter (free-tier models per v3 §2).

Supported models (free tier):
  * meta-llama/llama-3.3-70b-instruct:free       (128K ctx)
  * deepseek/deepseek-chat:free                  (64K ctx)
  * qwen/qwen-2.5-72b-instruct:free              (32K ctx)

OpenRouter uses the OpenAI-compatible Chat Completions API. We use the raw
HTTP path to avoid a dependency on the `openai` package and to match the
streaming-disabled style of the LlamaCppClient.

Rate limit: ~10K req/day shared across free-tier models. We add 1.5s sleep
between calls as a soft pacing measure. On HTTP 429 we exponential-backoff
to 60s.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from llm.base import GenerateResult


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """Synchronous OpenRouter chat-completions client (free tier)."""

    # Friendly name → OpenRouter model_id mapping.
    # Paid endpoints used after free tier proved permanently rate-limited upstream
    # (Venice provider returns 429 on every request without prepaid credits).
    # MODELS dict is now derived from the central registry (single source of truth).
    # Adding `qwen3-next` for Phase 2 + free-tier aliases for backward compat.
    from llm.model_registry import MODEL_REGISTRY as _REG
    MODELS = {name: m["openrouter_id"] for name, m in _REG.items()}
    # Plus free-tier aliases (not in registry — registry is for paid main grid)
    MODELS.update({
        "llama-3.3-70b:free": "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek-v4:free":   "deepseek/deepseek-v4-flash:free",
        "qwen3-coder:free":   "qwen/qwen3-coder:free",
        "gpt-oss-120b:free":  "openai/gpt-oss-120b:free",
        "deepseek-chat":      "deepseek/deepseek-chat:free",
        "qwen-2.5-72b:free":  "qwen/qwen-2.5-72b-instruct:free",
    })

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        pace_sleep_s: float = 1.5,
        max_backoff_s: float = 60.0,
        max_retries: int = 4,
        timeout_s: float = 180.0,
        site_url: str = "https://github.com/MortiferumUrsus/kursach",
        site_name: str = "kursach-experiment",
    ):
        self.model_id = self.MODELS.get(model, model)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Add to .env or export env var."
            )
        self.pace_sleep_s = pace_sleep_s
        self.max_backoff_s = max_backoff_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.site_url = site_url
        self.site_name = site_name
        self._last_call_t = 0.0

    def _wait_pace(self) -> None:
        delta = time.time() - self._last_call_t
        if delta < self.pace_sleep_s:
            time.sleep(self.pace_sleep_s - delta)
        self._last_call_t = time.time()

    def generate(
        self,
        prompt: str,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> GenerateResult:
        self._wait_pace()
        # Proper chat split: system holds the "how to answer" instructions,
        # user holds the document + question. Without system, chat-tuned models
        # (GPT-4o-mini, Gemini, Claude) sometimes refuse on long completion-style
        # prompts that look like jailbreak attempts.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "stream": False,
        }
        if stop:
            body["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(OPENROUTER_URL, data=payload, headers=headers)

        backoff = 1.0
        last_err = ""
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    data = json.loads(resp.read())
                wall_ms = (time.time() - t0) * 1000.0
                choices = data.get("choices") or []
                text = ""
                finish = "stop"
                if choices:
                    msg = choices[0].get("message") or {}
                    text = (msg.get("content") or "").strip()
                    fr = choices[0].get("finish_reason") or "stop"
                    if fr == "length":
                        finish = "length"
                    elif fr in ("content_filter", "function_call"):
                        finish = "parse_error"
                usage = data.get("usage") or {}
                return GenerateResult(
                    text=text,
                    input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    output_tokens=int(usage.get("completion_tokens", 0) or 0),
                    wall_ms=wall_ms,
                    finish_reason=finish,
                    extra={"attempt": attempt + 1, "model": self.model_id},
                )
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
                if e.code == 429 or 500 <= e.code < 600:
                    time.sleep(min(backoff, self.max_backoff_s))
                    backoff *= 2
                    continue
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(min(backoff, self.max_backoff_s))
                backoff *= 2
        return GenerateResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            wall_ms=0.0,
            finish_reason="parse_error",
            extra={"error": last_err, "attempts": self.max_retries,
                   "model": self.model_id},
        )

    def count_tokens(self, text: str) -> int:
        # OpenRouter doesn't expose a tokenizer endpoint; use char/4 heuristic.
        return max(1, len(text) // 4)
