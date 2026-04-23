"""
LLM utilities for the Math Inquiries prototype.

Goal: make LLM calls observable, retryable, and debuggable across agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class LLMCallRecord:
    agent: str
    purpose: str
    provider: str
    model: str
    latency_s: float
    attempts: int
    ok: bool
    prompt_chars: int
    response_chars: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    total_tokens: Optional[int] = None
    parse_ok: Optional[bool] = None
    repaired: Optional[bool] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timestamp_unix: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def usage_from_openai_response(response: Any) -> Dict[str, Optional[int]]:
    """
    Extract usage from OpenAI SDK responses.

    Returns keys: tokens_in, tokens_out, total_tokens (all optional).
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"tokens_in": None, "tokens_out": None, "total_tokens": None}

    # Newer SDKs: usage.prompt_tokens/completion_tokens/total_tokens
    tokens_in = _safe_int(getattr(usage, "prompt_tokens", None))
    tokens_out = _safe_int(getattr(usage, "completion_tokens", None))
    total_tokens = _safe_int(getattr(usage, "total_tokens", None))
    return {"tokens_in": tokens_in, "tokens_out": tokens_out, "total_tokens": total_tokens}


def append_llm_call(state: Dict[str, Any], record: LLMCallRecord) -> None:
    """Append a single LLM call record to state['metadata']['llm_calls']."""
    metadata = state.setdefault("metadata", {})
    calls = metadata.setdefault("llm_calls", [])
    if isinstance(calls, list):
        record.timestamp_unix = time.time()
        calls.append(record.to_dict())

