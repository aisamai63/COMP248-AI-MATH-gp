"""
Shared LLM gateway for provider-agnostic calls.

This centralizes:
- provider initialization (OpenAI / Mistral / Gemini)
- retries + backoff
- timeout handling
- JSON-mode requests + self-healing JSON repair
- per-call telemetry into state['metadata']['llm_calls']
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import random
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from prototype.config import llm_config
from prototype.llm import LLMCallRecord, append_llm_call, usage_from_openai_response


@dataclass
class LLMResult:
    text: str
    latency_s: float
    attempts: int
    provider: str
    model: str
    response_obj: Any = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    total_tokens: Optional[int] = None


def _trust_env_for_http() -> bool:
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        value = (os.environ.get(name) or "").strip()
        if not value:
            continue
        try:
            parsed = urllib.parse.urlparse(value)
            host = (parsed.hostname or "").lower()
            port = parsed.port
            if host in {"127.0.0.1", "localhost"} and port == 9:
                return False
        except Exception:
            continue
    return True


def _extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)

    candidates = getattr(response, "candidates", None) or []
    parts = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(str(part_text))
    return "\n".join(parts)


def _extract_json(raw: str) -> Dict[str, Any]:
    """Extract JSON safely from model output, including fenced responses."""
    try:
        return json.loads(raw)
    except Exception:
        pass

    try:
        candidate = ast.literal_eval(raw)
        if isinstance(candidate, dict):
            return candidate
    except Exception:
        pass

    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    for block in code_blocks:
        try:
            return json.loads(block.strip())
        except Exception:
            try:
                candidate = ast.literal_eval(block.strip())
                if isinstance(candidate, dict):
                    return candidate
            except Exception:
                continue

    for match in re.finditer(r"\{[\s\S]*?\}", raw):
        candidate = match.group(0).strip()
        try:
            return json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

    greedy_match = re.search(r"\{[\s\S]*\}", raw)
    if greedy_match:
        candidate = greedy_match.group(0).strip()
        try:
            return json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    raise ValueError("Model did not return parseable JSON")


class LLMGateway:
    def __init__(self) -> None:
        self.provider = llm_config.LLM_PROVIDER
        self.ready = False
        self.init_error: str = ""
        self.openai_client = None
        self.gemini_model = None
        self.mistral_client = None
        self._initialize()

    @property
    def model_name(self) -> str:
        if self.provider == "openai":
            return llm_config.OPENAI_MODEL
        if self.provider == "gemini":
            return llm_config.GEMINI_MODEL
        return llm_config.MISTRAL_MODEL

    def _initialize(self) -> None:
        try:
            if self.provider == "gemini":
                if not llm_config.GEMINI_API_KEY:
                    self.init_error = "GEMINI_API_KEY missing"
                    return
                genai = importlib.import_module("google.generativeai")
                genai.configure(api_key=llm_config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(llm_config.GEMINI_MODEL)
                self.ready = True
                return

            if self.provider == "openai":
                if not llm_config.OPENAI_API_KEY:
                    self.init_error = "OPENAI_API_KEY missing"
                    return
                openai_module = importlib.import_module("openai")
                OpenAI = getattr(openai_module, "OpenAI")

                try:
                    http_client = None
                    if not _trust_env_for_http():
                        import httpx

                        http_client = httpx.Client(
                            timeout=llm_config.LLM_TIMEOUT_SECONDS,
                            trust_env=False,
                        )
                    self.openai_client = OpenAI(
                        api_key=llm_config.OPENAI_API_KEY,
                        timeout=llm_config.LLM_TIMEOUT_SECONDS,
                        max_retries=0,
                        http_client=http_client,
                    )
                except TypeError:
                    self.openai_client = OpenAI(api_key=llm_config.OPENAI_API_KEY)
                self.ready = True
                return

            # Default: Mistral
            if not llm_config.MISTRAL_API_KEY:
                self.init_error = "MISTRAL_API_KEY missing"
                return
            try:
                from mistralai import Mistral
            except ImportError:
                from mistralai.client import Mistral
            self.mistral_client = Mistral(api_key=llm_config.MISTRAL_API_KEY)
            self.ready = True
        except Exception as exc:
            self.ready = False
            self.init_error = str(exc)

    def _retry_sleep(self, attempt: int) -> None:
        base_sleep = float(llm_config.LLM_RETRY_BACKOFF_SECONDS)
        jitter = random.random() * 0.25
        time.sleep(base_sleep * (2**attempt) + jitter)

    def _call_once(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        json_object: bool,
        system: Optional[str],
    ) -> Tuple[str, Any]:
        if self.provider == "gemini" and self.gemini_model is not None:
            generation_config: Dict[str, Any] = {
                "temperature": float(temperature),
                "max_output_tokens": int(max_tokens),
            }
            if json_object:
                generation_config["response_mime_type"] = "application/json"
            try:
                response = self.gemini_model.generate_content(
                    prompt, generation_config=generation_config
                )
            except TypeError:
                # Older SDK: strip unsupported keys.
                generation_config.pop("response_mime_type", None)
                response = self.gemini_model.generate_content(
                    prompt, generation_config=generation_config
                )
            return _extract_gemini_text(response).strip(), response

        if self.provider == "openai" and self.openai_client is not None:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            create_kwargs: Dict[str, Any] = dict(
                model=llm_config.OPENAI_MODEL,
                messages=messages,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            if json_object:
                try:
                    response = self.openai_client.chat.completions.create(
                        **(create_kwargs | {"response_format": {"type": "json_object"}})
                    )
                except TypeError:
                    response = self.openai_client.chat.completions.create(**create_kwargs)
            else:
                response = self.openai_client.chat.completions.create(**create_kwargs)
            content = (response.choices[0].message.content or "").strip()
            return content, response

        if self.mistral_client is None:
            raise RuntimeError("LLM client not initialized")
        chat_obj = getattr(self.mistral_client, "chat", None)
        complete_fn = getattr(chat_obj, "complete", None) if chat_obj else None
        if not callable(complete_fn):
            raise AttributeError("Mistral client does not provide chat.complete()")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = complete_fn(
            model=llm_config.MISTRAL_MODEL,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        content = (response.choices[0].message.content or "").strip()
        return content, response

    def complete_text(
        self,
        state: Dict[str, Any],
        *,
        agent: str,
        purpose: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        json_object: bool = False,
        system: Optional[str] = None,
    ) -> str:
        if not self.ready:
            raise RuntimeError(self.init_error or "LLM gateway not ready")

        started = time.perf_counter()
        last_error: Optional[Exception] = None
        response_obj = None
        text = ""
        attempts = 0

        for attempt in range(max(1, int(llm_config.LLM_MAX_RETRIES) + 1)):
            attempts = attempt + 1
            try:
                text, response_obj = self._call_once(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_object=json_object,
                    system=system,
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                # Hard-fail on local connection refused (common when proxies are misconfigured).
                if "10061" in str(exc) or "actively refused" in str(exc).lower():
                    break
                if attempt >= int(llm_config.LLM_MAX_RETRIES):
                    break
                self._retry_sleep(attempt)

        latency_s = time.perf_counter() - started
        usage = (
            usage_from_openai_response(response_obj)
            if self.provider == "openai"
            else {"tokens_in": None, "tokens_out": None, "total_tokens": None}
        )

        if last_error is not None:
            append_llm_call(
                state,
                LLMCallRecord(
                    agent=agent,
                    purpose=purpose,
                    provider=self.provider,
                    model=self.model_name,
                    latency_s=float(latency_s),
                    attempts=attempts,
                    ok=False,
                    prompt_chars=len(prompt),
                    response_chars=len(text or ""),
                    tokens_in=usage.get("tokens_in"),
                    tokens_out=usage.get("tokens_out"),
                    total_tokens=usage.get("total_tokens"),
                    error_type=type(last_error).__name__,
                    error_message=str(last_error),
                ),
            )
            raise last_error

        append_llm_call(
            state,
            LLMCallRecord(
                agent=agent,
                purpose=purpose,
                provider=self.provider,
                model=self.model_name,
                latency_s=float(latency_s),
                attempts=attempts,
                ok=True,
                prompt_chars=len(prompt),
                response_chars=len(text or ""),
                tokens_in=usage.get("tokens_in"),
                tokens_out=usage.get("tokens_out"),
                total_tokens=usage.get("total_tokens"),
            ),
        )
        return text

    def complete_json(
        self,
        state: Dict[str, Any],
        *,
        agent: str,
        purpose: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema_hint: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns (parsed_json, meta) where meta includes parse_ok + repaired flags.
        """
        if not self.ready:
            raise RuntimeError(self.init_error or "LLM gateway not ready")

        raw = self.complete_text(
            state,
            agent=agent,
            purpose=purpose,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_object=True,
            system=system,
        )

        meta = {"parse_ok": False, "repaired": False, "raw_chars": len(raw or "")}
        try:
            parsed = _extract_json(raw)
            meta["parse_ok"] = True
            return parsed, meta
        except Exception:
            pass

        repair_prompt = (
            "Convert the text below into ONLY valid JSON."
            + ("\nSchema:\n" + schema_hint if schema_hint else "")
            + "\nNo markdown. No commentary. Output JSON only.\n\nTEXT:\n"
            + raw
        )
        repaired_text = self.complete_text(
            state,
            agent=agent,
            purpose=f"{purpose}_repair",
            prompt=repair_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            json_object=True,
            system=system,
        )
        meta["repaired"] = True
        parsed = _extract_json(repaired_text)
        meta["parse_ok"] = True
        return parsed, meta


_GATEWAY: Optional[LLMGateway] = None


def get_llm_gateway() -> LLMGateway:
    global _GATEWAY
    if _GATEWAY is None:
        _GATEWAY = LLMGateway()
    return _GATEWAY

