"""
LLM-based Reflective Agent.

Evaluates summary quality using prompt-based LLM scoring across:
- factual_correctness
- completeness
- relevance

Returns structured JSON-compatible metrics for planner loop integration.
"""

import json
import importlib
import logging
import re
import time
from typing import Any, Dict, List

from prototype.agents.base import BaseAgent
from prototype.workflow.state import WorkflowState
from prototype.config import llm_config, reflection_config, runtime_config

logger = logging.getLogger(__name__)


class ReflectiveAgent(BaseAgent):
    """LLM-driven evaluator for summary quality and retry decisions."""

    name = "Reflector"

    def __init__(self, use_llm: bool = True):
        super().__init__()
        self.use_llm = use_llm and (not runtime_config.FAST_MODE)
        self.llm_ready = False
        self.provider = llm_config.LLM_PROVIDER
        self.mistral_client = None
        self.openai_client = None
        self.gemini_model = None

        if self.use_llm:
            self._initialize_llm_client()
        elif runtime_config.FAST_MODE:
            self.logger.info("FAST_MODE enabled: reflector LLM disabled")

    def _initialize_llm_client(self) -> None:
        """Initialize LLM client based on configured provider."""
        try:
            if self.provider == "gemini":
                if not llm_config.GEMINI_API_KEY:
                    self.logger.warning(
                        "GEMINI_API_KEY missing; reflector uses fallback."
                    )
                    return
                genai = importlib.import_module("google.generativeai")

                genai.configure(api_key=llm_config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(llm_config.GEMINI_MODEL)
                self.llm_ready = True
                self.logger.info("LLM reflection enabled (Gemini)")
                return

            if self.provider == "openai":
                if not llm_config.OPENAI_API_KEY:
                    self.logger.warning(
                        "OPENAI_API_KEY missing; reflector uses fallback."
                    )
                    return
                openai_module = importlib.import_module("openai")
                OpenAI = getattr(openai_module, "OpenAI")

                self.openai_client = OpenAI(api_key=llm_config.OPENAI_API_KEY)
                self.llm_ready = True
                self.logger.info("LLM reflection enabled (OpenAI)")
                return

            if not llm_config.MISTRAL_API_KEY:
                self.logger.warning("MISTRAL_API_KEY missing; reflector uses fallback.")
                return
            from mistralai.client import Mistral

            self.mistral_client = Mistral(api_key=llm_config.MISTRAL_API_KEY)
            self.llm_ready = True
            self.logger.info("LLM reflection enabled (Mistral)")
        except Exception as exc:
            self.logger.warning(f"Failed to init {self.provider} for reflection: {exc}")
            self.llm_ready = False

    def run(self, state: WorkflowState) -> WorkflowState:
        """Evaluate current summary and update reflection_metrics in state."""
        try:
            start_time = time.time()

            query = state.get("user_query", "")
            summary = state.get("summary", "")
            docs = state.get("retrieved_docs", [])

            self._log_step(
                "Evaluating with LLM",
                {
                    "summary_words": len(summary.split()),
                    "num_docs": len(docs),
                    "llm_ready": self.llm_ready,
                },
            )

            if self.llm_ready:
                try:
                    metrics = self._evaluate_with_llm(
                        query=query, summary=summary, docs=docs
                    )
                except Exception as exc:
                    self.logger.warning(
                        "LLM reflection failed; using fallback evaluation. Reason: %s",
                        exc,
                    )
                    metrics = self._fallback_evaluation(
                        query=query, summary=summary, docs=docs
                    )
                    metrics["evaluation_source"] = "fallback_llm_parse_error"
            else:
                metrics = self._fallback_evaluation(
                    query=query, summary=summary, docs=docs
                )

            # Planner loop integration: single confidence and retry signal.
            confidence = float(metrics.get("confidence", 0.0))
            should_retry = confidence < reflection_config.CONFIDENCE_THRESHOLD
            metrics["should_retry"] = should_retry

            # Compatibility with existing UI code that expects notes list.
            feedback_text = metrics.get("feedback_text", "")
            metrics["notes"] = (
                [feedback_text] if feedback_text else ["No feedback provided."]
            )

            state["reflection_metrics"] = metrics
            state["metadata"]["timestamps"]["reflector"] = time.time() - start_time

            self._log_step(
                "Reflection complete",
                {
                    "confidence": f"{confidence:.2f}",
                    "should_retry": should_retry,
                },
            )

            return state
        except Exception as exc:
            return self._handle_error(state, exc)

    def _evaluate_with_llm(
        self, query: str, summary: str, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prompt-based evaluation with strict JSON output contract."""
        prompt = self._build_evaluation_prompt(query=query, summary=summary, docs=docs)

        llm_started = time.perf_counter()
        if self.provider == "gemini" and self.gemini_model is not None:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": llm_config.MISTRAL_MAX_TOKENS_REFLECTION,
                },
            )
            raw = self._extract_gemini_text(response).strip()
        elif self.provider == "openai" and self.openai_client is not None:
            response = self.openai_client.chat.completions.create(
                model=llm_config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=llm_config.MISTRAL_MAX_TOKENS_REFLECTION,
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
        else:
            response = self.mistral_client.chat.complete(
                model=llm_config.MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=llm_config.MISTRAL_MAX_TOKENS_REFLECTION,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()

        llm_elapsed = time.perf_counter() - llm_started
        self.logger.info(
            "Timing | reflector_llm_call | provider=%s | %.3fs",
            self.provider,
            llm_elapsed,
        )

        parsed = self._extract_json(raw)
        validated = self._validate_schema(parsed)

        # Weighted score for easy planner integration.
        confidence = self._compute_confidence(
            factual_correctness=validated["factual_correctness"],
            completeness=validated["completeness"],
            relevance=validated["relevance"],
        )

        return {
            "factual_correctness": validated["factual_correctness"],
            "completeness": validated["completeness"],
            "relevance": validated["relevance"],
            "confidence": confidence,
            "feedback_text": validated["feedback_text"],
            "evaluation_source": "llm",
        }

    @staticmethod
    def _extract_gemini_text(response) -> str:
        """Extract text robustly from Gemini SDK response."""
        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts)

    def _build_evaluation_prompt(
        self, query: str, summary: str, docs: List[Dict[str, Any]]
    ) -> str:
        """Builds a strict, JSON-only evaluation prompt."""
        context_blocks = []
        for idx, doc in enumerate(docs[:5], 1):
            title = doc.get("title", f"Doc {idx}")
            text = doc.get("excerpt") or doc.get("text", "")
            context_blocks.append(f"[Document {idx}] {title}\n{text[:900]}")

        docs_text = (
            "\n\n".join(context_blocks) if context_blocks else "[No documents provided]"
        )

        return (
            "You are a strict evaluator for RAG responses. Evaluate ONLY using the provided documents.\n"
            "Task: score the candidate summary on three axes from 0.0 to 1.0:\n"
            "1) factual_correctness: claims are grounded in source docs\n"
            "2) completeness: answer addresses the user query sufficiently\n"
            "3) relevance: answer stays focused on the query\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  "factual_correctness": <float 0..1>,\n'
            '  "completeness": <float 0..1>,\n'
            '  "relevance": <float 0..1>,\n'
            '  "feedback_text": "<short actionable feedback, max 2 sentences>"\n'
            "}\n"
            "No markdown. No explanation outside JSON.\n\n"
            f"USER_QUERY:\n{query}\n\n"
            f"CANDIDATE_SUMMARY:\n{summary}\n\n"
            f"SOURCE_DOCUMENTS:\n{docs_text}"
        )

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        """Extract JSON safely from model output, including fenced responses."""
        # Try direct parse first.
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Try fenced code blocks first.
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
        for block in code_blocks:
            try:
                return json.loads(block.strip())
            except Exception:
                continue

        # Try embedded JSON objects (non-greedy and then greedy fallback).
        for match in re.finditer(r"\{[\s\S]*?\}", raw):
            candidate = match.group(0).strip()
            try:
                return json.loads(candidate)
            except Exception:
                continue

        greedy_match = re.search(r"\{[\s\S]*\}", raw)
        if greedy_match:
            try:
                return json.loads(greedy_match.group(0).strip())
            except Exception:
                pass

        raise ValueError("LLM evaluator did not return parseable JSON")

    def _validate_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output shape and clamp scores to [0, 1]."""
        required = [
            "factual_correctness",
            "completeness",
            "relevance",
            "feedback_text",
        ]
        for key in required:
            if key not in payload:
                raise ValueError(f"Missing field in evaluator output: {key}")

        def clamp(value: Any) -> float:
            try:
                num = float(value)
            except Exception as exc:
                raise ValueError(
                    f"Non-numeric score in evaluator output: {value}"
                ) from exc
            return max(0.0, min(1.0, num))

        return {
            "factual_correctness": clamp(payload["factual_correctness"]),
            "completeness": clamp(payload["completeness"]),
            "relevance": clamp(payload["relevance"]),
            "feedback_text": str(payload["feedback_text"]).strip(),
        }

    def _compute_confidence(
        self, factual_correctness: float, completeness: float, relevance: float
    ) -> float:
        """Compute single confidence score used by planner retry loop."""
        score = (
            reflection_config.FACTUAL_CORRECTNESS_WEIGHT * factual_correctness
            + reflection_config.COMPLETENESS_WEIGHT * completeness
            + reflection_config.RELEVANCE_WEIGHT * relevance
        )
        return max(0.0, min(1.0, score))

    def _fallback_evaluation(
        self, query: str, summary: str, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fail-safe minimal evaluation when LLM is unavailable."""
        # Lightweight conservative fallback for runtime safety.
        summary_words = len(summary.split())
        has_docs = 1.0 if docs else 0.0
        relevance = (
            1.0
            if any(term in summary.lower() for term in query.lower().split()[:3])
            else 0.5
        )

        factual_correctness = 0.55 * has_docs
        completeness = 0.7 if summary_words >= 30 else 0.45
        confidence = self._compute_confidence(
            factual_correctness, completeness, relevance
        )

        return {
            "factual_correctness": factual_correctness,
            "completeness": completeness,
            "relevance": relevance,
            "confidence": confidence,
            "feedback_text": "LLM evaluator unavailable; used fallback scoring. Consider checking API configuration.",
            "evaluation_source": "fallback",
        }


# Example I/O (for integration tests)
# Input:
#   query="What is the quadratic formula?"
#   summary="The quadratic formula solves ax^2+bx+c=0 using x = (-b ± sqrt(b^2-4ac))/(2a)."
# Output (state["reflection_metrics"]):
# {
#   "factual_correctness": 0.92,
#   "completeness": 0.86,
#   "relevance": 0.95,
#   "confidence": 0.91,
#   "feedback_text": "Strongly grounded and relevant. Add one short constraint note (a != 0) for completeness.",
#   "evaluation_source": "llm",
#   "should_retry": false,
#   "notes": ["Strongly grounded and relevant. Add one short constraint note (a != 0) for completeness."]
# }
