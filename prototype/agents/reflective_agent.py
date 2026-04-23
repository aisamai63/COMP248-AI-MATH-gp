"""
LLM-based Reflective Agent.

Evaluates summary quality using prompt-based LLM scoring across:
- factual_correctness
- completeness
- relevance

Returns structured JSON-compatible metrics for planner loop integration.
"""

import logging
import re
import time
from typing import Any, Dict, List

from prototype.agents.base import BaseAgent
from prototype.llm_gateway import get_llm_gateway
from prototype.workflow.state import WorkflowState
from prototype.config import llm_config, reflection_config, runtime_config

logger = logging.getLogger(__name__)


class ReflectiveAgent(BaseAgent):
    """LLM-driven evaluator for summary quality and retry decisions."""

    name = "Reflector"

    def __init__(self, use_llm: bool = True):
        super().__init__()
        self.use_llm = use_llm and (not runtime_config.FAST_MODE)
        self.gateway = get_llm_gateway() if self.use_llm else None
        self.llm_ready = bool(self.gateway and self.gateway.ready)
        self.provider = llm_config.LLM_PROVIDER
        self.init_error: str = ""
        if self.use_llm and self.gateway and not self.gateway.ready:
            self.init_error = self.gateway.init_error or "LLM gateway init failed"
        if runtime_config.FAST_MODE:
            self.logger.info("FAST_MODE enabled: reflector LLM disabled")

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
                        state=state, query=query, summary=summary, docs=docs
                    )
                except Exception as exc:
                    self.logger.warning(
                        "LLM reflection failed; using fallback evaluation. Reason: %s",
                        exc,
                    )
                    metrics = self._fallback_evaluation(
                        query=query, summary=summary, docs=docs
                    )
                    metrics["evaluation_source"] = "fallback_llm_error"
                    metrics["llm_error_type"] = type(exc).__name__
                    metrics["llm_error"] = str(exc)
            else:
                metrics = self._fallback_evaluation(
                    query=query, summary=summary, docs=docs
                )
                if self.init_error:
                    metrics["evaluation_source"] = "fallback_init_error"
                    metrics["feedback_text"] = (
                        f"LLM evaluator unavailable ({self.init_error}); used fallback scoring."
                    )

            # Planner loop integration: single confidence and retry signal.
            confidence_raw = float(metrics.get("confidence", 0.0))
            doc_count = len(docs) if isinstance(docs, list) else 0
            similarities = []
            if isinstance(docs, list):
                for doc in docs[:5]:
                    if not isinstance(doc, dict):
                        continue
                    sim = doc.get("similarity", None)
                    try:
                        if sim is not None:
                            similarities.append(float(sim))
                    except Exception:
                        continue
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            doc_count_factor = min(1.0, doc_count / 5.0) if doc_count else 0.0
            retrieval_quality = max(
                0.0, min(1.0, 0.5 * doc_count_factor + 0.5 * max(0.0, min(1.0, avg_similarity)))
            )
            # Confidence should reflect both evaluation scores and evidence quality.
            confidence = max(0.0, min(1.0, confidence_raw * (0.7 + 0.3 * retrieval_quality)))
            metrics["confidence_raw"] = confidence_raw
            metrics["confidence"] = confidence
            metrics["retrieval_quality"] = retrieval_quality
            metrics["avg_similarity_top5"] = avg_similarity
            metrics["doc_count"] = doc_count
            evaluation_source = str(metrics.get("evaluation_source", "") or "")
            iteration = int(state.get("iteration_count", 0) or 0)
            should_retry = (
                confidence < reflection_config.CONFIDENCE_THRESHOLD
                and iteration < int(reflection_config.MAX_ITERATIONS)
            )
            # Reliability: if reflection is not actually LLM-based, do not loop.
            if evaluation_source != "llm":
                should_retry = False
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
        self, state: WorkflowState, query: str, summary: str, docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prompt-based evaluation with strict JSON output contract."""
        if self.gateway is None:
            raise RuntimeError("LLM gateway not configured")

        prompt = self._build_evaluation_prompt(query=query, summary=summary, docs=docs)
        schema_hint = (
            "{\n"
            '  "factual_correctness": <float 0..1>,\n'
            '  "completeness": <float 0..1>,\n'
            '  "relevance": <float 0..1>\n'
            "}"
        )
        system = (
            "Return only valid JSON matching the requested schema. "
            "Do not include markdown, code fences, or commentary."
        )
        parsed, meta = self.gateway.complete_json(
            state,
            agent=self.name,
            purpose="reflect",
            prompt=prompt,
            max_tokens=llm_config.MISTRAL_MAX_TOKENS_REFLECTION,
            temperature=0.0,
            schema_hint=schema_hint,
            system=system,
        )
        validated = self._validate_schema(parsed)

        confidence = self._compute_confidence(
            factual_correctness=validated["factual_correctness"],
            completeness=validated["completeness"],
            relevance=validated["relevance"],
        )
        feedback_text = self._build_feedback_text(
            factual_correctness=validated["factual_correctness"],
            completeness=validated["completeness"],
            relevance=validated["relevance"],
        )

        return {
            "factual_correctness": validated["factual_correctness"],
            "completeness": validated["completeness"],
            "relevance": validated["relevance"],
            "confidence": confidence,
            "feedback_text": feedback_text,
            "evaluation_source": "llm",
            "parse_ok": bool(meta.get("parse_ok")),
            "repaired": bool(meta.get("repaired")),
            "model": self.gateway.model_name,
            "provider": self.gateway.provider,
        }

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

        # Avoid LangChain templating here because literal JSON braces `{}` can be
        # misinterpreted as template variables. Build the final prompt string directly.
        return (
            "You are a strict evaluator for RAG responses. Evaluate ONLY using the provided documents.\n"
            "Task: score the candidate summary on three axes from 0.0 to 1.0:\n"
            "1) factual_correctness: claims are grounded in source docs\n"
            "2) completeness: answer addresses the user query sufficiently\n"
            "3) relevance: answer stays focused on the query\n\n"
            "Return ONLY valid JSON with this exact schema:\n"
            "{\n"
            '  \"factual_correctness\": <float 0..1>,\n'
            '  \"completeness\": <float 0..1>,\n'
            '  \"relevance\": <float 0..1>\n'
            "}\n"
            "No markdown. No explanation outside JSON.\n"
            "DO NOT OMIT ANY FIELD. If unsure, set the value to 0.0. Always include all three fields.\n"
            "If you cannot score, set all values to 0.0.\n\n"
            f"USER_QUERY:\n{query}\n\n"
            f"CANDIDATE_SUMMARY:\n{summary}\n\n"
            f"SOURCE_DOCUMENTS:\n{docs_text}"
        )

    def _validate_schema(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate output shape and clamp scores to [0, 1]. Fill missing fields with 0.0 and log a warning."""
        required = ["factual_correctness", "completeness", "relevance"]

        # Normalize common variants from different model styles.
        if payload and any(k not in payload for k in required):
            normalized = {}

            def norm_key(name: str) -> str:
                return re.sub(r"[^a-z0-9]+", "", str(name).lower())

            for k, v in payload.items():
                normalized[norm_key(k)] = v

            for key in required:
                if key in payload:
                    continue
                alias = norm_key(key)
                if alias in normalized:
                    payload[key] = normalized[alias]

        # Fill missing fields with 0.0 and log a warning
        for key in required:
            if key not in payload:
                logger.warning(
                    f"LLM evaluator output missing field '{key}'. Setting to 0.0. Full output: {payload}"
                )
                payload[key] = 0.0

        def clamp(value: Any) -> float:
            try:
                num = float(value)
            except Exception as exc:
                logger.warning(
                    f"Non-numeric score in evaluator output: {value}. Setting to 0.0. Full output: {payload}"
                )
                num = 0.0
            return max(0.0, min(1.0, num))

        return {
            "factual_correctness": clamp(payload["factual_correctness"]),
            "completeness": clamp(payload["completeness"]),
            "relevance": clamp(payload["relevance"]),
        }

    def _build_feedback_text(
        self, factual_correctness: float, completeness: float, relevance: float
    ) -> str:
        """Create concise feedback text from numeric reflection scores."""
        notes = []
        if factual_correctness < 0.8:
            notes.append("improve factual grounding")
        if completeness < 0.8:
            notes.append("add missing details")
        if relevance < 0.8:
            notes.append("stay closer to the question")

        if not notes:
            return "Strong reflection scores. The answer is grounded, complete, and relevant."

        return (
            "Consider to "
            + ", ".join(notes[:-1])
            + (f", and {notes[-1]}" if len(notes) > 1 else notes[0])
            + "."
        )

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        """Extract simple content words for lightweight alignment checks."""
        stopwords = {
            "about",
            "after",
            "also",
            "and",
            "are",
            "can",
            "for",
            "from",
            "have",
            "how",
            "into",
            "that",
            "the",
            "their",
            "this",
            "what",
            "when",
            "where",
            "which",
            "with",
            "would",
            "your",
            "is",
            "it",
            "of",
            "on",
            "to",
            "a",
            "an",
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {token for token in tokens if len(token) > 2 and token not in stopwords}

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
        summary_words = len(summary.split())

        query_terms = self._content_terms(query)
        summary_terms = self._content_terms(summary)
        doc_terms = self._content_terms(
            " ".join(
                f"{doc.get('title', '')} {doc.get('excerpt') or doc.get('text', '')}"
                for doc in docs[:5]
                if isinstance(doc, dict)
            )
        )

        query_alignment = 0.0
        evidence_alignment = 0.0
        if query_terms:
            query_alignment = len(query_terms & summary_terms) / len(query_terms)
            evidence_alignment = len(query_terms & doc_terms) / len(query_terms)

        overlap = max(query_alignment, evidence_alignment)

        factual_correctness = 0.15 + 0.55 * evidence_alignment
        completeness = 0.20 + 0.50 * overlap + min(summary_words / 120.0, 0.15)
        relevance = 0.15 + 0.70 * query_alignment

        if len(docs) > 1 and overlap < 0.5:
            factual_correctness *= 0.75
            completeness *= 0.75
            relevance *= 0.75

        if not summary.strip():
            factual_correctness = 0.0
            completeness = 0.0
            relevance = 0.0

        factual_correctness = max(0.0, min(1.0, factual_correctness))
        completeness = max(0.0, min(1.0, completeness))
        relevance = max(0.0, min(1.0, relevance))
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
