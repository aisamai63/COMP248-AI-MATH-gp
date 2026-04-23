"""
Summarizer Agent: Generates summaries using LLM.

Role:
  - Receives query + documents
  - Uses LLM to generate focused summary
  - Returns compact answer

Flow:
  Input: user_query, retrieved_docs
  Step 1: Build prompt with query + doc excerpts
  Step 2: Call Mistral LLM
  Step 3: Return summary
  Output: summary
"""

import logging
import time
import re
from typing import Optional
from prototype.agents.base import BaseAgent
from prototype.llm_gateway import get_llm_gateway
from prototype.workflow.state import WorkflowState
from prototype.config import llm_config, runtime_config

logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """
    Summarizer agent: generates LLM-based summaries of retrieved documents.

    Uses Mistral API to create concise, query-focused summaries.
    Falls back to naive summarization if LLM is unavailable.
    """

    name = "Summarizer"

    def __init__(self, use_llm: bool = True):
        """
        Initialize Summarizer agent.

        Args:
            use_llm: Whether to use LLM (True) or fallback (False)
        """
        super().__init__()
        # FAST_MODE prioritizes latency over generation quality.
        self.use_llm = use_llm and (not runtime_config.FAST_MODE)
        self.gateway = get_llm_gateway() if self.use_llm else None
        self.provider = llm_config.LLM_PROVIDER
        self.init_error: Optional[str] = None
        self.llm_ready = bool(self.gateway and self.gateway.ready)
        if self.use_llm and self.gateway and not self.gateway.ready:
            self.init_error = self.gateway.init_error or "LLM gateway init failed"
        if runtime_config.FAST_MODE:
            self.logger.info("FAST_MODE enabled: summarizer LLM disabled")

    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute summarization: generate summary from documents.

        Args:
            state: Workflow state with retrieved_docs and user_query

        Returns:
            Modified state with summary populated
        """
        try:
            start_time = time.time()
            metadata = state.setdefault("metadata", {})
            timestamps = metadata.setdefault("timestamps", {})
            decisions = metadata.setdefault("decisions", {})

            docs = state["retrieved_docs"]
            query = state["user_query"]

            if not docs:
                state["summary"] = "[No documents retrieved; cannot generate summary]"
                decisions["summarizer"] = {
                    "mode": "fallback",
                    "reason": "no_documents",
                    "provider": self.provider,
                }
                timestamps["summarizer"] = time.time() - start_time
                self.logger.warning("No documents to summarize")
                return state

            self._log_step(
                "Generating summary", {"num_docs": len(docs), "query": query}
            )

            # Choose summarization method
            if self.use_llm and self.llm_ready:
                summary = self._summarize_with_llm(state, query, docs)
            else:
                self.logger.info("Using fallback summarization (no LLM)")
                summary = self._summarize_fallback(query, docs)
                fallback_reason = "llm_disabled_or_unavailable"
                if self.use_llm and not self.llm_ready and self.init_error:
                    fallback_reason = f"llm_init_error: {self.init_error}"
                decisions["summarizer"] = {
                    "mode": "fallback",
                    "reason": fallback_reason,
                    "provider": self.provider,
                }

            state["summary"] = summary

            # Log execution time
            elapsed = time.time() - start_time
            timestamps["summarizer"] = elapsed

            decisions.setdefault(
                "summarizer",
                {
                    "mode": "llm" if self.use_llm and self.llm_ready else "fallback",
                    "provider": self.provider,
                },
            )

            self._log_step("Summary generated", {"length": len(summary.split())})

            return state

        except Exception as e:
            return self._handle_error(state, e)

    def _summarize_with_llm(self, state: WorkflowState, query: str, docs: list) -> str:
        """
        Summarize using Mistral LLM.

        Args:
            query: User query
            docs: Retrieved documents

        Returns:
            LLM-generated summary
        """
        try:
            if self.gateway is None:
                raise RuntimeError("LLM gateway not configured")

            prompt = self._build_summary_prompt(state, query, docs)
            planner_decision = (
                state.get("metadata", {}).get("decisions", {}).get("planner", {}) or {}
            )
            query_type = str(planner_decision.get("query_type", "") or "").strip().lower()

            max_tokens = int(llm_config.MISTRAL_MAX_TOKENS_DEFAULT)
            # Avoid truncation for typical classroom explanations.
            if query_type in {"definition", "concept", "research"}:
                max_tokens = max(max_tokens, 160)
            elif query_type in {"calculation", "proof"}:
                max_tokens = max(max_tokens, 240)

            content = self.gateway.complete_text(
                state,
                agent=self.name,
                purpose="summarize",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=llm_config.MISTRAL_TEMPERATURE,
            )
            state.setdefault("metadata", {}).setdefault("decisions", {})[
                "summarizer"
            ] = {
                "mode": "llm",
                "provider": self.gateway.provider,
                "model": self.gateway.model_name,
                "query_type": query_type,
                "max_tokens": max_tokens,
            }
            return self._clean_answer((content or "").strip(), query_type=query_type)

        except Exception as e:
            self.logger.error(f"LLM summarization failed: {e}")
            fallback_summary = self._summarize_fallback(query, docs)
            state.setdefault("metadata", {}).setdefault("decisions", {})[
                "summarizer"
            ] = {
                "mode": "fallback",
                "reason": f"llm_error: {e}",
                "provider": self.provider,
            }
            return fallback_summary

    @staticmethod
    def _clean_answer(text: str, *, query_type: str) -> str:
        """Light post-processing to keep answers unified, friendly, and readable."""
        if not text:
            return ""

        # Normalize whitespace.
        text = text.replace("\r\n", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        def canon(s: str) -> str:
            s = s.strip().lower()
            s = re.sub(r"\\\(|\\\)", "", s)
            s = re.sub(r"[^a-z0-9]+", "", s)
            return s

        # Remove duplicated consecutive sentences (common when models repeat themselves).
        parts = re.split(r"(?<=[.!?])\s+", text)
        cleaned = []
        last = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            c = canon(part)
            if last is not None and c and c == last:
                continue
            cleaned.append(part)
            last = c
        text = " ".join(cleaned).strip()

        # If the response ends mid-sentence, trim to last sentence boundary.
        if text and text[-1] not in ".!?":
            last_end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_end > 60:
                text = text[: last_end + 1].strip()

        # Keep definition answers compact.
        if query_type in {"definition", "concept"}:
            text = re.sub(r"\s+", " ", text).strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 5:
                text = " ".join(sentences[:5]).strip()

        return text

    def _summarize_fallback(self, query: str, docs: list) -> str:
        """
        Fallback: simple summarization without LLM.

        Prefer the most query-relevant retrieved document instead of stitching
        together unrelated passages.

        Args:
            query: User query (unused in fallback)
            docs: Retrieved documents

        Returns:
            Query-focused grounded summary or a clear no-answer message
        """
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

        def content_terms(text: str) -> set[str]:
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            return {
                token for token in tokens if len(token) > 2 and token not in stopwords
            }

        query_terms = content_terms(query)
        best_score = -1.0
        best_doc = None

        for doc in docs:
            title = doc.get("title", "Unknown")
            text = doc.get("excerpt") or doc.get("text", "")
            if not text:
                continue

            doc_terms = content_terms(f"{title} {text}")
            overlap = len(query_terms & doc_terms) if query_terms else 0
            similarity = float(doc.get("similarity", 0.0) or 0.0)
            score = (overlap * 2.0) + similarity

            if score > best_score:
                best_score = score
                best_doc = doc

        if best_doc is None or best_score < 0.5:
            return f"I could not find a direct answer in the retrieved documents for: {query}"

        title = best_doc.get("title", "Unknown")
        source_text = best_doc.get("excerpt") or best_doc.get("text", "")
        source_text = " ".join(source_text.split())

        # Pick the most query-aligned sentence for cleaner fallback output.
        candidate_sentences = re.split(r"(?<=[.!?])\s+", source_text)
        best_sentence = source_text
        best_sentence_score = -1

        for sentence in candidate_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_terms = content_terms(sentence)
            sentence_score = len(query_terms & sentence_terms)
            if sentence_score > best_sentence_score:
                best_sentence_score = sentence_score
                best_sentence = sentence

        lowered_sentence = best_sentence.lower()
        earliest_term_index = None
        for term in sorted(query_terms, key=len, reverse=True):
            match = re.search(rf"\b{re.escape(term)}\b", lowered_sentence)
            if not match:
                continue
            idx = match.start()
            if earliest_term_index is None or idx < earliest_term_index:
                earliest_term_index = idx

        # Trim noisy leading fragment from OCR/chunk boundaries when a query term appears early.
        if earliest_term_index is not None and 0 < earliest_term_index <= 40:
            best_sentence = best_sentence[earliest_term_index:]

        best_sentence = best_sentence.strip(" ,;:-")
        if best_sentence and best_sentence[0].islower():
            best_sentence = best_sentence[0].upper() + best_sentence[1:]
        if best_sentence and best_sentence[-1] not in ".!?":
            best_sentence += "."

        if len(best_sentence) > 320:
            best_sentence = best_sentence[:320].rsplit(" ", 1)[0] + "..."

        return f"Definition from {title}: {best_sentence}"

    def _build_summary_prompt(self, state: WorkflowState, query: str, docs: list) -> str:
        """
        Build prompt for LLM summarization.

        Instructs the LLM to:
        - Answer naturally and directly first
        - Use provided excerpts only
        - Keep formatting minimal and readable

        Args:
            query: User query
            docs: Retrieved documents

        Returns:
            Formatted prompt for LLM
        """
        planner_decision = (
            state.get("metadata", {}).get("decisions", {}).get("planner", {}) or {}
        )
        query_type = str(planner_decision.get("query_type", "") or "").strip().lower()

        tool_calc = (
            state.get("metadata", {})
            .get("decisions", {})
            .get("pre_tools", {})
            .get("calculator_result")
        )
        tool_block = ""
        if tool_calc and isinstance(tool_calc, str) and tool_calc.strip():
            tool_block = (
                "\nTOOL RESULT (authoritative):\n"
                f"{tool_calc.strip()}\n"
                "Use this as the final numeric/symbolic result. Your job is to explain it clearly.\n\n"
            )

        mode_hint = "research"
        if query_type in {"definition", "concept"}:
            mode_hint = "definition"
        elif query_type in {"calculation", "solve", "proof"}:
            mode_hint = "calculation"

        format_rules = (
            "FORMATTING RULES (use Markdown + LaTeX when helpful):\n"
            "- Always include **Final Answer**.\n"
            "- Include **Formula/Setup** ONLY if the question asks for a formula or the answer truly depends on one.\n"
            "- Include **Step-by-step** ONLY for calculations/derivations (or if the user asks for steps).\n"
            "- For a purely conceptual/definition question, answer in 2-5 sentences and DO NOT force a formula section.\n"
            "- Use $$...$$ for displayed equations; for matrices use $$\\\\begin{bmatrix} ... \\\\end{bmatrix}$$.\n"
            "- IMPORTANT: Any LaTeX command (like \\\\times, \\\\sqrt, subscripts) MUST be inside $...$ or $$...$$.\n"
            "- Do not repeat sentences or restate the same definition twice.\n"
            "- Do not invent facts not present in the excerpts.\n"
        )

        section_template = (
            "Use ONLY the sections that are relevant (omit irrelevant ones):\n"
            "- **Concept**\n"
            "- **Formula/Setup**\n"
            "- **Step-by-step**\n"
            "- **Final Answer**\n"
            "- **Optional Check**\n"
        )

        prompt = (
            "You are an expert mathematics professor.\n"
            "Answer the user's question using ONLY the provided document excerpts.\n"
            "If the excerpts do not contain enough information, say so explicitly and state what is missing.\n\n"
            f"MODE_HINT: {mode_hint}\n\n"
            + format_rules
            + "\n"
            + section_template
            + "\nUSER QUESTION:\n{query}\n\n"
            + tool_block
            + "DOCUMENT EXCERPTS:\n{documents}\n"
        )

        docs_block = ""
        for i, doc in enumerate(docs, 1):
            excerpt = doc.get("excerpt") or doc.get("text", "")
            title = doc.get("title", "Document")
            source = doc.get("source", "unknown")
            docs_block += f"\n[Document {i}] {title} (source: {source})\n{excerpt}\n"

        # Avoid LangChain templating here because literal `{}` in LaTeX (e.g.
        # \begin{bmatrix}) can be interpreted as template variables.
        return prompt.replace("{query}", query).replace("{documents}", docs_block)
