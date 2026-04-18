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
        self.llm_ready = False
        self.provider = llm_config.LLM_PROVIDER
        self.mistral_client = None
        self.openai_client = None
        self.gemini_model = None

        if self.use_llm:
            self._initialize_llm_client()
        elif runtime_config.FAST_MODE:
            self.logger.info("FAST_MODE enabled: summarizer LLM disabled")

    def _initialize_llm_client(self) -> None:
        """Initialize LLM client based on configured provider."""
        try:
            if self.provider == "gemini":
                if not llm_config.GEMINI_API_KEY:
                    self.logger.warning("GEMINI_API_KEY missing; using fallback.")
                    return
                import google.generativeai as genai

                genai.configure(api_key=llm_config.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(llm_config.GEMINI_MODEL)
                self.llm_ready = True
                self.logger.info("Gemini client initialized successfully")
                return

            if self.provider == "openai":
                if not llm_config.OPENAI_API_KEY:
                    self.logger.warning("OPENAI_API_KEY missing; using fallback.")
                    return
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=llm_config.OPENAI_API_KEY)
                self.llm_ready = True
                self.logger.info("OpenAI client initialized successfully")
                return

            # Default to Mistral for backward compatibility.
            if not llm_config.MISTRAL_API_KEY:
                self.logger.warning("MISTRAL_API_KEY missing; using fallback.")
                return
            from mistralai.client import Mistral

            self.mistral_client = Mistral(api_key=llm_config.MISTRAL_API_KEY)
            self.llm_ready = True
            self.logger.info("Mistral client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize {self.provider} client: {e}")
            self.llm_ready = False

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
                decisions["summarizer"] = {
                    "mode": "fallback",
                    "reason": "llm_disabled_or_unavailable",
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
            prompt = self._build_summary_prompt(query, docs)

            llm_started = time.perf_counter()

            if self.provider == "gemini" and self.gemini_model is not None:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": llm_config.MISTRAL_TEMPERATURE,
                        "max_output_tokens": llm_config.MISTRAL_MAX_TOKENS_DEFAULT,
                    },
                )
                content = self._extract_gemini_text(response)
            elif self.provider == "openai" and self.openai_client is not None:
                response = self.openai_client.chat.completions.create(
                    model=llm_config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=llm_config.MISTRAL_MAX_TOKENS_DEFAULT,
                    temperature=llm_config.MISTRAL_TEMPERATURE,
                )
                content = response.choices[0].message.content or ""
            else:
                response = self.mistral_client.chat.complete(
                    model=llm_config.MISTRAL_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=llm_config.MISTRAL_MAX_TOKENS_DEFAULT,
                    temperature=llm_config.MISTRAL_TEMPERATURE,
                )
                content = response.choices[0].message.content

            llm_elapsed = time.perf_counter() - llm_started
            self.logger.info(
                "Timing | summarizer_llm_call | provider=%s | %.3fs",
                self.provider,
                llm_elapsed,
            )

            state.setdefault("metadata", {}).setdefault("decisions", {})[
                "summarizer"
            ] = {
                "mode": "llm",
                "provider": self.provider,
                "llm_elapsed": llm_elapsed,
            }

            return content.strip()

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
        return "\n".join(parts).strip()

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

    def _build_summary_prompt(self, query: str, docs: list) -> str:
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
        prompt = (
            "You are an expert research assistant. Answer the user's question using ONLY "
            "the provided document excerpts. Do not use external knowledge.\n\n"
            f"USER QUESTION: {query}\n\n"
            "RESPONSE STYLE:\n"
            "- Start with a short direct answer in plain language.\n"
            "- Add 2-4 bullet points only if they help clarity.\n"
            "- Avoid numbered section headers unless explicitly requested.\n"
            "- Keep it concise and classroom-friendly.\n\n"
            "DOCUMENTS:\n"
        )

        for i, doc in enumerate(docs, 1):
            excerpt = doc.get("excerpt") or doc.get("text", "")
            title = doc.get("title", "Document")
            source = doc.get("source", "unknown")
            prompt += f"\n[Document {i}] {title} (source: {source})\n{excerpt}\n"

        prompt += (
            "\nPlease answer based only on the above documents. "
            "If the user has a typo, briefly interpret it and answer the intended question."
        )
        return prompt
