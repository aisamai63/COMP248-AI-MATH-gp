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

        Extracts first sentence from each document and joins them.

        Args:
            query: User query (unused in fallback)
            docs: Retrieved documents

        Returns:
            Simple concatenated summary
        """
        parts = []
        for doc in docs:
            title = doc.get("title", "Unknown")
            text = doc.get("excerpt") or doc.get("text", "")

            # Get first sentence
            first_sentence = text.split(".")[0] if text else ""
            if first_sentence:
                parts.append(f"**{title}**: {first_sentence.strip()}.")

        return " ".join(parts) if parts else "[No content available]"

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
