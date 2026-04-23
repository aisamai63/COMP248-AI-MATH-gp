"""Streamlit frontend for the Math Inquiries LangGraph workflow."""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Dict, Any

import streamlit as st
import re


def render_math_or_text(text):
    """Render LaTeX blocks with st.latex and fallback to markdown for plain text."""
    if not text:
        return

    # Normalize common LaTeX delimiters into forms Streamlit handles well.
    # - Convert \[...\] blocks into $$...$$ blocks.
    # - Convert \( ... \) inline into $...$ inline.
    text = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", r"$$\1$$", text)
    text = re.sub(r"\\\(\s*([\s\S]*?)\s*\\\)", r"$\1$", text)

    # If the model outputs LaTeX commands inside normal parentheses, wrap as inline math.
    # Example: "( m \\times n )" -> "$m \\times n$"
    def _wrap_paren_latex(match: re.Match) -> str:
        inner = match.group(1) or ""
        inner = inner.strip()
        return f"${inner}$" if inner else match.group(0)

    text = re.sub(r"\(\s*([^)]*\\[a-zA-Z]+[^)]*)\s*\)", _wrap_paren_latex, text)

    # Render $$...$$ blocks as LaTeX
    latex_blocks = re.findall(r"\$\$(.*?)\$\$", text, re.DOTALL)
    for block in latex_blocks:
        st.latex(block)
    # Remove rendered blocks from text
    text = re.sub(r"\$\$(.*?)\$\$", "", text, flags=re.DOTALL)
    # Render remaining text (may include inline $...$)
    if text.strip():
        st.markdown(text)


def _extract_chat_answer(summary: str) -> str:
    """
    Make the chat bubble show just the answer, not section labels like "Final Answer".

    Diagnostics still show the full structured summary.
    """
    if not summary:
        return ""

    text = summary.replace("\r\n", "\n").strip()

    # If the model produced a "Final Answer" section, extract its body until the next section.
    section_header = re.compile(
        r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?\s*(concept|formula/setup|step-by-step|final answer|optional check)\s*(?:\*\*)?\s*:?\s*$"
    )
    final_inline = re.compile(
        r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?\s*final answer\s*(?:\*\*)?\s*:?\s*(.+?)\s*$"
    )

    m = final_inline.search(text)
    if m:
        start = m.start(0)
        # slice from the matched line to search for next header after it
        rest = text[start:]
        # current line content after "Final Answer"
        first_line = m.group(1).strip()
        # remaining after the line
        after_line = rest.split("\n", 1)[1] if "\n" in rest else ""

        collected = [first_line] if first_line else []
        if after_line.strip():
            # Stop at the next known header (Concept/Formula/...).
            lines = after_line.split("\n")
            for line in lines:
                if section_header.match(line):
                    break
                collected.append(line)
        return "\n".join([l for l in collected if l is not None]).strip()

    # No inline final answer; try to extract a block starting at a "Final Answer" header line.
    lines = text.split("\n")
    out_lines: list[str] = []
    in_final = False
    for line in lines:
        if section_header.match(line):
            name = section_header.match(line).group(1).lower()
            in_final = name == "final answer"
            continue
        if in_final:
            # Stop if another header appears (handled above) or if we hit an empty line after content.
            out_lines.append(line)
    if any(l.strip() for l in out_lines):
        return "\n".join(out_lines).strip()

    # Otherwise: strip common section labels if present and return the remaining text.
    stripped = []
    for line in lines:
        # Remove leading "Final Answer ..." even if it's not formatted as a header.
        line2 = re.sub(r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?\s*final answer\s*(?:\*\*)?\s*:?\s*", "", line)
        line2 = re.sub(r"(?im)^\s*(?:\d+\.\s*)?(?:\*\*)?\s*(concept|formula/setup|step-by-step|optional check)\s*(?:\*\*)?\s*:?\s*", "", line2)
        stripped.append(line2)
    return "\n".join(stripped).strip()


# Allow running from repo root
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.workflow import create_runtime_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_graph():
    logger.info("Initializing workflow graph for Streamlit")
    return create_runtime_graph()


@st.cache_resource
def warm_runtime_resources() -> bool:
    """
    Warm expensive runtime resources (embeddings + Chroma) only when needed.

    Putting model/DB warm-up at app startup delays the first paint and keeps the
    Streamlit loading screen (logo/spinner) visible for a long time. We warm on
    the first user query instead.
    """
    from prototype.db import get_embedding_model as get_db_embedding_model
    from prototype.chroma_setup import get_chroma_client, get_chroma_collection

    get_db_embedding_model()

    client = get_chroma_client()
    collection = get_chroma_collection(client)
    try:
        collection.count()
    except Exception as exc:
        logger.warning("ChromaDB warm-up count failed: %s", exc)

    return True


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", {})


def _append_chat_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


# ---------------- UI / CSS ---------------- #
def _inject_css(has_messages: bool) -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #212121;
            --bg-user: #10a37f;
            --bg-assistant: #353740;
            --text-main: #f4f4f7;
            --text-muted: #a9adbd;
            --border: #2d2f36;
            --card: #232428;
        }

        .stApp {
            background: var(--bg-main);
            color: var(--text-main);
        }

        #MainMenu, footer, header {
            display: none;
        }

        .block-container {
            max-width: 820px;
            margin: 0 auto;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        .center-screen {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 65vh;
        }

        .welcome-title {
            text-align: center;
            font-size: 2rem;
            font-weight: 600;
        }

        .page-subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 2rem;
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
        }

        .chat-row {
            display: flex;
        }

        .chat-row.user {
            justify-content: flex-end;
        }

        .chat-row.assistant {
            justify-content: flex-start;
        }

        .chat-bubble {
            max-width: 80%;
            padding: 0.85rem 1.1rem;
            border-radius: 12px;
            line-height: 1.6;
            font-size: 1.05rem;
            margin-bottom: 2px;
        }

        .chat-bubble.user {
            background: var(--bg-user);
            color: #fff;
        }

        .chat-bubble.assistant {
            background: var(--bg-assistant);
            color: var(--text-main);
        }

        div[data-testid="stChatInput"] {
            position: fixed;
            left: 50%;
            transform: translateX(-50%);
            bottom: 1.2rem;
            width: min(600px, 98vw);
            z-index: 1000;
            background: var(--bg-main);
            border-radius: 16px;
            box-shadow: 0 2px 16px rgba(0,0,0,0.10);
        }

        div[data-testid="stChatInput"] input {
            background: #232428 !important;
            color: var(--text-main) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 1rem 1.2rem !important;
            font-size: 1.08rem !important;
        }

        div[data-testid="stChatInput"] button {
            background: var(--bg-user) !important;
            color: #fff !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
        }

        /* Modernize default Streamlit chat */
        section.main > div {
            padding-bottom: 6rem;
        }

        div[data-testid="stChatMessage"] > div {
            border-radius: 14px;
        }

        div[data-testid="stChatMessage"] p,
        div[data-testid="stChatMessage"] li {
            line-height: 1.65;
            font-size: 1.05rem;
        }

        /* Diagnostics card */
        .diag-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.25rem 1.25rem 0.75rem 1.25rem;
            margin: 1rem 0 1.25rem 0;
        }
        .diag-title {
            font-size: 1.1rem;
            font-weight: 650;
            margin-bottom: 0.75rem;
            color: var(--text-main);
        }
        .stExpander {
            background: var(--card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 0.65rem 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- CHAT ---------------- #
def _render_chat_container(messages):
    if not messages:
        st.markdown(
            """
            <div style="text-align:center; color:#acacbe;">
                <h3>How can I help you today?</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in messages:
        role = message.get("role", "assistant") or "assistant"
        content = message.get("content", "") or ""
        with st.chat_message(role):
            if role == "assistant":
                render_math_or_text(content)
            else:
                st.markdown(content)


def _render_input_form():
    query = st.chat_input("Ask anything...")
    submitted = bool(query)
    return submitted, query or ""


# ---------------- DIAGNOSTICS ---------------- #
def _render_diagnostics(result: Dict[str, object]) -> None:
    metrics = result.get("reflection_metrics", {}) or {}
    docs = result.get("retrieved_docs", []) or []
    metadata = result.get("metadata", {}) or {}
    summarizer_meta = metadata.get("decisions", {}).get("summarizer", {}) or {}
    reflection_source = metrics.get("evaluation_source", "") or ""

    if summarizer_meta.get("mode") == "fallback":
        reason = summarizer_meta.get("reason", "fallback summary used")
        provider = summarizer_meta.get("provider", "?")
        # Show more details if available
        error_details = ""
        if reason and ("llm_init_error" in reason or "llm_error" in reason):
            error_details = f"\n**LLM Error Details:** {reason}"
        st.warning(
            f"Summary fallback active (provider: {provider}): {reason}{error_details}"
        )

    if reflection_source and reflection_source != "llm":
        st.info(f"Reflection fallback active: {reflection_source}")

    st.markdown('<div class="diag-card">', unsafe_allow_html=True)
    st.markdown('<div class="diag-title">Diagnostics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Docs", str(len(docs)))
    c2.metric("Confidence", f"{metrics.get('confidence', 0):.0%}")
    c3.metric("Relevance", f"{metrics.get('relevance', 0):.0%}")
    c4.metric("Iterations", str(result.get("iteration_count", 0)))

    with st.expander("Open detailed diagnostics", expanded=False):
        t1, t2, t3, t4, t5, t6 = st.tabs(
            ["Answer", "Retrieval", "Reflection", "Tools", "Metadata", "LLM"]
        )

        with t1:
            render_math_or_text(result.get("summary", "[No summary generated]"))

        with t2:
            st.write(f"Retrieved: {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                st.markdown(
                    f"- {i}. {doc.get('title', 'Untitled')} ({doc.get('source', 'unknown')})"
                )

        with t3:
            st.json(metrics)

        with t4:
            pre_tools = metadata.get("decisions", {}).get("pre_tools", {}) or {}
            calc = pre_tools.get("calculator_result")
            tool_results = result.get("tool_results")

            if calc:
                st.markdown("### Calculator")
                render_math_or_text(str(calc))

            if tool_results:
                st.markdown("### Tool Results")
                render_math_or_text(str(tool_results))

            if not calc and not tool_results:
                st.info("No tool output for this run.")

        with t5:
            st.json(metadata)

        with t6:
            llm_calls = metadata.get("llm_calls", []) or []
            if not llm_calls:
                st.info("No LLM call diagnostics recorded for this run.")
            else:
                failures = [
                    c
                    for c in llm_calls
                    if isinstance(c, dict) and not c.get("ok", True)
                ]
                if failures:
                    st.warning(f"LLM call failures: {len(failures)}")
                st.dataframe(llm_calls, width="stretch")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- CORE ---------------- #
def _run_query(graph, query: str) -> Dict[str, Any]:
    try:
        return graph.run(query)
    except Exception as exc:
        logger.error("Workflow failed: %s", exc, exc_info=True)
        raise


# ---------------- MAIN ---------------- #
def main():
    # Startup warm-up delays first paint; warm on first query instead.
    # Set PREWARM_ON_STARTUP=1 if you prefer slower initial load and faster first query.
    prewarm_on_startup = os.getenv("PREWARM_ON_STARTUP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if prewarm_on_startup and "_embedding_and_chromadb_warmed" not in st.session_state:
        try:
            from prototype.chroma_setup import (
                get_embedding_model,
                get_chroma_client,
                get_chroma_collection,
            )

            # Singleton cache for ChromaDB client and collection
            if (
                "_chroma_client" not in st.session_state
                or "_chroma_collection" not in st.session_state
            ):
                get_embedding_model()
                client = get_chroma_client()
                collection = get_chroma_collection(client)
                st.session_state["_chroma_client"] = client
                st.session_state["_chroma_collection"] = collection
                # Dummy query to force ChromaDB to fully load
                try:
                    collection.count()
                except Exception as dummy_exc:
                    logger.warning("ChromaDB dummy query failed: %s", dummy_exc)
            logger.info(
                "Embedding model and ChromaDB pre-warm complete (singleton cache + dummy query)"
            )
            st.session_state["_embedding_and_chromadb_warmed"] = True
        except Exception:
            logger.warning(
                "Embedding/ChromaDB pre-warm failed; continuing without pre-warm",
                exc_info=True,
            )

    st.set_page_config(
        page_title="Math Inquiries",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session_state()
    _inject_css(bool(st.session_state.messages))
    loading_mount = st.empty()

    submitted = False
    query = ""

    # Centered layout on first load
    if not st.session_state.messages:
        st.markdown('<div class="center-screen">', unsafe_allow_html=True)

        st.markdown(
            '<div class="welcome-title">What\'s on the agenda today?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="page-subtitle">Ask a math question and explore the reasoning.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="welcome-title">Math Assistant</div>',
            unsafe_allow_html=True,
        )

    # Keep loading indicator anchored above diagnostics/chat.
    loading_mount.markdown('<div class="loading-anchor"></div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        _render_diagnostics(st.session_state.last_result)

    _render_chat_container(st.session_state.messages)

    submitted, query = _render_input_form()

    if submitted:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.error("Please enter a query.")
            return

        with loading_mount.container():
            with st.spinner("Preparing..."):
                try:
                    warm_runtime_resources()
                except Exception:
                    logger.warning(
                        "Warm-up failed; continuing without warm-up",
                        exc_info=True,
                    )
                graph = initialize_graph()
                result = _run_query(graph, cleaned_query)

        st.session_state.last_result = result
        _append_chat_message("user", cleaned_query)

        # Chat shows a compact answer (no "Final Answer" label); full output stays in diagnostics.
        summary = result.get("summary", "") or ""
        _append_chat_message("assistant", _extract_chat_answer(summary))

        st.rerun()

    if not st.session_state.messages:
        loading_mount.markdown(
            '<div class="loading-anchor"></div>', unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
