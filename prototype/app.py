"""Streamlit frontend for the Math Inquiries LangGraph workflow."""

from __future__ import annotations

import logging
import pathlib
import sys
import html
from typing import Dict, Any

import streamlit as st

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
            --bg-main: #343541;
            --bg-assistant: #444654;
            --bg-panel: #3b3d4e;
            --text-main: #ececf1;
            --text-muted: #acacbe;
            --border: #565869;
            --accent: #8ab4ff;
        }

        .stApp {
            background-color: var(--bg-main);
            color: var(--text-main);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            max-width: 850px;
            padding-top: 2rem;
            padding-bottom: 8.5rem;
        }

        /* Center first screen */
        .center-screen {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 65vh;
        }

        .welcome-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 600;
        }

        .page-subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 2rem;
        }

        .loading-anchor {
            min-height: 2.2rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0.35rem 0 0.8rem 0;
        }

        .diag-card {
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.8rem;
            margin-bottom: 1rem;
        }

        .diag-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.55rem;
        }

        .diag-card .stMetric {
            background: #323443;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.35rem 0.55rem;
        }

        .diag-card .stExpander {
            margin-top: 0.65rem;
        }

        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1.2rem;
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
            max-width: 75%;
            padding: 0.9rem 1.1rem;
            border-radius: 14px;
            line-height: 1.6;
        }

        .chat-bubble.user {
            background: var(--bg-assistant);
        }

        .chat-bubble.assistant {
            background: transparent;
            border: 1px solid var(--border);
        }

        /* Keep chat input visible and centered while scrolling */
        div[data-testid="stChatInput"] {
            position: fixed;
            left: 50%;
            transform: translateX(-50%);
            bottom: 0.95rem;
            width: min(850px, calc(100% - 2rem));
            z-index: 1000;
            background: transparent;
        }

        div[data-testid="stChatInput"] > div {
            background: #40414f;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
        }

        div[data-testid="stChatInput"] input {
            background: transparent !important;
            border: none !important;
            color: var(--text-main) !important;
        }

        div[data-testid="stChatInput"] button {
            background: white !important;
            color: black !important;
            border-radius: 8px !important;
        }

        .stExpander {
            background: #2f303a !important;
            border: 1px solid var(--border) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------- CHAT ---------------- #
def _render_chat_container(messages):
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not messages:
        st.markdown(
            """
            <div style="text-align:center; color:#acacbe;">
                <h3>How can I help you today?</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for message in messages:
            role = message.get("role", "assistant")
            content = message.get("content", "")
            safe_content = html.escape(content).replace("\n", "<br>")

            if role == "user":
                st.markdown(
                    f"<div class='chat-row user'><div class='chat-bubble user'>{safe_content}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-row assistant'><div class='chat-bubble assistant'>{safe_content}</div></div>",
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)


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
        st.warning(f"Summary fallback active: {reason}")

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
        t1, t2, t3, t4 = st.tabs(["Answer", "Retrieval", "Reflection", "Metadata"])

        with t1:
            st.markdown(result.get("summary", "[No summary generated]"))

        with t2:
            st.write(f"Retrieved: {len(docs)} documents")
            for i, doc in enumerate(docs, 1):
                st.markdown(
                    f"- {i}. {doc.get('title', 'Untitled')} ({doc.get('source', 'unknown')})"
                )

        with t3:
            st.json(metrics)

        with t4:
            st.json(metadata)

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

        graph = initialize_graph()

        with loading_mount.container():
            with st.spinner("Thinking..."):
                result = _run_query(graph, cleaned_query)

        st.session_state.last_result = result
        _append_chat_message("user", cleaned_query)

        _append_chat_message("assistant", result.get("summary", ""))

        st.rerun()

    if not st.session_state.messages:
        loading_mount.markdown(
            '<div class="loading-anchor"></div>', unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
