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
            --bg-main: #212121;
            --bg-user: #10a37f;
            --bg-assistant: #353740;
            --text-main: #f4f4f7;
            --text-muted: #a9adbd;
            --border: #2d2f36;
        }

        .stApp {
            background: var(--bg-main);
            color: var(--text-main);
        }

        #MainMenu, footer, header {
            display: none;
        }

        .block-container {
            max-width: 600px;
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

        .stExpander {
            background: #232428 !important;
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

    # Advanced optimization: Singleton cache for ChromaDB and dummy query to fully warm up
    if "_embedding_and_chromadb_warmed" not in st.session_state:
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
