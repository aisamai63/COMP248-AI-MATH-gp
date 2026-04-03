"""Streamlit frontend for the Math Inquiries LangGraph workflow."""

from __future__ import annotations

import logging
import pathlib
import sys
import html
from typing import Dict, Any

import streamlit as st

# Allow running from repo root: streamlit run prototype/app.py
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.workflow import create_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def initialize_graph():
    """Create and cache compiled workflow graph for Streamlit reruns."""
    logger.info("Initializing workflow graph for Streamlit")
    return create_graph()


def _init_session_state() -> None:
    """Initialize chat state once per browser session."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", {})


def _append_chat_message(role: str, content: str) -> None:
    """Store a message in the session history."""
    st.session_state.messages.append({"role": role, "content": content})


def _inject_css(has_messages: bool) -> None:
    """Inject layout and chat styling for centered/anchored input modes."""
    layout_mode = "has-messages" if has_messages else "empty-chat"
    st.markdown(
        f"""
        <style>
            :root {{
                --page-bg: #1f2024;
                --page-bg-2: #22242a;
                --panel-bg: #2a2d34;
                --panel-bg-2: #343841;
                --border: #3d414c;
                --text-main: #ececf1;
                --text-muted: #a7acb6;
                --assistant-bg: #2f323a;
                --user-bg: #3a3f49;
            }}

            .stApp {{
                background:
                    radial-gradient(1200px 520px at 50% 100%, rgba(78, 87, 108, 0.18), transparent 70%),
                    linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg-2) 100%);
                color: var(--text-main);
            }}

            #MainMenu,
            footer,
            header,
            [data-testid="stSidebar"],
            [data-testid="collapsedControl"],
            [data-testid="stDecoration"],
            [data-testid="stToolbar"],
            [data-testid="stStatusWidget"] {{
                display: none;
            }}

            [data-testid="stHeader"] {{
                height: 0;
            }}

            .block-container {{
                padding-top: 0.5rem;
                padding-bottom: 0.5rem;
                max-width: 980px;
            }}

            .empty-layout {{
                min-height: calc(100vh - 1rem);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 1.1rem;
            }}

            .welcome-title {{
                font-size: clamp(1.8rem, 3.8vw, 2.75rem);
                font-weight: 500;
                color: var(--text-main);
                text-align: center;
                letter-spacing: -0.01em;
            }}

            .chat-layout {{
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }}

            .chat-container {{
                gap: 0.75rem;
                overflow-y: auto;
                height: calc(100vh - 150px);
                padding: 1rem 0.3rem 7rem 0.3rem;
                scrollbar-width: thin;
            }}

            .chat-row {{
                width: 100%;
                display: flex;
            }}

            .chat-row.user {{
                justify-content: flex-end;
            }}

            .chat-row.assistant {{
                justify-content: flex-start;
            }}

            .chat-bubble {{
                max-width: min(80%, 740px);
                padding: 0.85rem 1rem;
                border-radius: 18px;
                line-height: 1.5;
                border: 1px solid var(--border);
                white-space: pre-wrap;
                word-break: break-word;
                color: var(--text-main);
            }}

            .chat-bubble.user {{
                background: var(--user-bg);
                border-top-right-radius: 6px;
            }}

            .chat-bubble.assistant {{
                background: var(--assistant-bg);
                border-top-left-radius: 6px;
            }}

            .center-input-wrap,
            .bottom-input-wrap {{
                width: min(760px, calc(100vw - 1.2rem));
                margin-left: auto;
                margin-right: auto;
            }}

            .center-input-wrap {{
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .bottom-input-wrap {{
                position: fixed;
                left: 50%;
                transform: translateX(-50%);
                bottom: 0.8rem;
                z-index: 1000;
            }}

            div[data-testid="stForm"] {{
                width: 100%;
                margin: 0;
            }}

            div[data-testid="stForm"] > form {{
                background: var(--panel-bg);
                border: 1px solid var(--border);
                border-radius: 28px;
                padding: 0.45rem 0.5rem;
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.24);
            }}

            .composer-row {{
                display: flex;
                align-items: center;
                gap: 0.45rem;
            }}

            div[data-testid="stTextInput"] input {{
                border-radius: 20px !important;
                border: none !important;
                background: transparent !important;
                color: var(--text-main) !important;
                padding: 0.75rem 0.7rem !important;
                box-shadow: none !important;
            }}

            div[data-testid="stTextInput"] input::placeholder {{
                color: var(--text-muted) !important;
            }}

            div[data-testid="stFormSubmitButton"] button {{
                background: #ececf1 !important;
                color: #1f2024 !important;
                border-radius: 999px !important;
                width: 2.25rem !important;
                height: 2.25rem !important;
                min-height: 2.25rem !important;
                padding: 0 !important;
                font-weight: 700 !important;
                border: none !important;
            }}

            div[data-testid="stFormSubmitButton"] button:hover {{
                filter: brightness(0.96);
            }}

            .stExpander {{
                border: 1px solid var(--border) !important;
                background: #252830 !important;
                border-radius: 14px !important;
            }}

            .mode-marker::before {{
                content: "{layout_mode}";
                display: none;
            }}

            @media (max-width: 640px) {{
                .welcome-title {{
                    font-size: 1.9rem;
                    padding: 0 0.4rem;
                }}

                div[data-testid="stForm"] > form {{
                    border-radius: 24px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_container(messages: list[dict[str, str]]) -> None:
    """Render messages at the top in a vertically scrollable chat area."""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in messages:
        role = message.get("role", "assistant")
        if role not in {"user", "assistant"}:
            role = "assistant"
        content = html.escape(message.get("content", "")).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="chat-row {role}">
                <div class="chat-bubble {role}">{content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_input_form() -> tuple[bool, str]:
    """Render the query form and return the submit state and input value."""
    with st.form("chat_input_form", clear_on_submit=True):
        st.markdown('<div class="composer-row">', unsafe_allow_html=True)
        c1, c2 = st.columns([15, 1])
        with c1:
            query = st.text_input(
                "Message",
                value="",
                placeholder="Ask anything",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("↑", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    return submitted, query


def _render_diagnostics(result: Dict[str, object]) -> None:
    """Render optional workflow diagnostics below the conversation."""
    with st.expander("Diagnostics", expanded=False):
        t1, t2, t3, t4 = st.tabs(["Answer", "Retrieval", "Reflection", "Metadata"])

        with t1:
            st.markdown(result.get("summary", "[No summary generated]"))
            if result.get("tool_results"):
                st.subheader("Tool Results")
                st.markdown(result["tool_results"])

        with t2:
            docs = result.get("retrieved_docs", [])
            st.write(f"Retrieved documents: {len(docs)}")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"Document {i}: {doc.get('title', 'Untitled')}"):
                    st.write(f"Source: {doc.get('source', 'unknown')}")
                    sim = doc.get("similarity")
                    if sim is not None:
                        st.write(f"Similarity: {sim:.2%}")
                    st.write(doc.get("excerpt") or doc.get("text", ""))

        with t3:
            metrics = result.get("reflection_metrics", {})
            if not metrics:
                st.info("No reflection metrics available")
            else:
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Factual Correctness",
                        f"{metrics.get('factual_correctness', 0):.1%}",
                    )
                with c2:
                    st.metric("Completeness", f"{metrics.get('completeness', 0):.1%}")
                with c3:
                    st.metric("Relevance", f"{metrics.get('relevance', 0):.1%}")
                with c4:
                    st.metric("Confidence", f"{metrics.get('confidence', 0):.1%}")

                feedback = metrics.get("feedback_text")
                if feedback:
                    st.subheader("Feedback")
                    st.write(feedback)

                notes = metrics.get("notes", [])
                if notes:
                    st.subheader("Notes")
                    for note in notes:
                        st.write(f"- {note}")

        with t4:
            metadata = result.get("metadata", {})
            sequence = metadata.get("agent_sequence", [])
            decisions = metadata.get("decisions", {})
            timestamps = metadata.get("timestamps", {})

            st.subheader("Execution Sequence")
            if sequence:
                st.write(" -> ".join(sequence))
            else:
                st.info("No agent sequence recorded")

            st.subheader("Node Decisions")
            if decisions:
                st.json(decisions)
            else:
                st.info("No node decisions recorded")

            st.subheader("Node Timestamps")
            if timestamps:
                st.json(timestamps)
            else:
                st.info("No node timestamps recorded")

            st.subheader("Full Metadata")
            st.json(metadata)
            st.write(f"Iterations: {result.get('iteration_count', 0)}")


def _run_query(graph, query: str) -> Dict[str, Any]:
    """Run the workflow safely and return the result payload."""
    try:
        return graph.run(query)
    except Exception as exc:
        logger.error("Workflow failed: %s", exc, exc_info=True)
        raise


def main() -> None:
    """Main Streamlit UI entrypoint."""
    st.set_page_config(
        page_title="Math Inquiries",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session_state()

    has_messages = bool(st.session_state.messages)
    _inject_css(has_messages)

    if has_messages:
        st.markdown('<div class="chat-layout">', unsafe_allow_html=True)
        _render_chat_container(st.session_state.messages)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="bottom-input-wrap">', unsafe_allow_html=True)
        submitted, query = _render_input_form()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-layout">', unsafe_allow_html=True)
        st.markdown(
            '<div class="welcome-title">What\'s on the agenda today?</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="center-input-wrap">', unsafe_allow_html=True)
        submitted, query = _render_input_form()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="mode-marker"></div>', unsafe_allow_html=True)

    if submitted:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.error("Please enter a query.")
            return

        graph = initialize_graph()
        with st.spinner("Running workflow..."):
            try:
                result = _run_query(graph, cleaned_query)
            except Exception as exc:
                st.error(f"Workflow failed: {exc}")
                return

        st.session_state.last_result = result
        _append_chat_message("user", cleaned_query)
        _append_chat_message(
            "assistant", result.get("summary", "[No summary generated]")
        )

        if result.get("tool_results"):
            st.session_state.messages.append(
                {"role": "assistant", "content": result["tool_results"]}
            )

        st.rerun()

    if has_messages and st.session_state.last_result:
        _render_diagnostics(st.session_state.last_result)


if __name__ == "__main__":
    main()
