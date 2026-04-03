"""Streamlit chat UI helpers for the Math Inquiries frontend.

This module contains only presentation logic: CSS, layout, message rendering,
and the centered composer. The workflow execution remains in prototype.app.
"""

from __future__ import annotations

import html
from typing import Dict, List, Tuple

import streamlit as st
from streamlit.components.v1 import html as st_html


def apply_chat_css() -> None:
    """Inject a modern, dark, ChatGPT-like visual theme."""
    st.markdown(
        """
        <style>
            :root {
                --bg-main: #0f1117;
                --bg-panel: #171923;
                --bg-card: #1f2430;
                --bg-card-2: #262b38;
                --border-subtle: rgba(255, 255, 255, 0.08);
                --text-main: #f4f4f7;
                --text-muted: #a9adbd;
                --accent: #10a37f;
                --accent-soft: rgba(16, 163, 127, 0.15);
            }

            .stApp {
                background:
                    radial-gradient(circle at top, rgba(16, 163, 127, 0.10), transparent 36%),
                    linear-gradient(180deg, #0c0e14 0%, #0f1117 100%);
                color: var(--text-main);
            }

            /* Remove Streamlit chrome for a clean app shell. */
            #MainMenu,
            footer,
            header {
                visibility: hidden;
            }

            [data-testid="stHeader"] {
                height: 0;
            }

            [data-testid="stSidebar"], [data-testid="collapsedControl"] {
                display: none;
            }

            .block-container {
                max-width: 1120px;
                padding-top: 1.25rem;
                padding-bottom: 1.5rem;
            }

            .chat-shell {
                min-height: calc(100vh - 2.5rem);
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }

            .chat-hero {
                margin: 1.25rem auto 0.5rem auto;
                text-align: center;
                max-width: 760px;
            }

            .chat-hero h1 {
                margin: 0;
                font-size: clamp(2rem, 3vw, 3.1rem);
                line-height: 1.05;
                letter-spacing: -0.04em;
                font-weight: 750;
            }

            .chat-hero p {
                margin: 0.55rem 0 0 0;
                color: var(--text-muted);
                font-size: 0.98rem;
            }

            .top-spacer {
                height: 5vh;
            }

            .intro-card {
                width: 100%;
                max-width: 760px;
                margin: 0 auto 1rem auto;
                padding: 1rem 1.15rem;
                border-radius: 20px;
                background: rgba(31, 36, 48, 0.65);
                border: 1px solid var(--border-subtle);
                box-shadow: 0 12px 32px rgba(0, 0, 0, 0.26);
                color: var(--text-main);
            }

            .intro-card .muted {
                color: var(--text-muted);
                margin-top: 0.35rem;
                font-size: 0.94rem;
            }

            /* Center the composer block and give it a floating feel. */
            div[data-testid="stForm"] {
                width: 100%;
                max-width: 760px;
                margin: 0 auto;
                padding: 0;
            }

            div[data-testid="stForm"] > form {
                background: rgba(31, 36, 48, 0.86);
                border: 1px solid var(--border-subtle);
                border-radius: 28px;
                padding: 0.95rem 0.95rem 0.9rem 0.95rem;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
                transition: box-shadow 180ms ease, border-color 180ms ease, transform 180ms ease;
            }

            div[data-testid="stForm"] > form:hover {
                border-color: rgba(16, 163, 127, 0.38);
                box-shadow: 0 0 0 1px rgba(16, 163, 127, 0.18), 0 22px 54px rgba(0, 0, 0, 0.42);
                transform: translateY(-1px);
            }

            /* Keep the text entry dark, rounded, and left-aligned. */
            div[data-testid="stTextInput"] input {
                background: var(--bg-card) !important;
                color: var(--text-main) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
                border-radius: 999px !important;
                padding: 0.9rem 1.05rem !important;
                box-shadow: inset 0 0 0 1px transparent;
            }

            div[data-testid="stTextInput"] input::placeholder {
                color: #7f8498 !important;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: rgba(16, 163, 127, 0.7) !important;
                box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.14) !important;
            }

            /* Make the submit button subtle and chat-like. */
            .stButton > button,
            div[data-testid="stFormSubmitButton"] button {
                background: linear-gradient(180deg, #10a37f 0%, #0f8f6f 100%) !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 999px !important;
                padding: 0.85rem 1rem !important;
                font-weight: 700 !important;
                transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
            }

            .stButton > button:hover,
            div[data-testid="stFormSubmitButton"] button:hover {
                filter: brightness(1.06);
                box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.14);
                transform: translateY(-1px);
            }

            /* Tabs and panels kept darker and minimal. */
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                border-bottom: 1px solid var(--border-subtle);
            }

            .stTabs [data-baseweb="tab"] {
                color: var(--text-muted);
                padding-left: 0;
                padding-right: 0;
            }

            .stTabs [aria-selected="true"] {
                color: var(--text-main) !important;
            }

            .stExpander {
                background: rgba(31, 36, 48, 0.72);
                border: 1px solid var(--border-subtle);
                border-radius: 16px;
            }

            .stMetric {
                background: rgba(31, 36, 48, 0.72);
                padding: 0.75rem 1rem;
                border-radius: 16px;
                border: 1px solid var(--border-subtle);
            }

            .stAlert {
                border-radius: 14px;
            }

            .results-panel {
                max-width: 760px;
                margin: 1rem auto 0 auto;
            }

            .results-panel .stMarkdown,
            .results-panel p,
            .results-panel label,
            .results-panel span {
                color: var(--text-main);
            }

            @media (max-width: 768px) {
                .block-container {
                    padding-left: 0.85rem;
                    padding-right: 0.85rem;
                }

                .chat-hero {
                    margin-top: 0.75rem;
                }

                div[data-testid="stForm"] > form {
                    border-radius: 22px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the centered title block."""
    st.markdown(
        """
        <div class="chat-hero">
            <h1>Math Inquiries</h1>
            <p>Planner → Retriever → Summarizer → Reflector → ToolAgent</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_spacer(has_history: bool) -> None:
    """Add vertical breathing room so the composer feels centered when chat is empty."""
    spacer_height = "3vh" if has_history else "16vh"
    st.markdown(
        f'<div class="top-spacer" style="height:{spacer_height};"></div>',
        unsafe_allow_html=True,
    )


def render_intro_if_empty(has_history: bool) -> None:
    """Show a small intro card only before the first message."""
    if has_history:
        return

    st.markdown(
        """
        <div class="intro-card">
            <strong>Ask anything about math.</strong>
            <div class="muted">The input stays centered, the chat history grows above it, and the workflow runs in the background.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escape_message(content: str) -> str:
    return html.escape(content).replace("\n", "<br>")


def _build_history_html(messages: List[Dict[str, str]]) -> str:
    """Build a scrollable, centered chat transcript using raw HTML for pixel-level control."""
    if not messages:
        empty_state = """
        <div class="empty-history">
            <div class="empty-title">No messages yet</div>
            <div class="empty-subtitle">Use the input below to start a conversation.</div>
        </div>
        """
    else:
        rows = []
        for message in messages:
            role = message.get("role", "assistant")
            content = _escape_message(message.get("content", ""))
            rows.append(
                f"""
                <div class="row {role}">
                    <div class="bubble {role}">{content}</div>
                </div>
                """
            )
        empty_state = "\n".join(rows)

    return f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            :root {{
                --bg-panel: rgba(31, 36, 48, 0.55);
                --bg-user: linear-gradient(180deg, rgba(16, 163, 127, 0.24), rgba(16, 163, 127, 0.12));
                --bg-assistant: rgba(41, 47, 61, 0.92);
                --border: rgba(255, 255, 255, 0.08);
                --text-main: #f4f4f7;
                --text-muted: #a9adbd;
            }}

            html, body {{
                margin: 0;
                background: transparent;
                color: var(--text-main);
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }}

            .history-wrap {{
                max-width: 760px;
                margin: 0 auto;
                background: var(--bg-panel);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 1rem;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
                max-height: 48vh;
                overflow-y: auto;
                scrollbar-width: thin;
                scrollbar-color: rgba(255,255,255,0.18) transparent;
            }}

            .history-wrap::-webkit-scrollbar {{ width: 8px; }}
            .history-wrap::-webkit-scrollbar-thumb {{
                background: rgba(255,255,255,0.16);
                border-radius: 999px;
            }}

            .empty-history {{
                min-height: 160px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                color: var(--text-muted);
                gap: 0.4rem;
                text-align: center;
            }}

            .empty-title {{
                font-size: 1.05rem;
                color: var(--text-main);
                font-weight: 700;
            }}

            .empty-subtitle {{
                font-size: 0.92rem;
                color: var(--text-muted);
            }}

            .row {{
                display: flex;
                margin-bottom: 0.85rem;
            }}

            .row.user {{ justify-content: flex-end; }}
            .row.assistant {{ justify-content: flex-start; }}

            .bubble {{
                max-width: 82%;
                padding: 0.9rem 1rem;
                border-radius: 18px;
                border: 1px solid var(--border);
                line-height: 1.5;
                white-space: normal;
                word-break: break-word;
                overflow-wrap: anywhere;
            }}

            .bubble.user {{
                background: var(--bg-user);
                border-top-right-radius: 6px;
                text-align: left;
            }}

            .bubble.assistant {{
                background: var(--bg-assistant);
                border-top-left-radius: 6px;
                text-align: left;
            }}

            .role-tag {{
                display: block;
                font-size: 0.74rem;
                color: var(--text-muted);
                margin-bottom: 0.35rem;
                letter-spacing: 0.02em;
                text-transform: uppercase;
            }}
        </style>
    </head>
    <body>
        <div class="history-wrap">
            {empty_state}
        </div>
    </body>
    </html>
    """


def render_chat_history(messages: List[Dict[str, str]]) -> None:
    """Render the scrollable chat transcript."""
    # Height scales a bit as messages grow, then caps to keep the composer visible.
    base_height = 220
    dynamic_height = min(520, base_height + len(messages) * 92)
    st_html(_build_history_html(messages), height=dynamic_height, scrolling=False)


def render_composer() -> Tuple[bool, str]:
    """Render a centered composer and return (submitted, query)."""
    submitted = False
    query = ""

    left, center, right = st.columns([1, 3, 1])
    with center:
        with st.form("composer", clear_on_submit=True):
            query = st.text_input(
                "Query",
                value="",
                placeholder="Ask anything...",
                label_visibility="collapsed",
                key="composer_query",
            )
            submitted = st.form_submit_button("Send", use_container_width=True)

    return submitted, query


def render_diagnostics(result: Dict[str, object]) -> None:
    """Keep the original outputs available without cluttering the chat UI."""
    st.markdown('<div class="results-panel">', unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)
