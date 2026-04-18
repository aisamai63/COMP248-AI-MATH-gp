def _escape_message(content: str) -> str:
    return html.escape(content).replace("\n", "<br>")


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

            [data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="collapsedControl"] {
                display: none;
            }

            .block-container {
                max-width: 600px;
                margin: 0 auto;
                padding-top: 2rem;
                padding-bottom: 7rem;
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
                max-width: 600px;
            }

            .chat-hero h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.1;
                font-weight: 700;
            }

            .chat-hero p {
                margin: 0.55rem 0 0 0;
                color: var(--text-muted);
                font-size: 1rem;
            }

            .top-spacer {
                height: 5vh;
            }

            .intro-card {
                width: 100%;
                max-width: 600px;
                margin: 0 auto 1rem auto;
                padding: 1rem 1.15rem;
                border-radius: 14px;
                background: var(--bg-assistant);
                border: 1px solid var(--border);
                color: var(--text-main);
            }

            .intro-card .muted {
                color: var(--text-muted);
                margin-top: 0.35rem;
                font-size: 0.98rem;
            }

            /* Chat bubbles and spacing */
            .history-wrap, .chat-container {
                background: rgba(53, 55, 64, 0.12);
                border-radius: 18px;
                box-shadow: 0 2px 16px rgba(0,0,0,0.10);
                padding: 1.2rem 0.5rem 1.2rem 0.5rem;
                margin-bottom: 1.5rem;
            }

            .chat-row {
                display: flex;
                margin-bottom: 1.2rem;
            }

            .chat-row.user {
                justify-content: flex-end;
            }

            .chat-row.assistant {
                justify-content: flex-start;
            }

            .chat-bubble {
                max-width: 80%;
                padding: 1.1rem 1.3rem;
                border-radius: 18px;
                line-height: 1.6;
                font-size: 1.08rem;
                margin-bottom: 2px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.10);
                transition: box-shadow 0.2s;
            }

            .chat-bubble.user {
                background: var(--bg-user);
                color: #fff;
                border-bottom-right-radius: 6px;
                border-top-right-radius: 18px;
                border-top-left-radius: 18px;
                border-bottom-left-radius: 18px;
                box-shadow: 0 2px 8px rgba(16,163,127,0.10);
            }

            .chat-bubble.assistant {
                background: var(--bg-assistant);
                color: var(--text-main);
                border-bottom-left-radius: 6px;
                border-top-right-radius: 18px;
                border-top-left-radius: 18px;
                border-bottom-right-radius: 18px;
                box-shadow: 0 2px 8px rgba(53,55,64,0.10);
            }

            .chat-bubble:hover {
                box-shadow: 0 4px 16px rgba(16,163,127,0.13);
            }

            /* Input and send button */
            div[data-testid="stForm"] {
                width: 100%;
                max-width: 600px;
                margin: 0 auto;
                padding: 0;
            }

            div[data-testid="stForm"] > form {
                background: var(--bg-assistant);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 0.7rem 0.7rem 0.7rem 1rem;
                box-shadow: 0 2px 16px rgba(0,0,0,0.10);
                display: flex;
                align-items: center;
            }

            div[data-testid="stForm"] > form:hover {
                border-color: var(--bg-user);
                box-shadow: 0 0 0 2px var(--bg-user), 0 2px 16px rgba(0,0,0,0.12);
                transform: translateY(-1px);
            }

            div[data-testid="stTextInput"] input {
                background: #232428 !important;
                color: var(--text-main) !important;
                border: none !important;
                border-radius: 12px !important;
                padding: 1rem 1.2rem !important;
                font-size: 1.08rem !important;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                margin-right: 0.7rem;
            }

            div[data-testid="stTextInput"] input::placeholder {
                color: #7f8498 !important;
            }

            div[data-testid="stTextInput"] input:focus {
                border-color: var(--bg-user) !important;
                box-shadow: 0 0 0 2px var(--bg-user) !important;
            }

            .stButton > button,
            div[data-testid="stFormSubmitButton"] button {
                background: var(--bg-user) !important;
                color: #fff !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 0.85rem 1.2rem !important;
                font-weight: 700 !important;
                font-size: 1.08rem !important;
                box-shadow: 0 2px 8px rgba(16,163,127,0.10);
                transition: box-shadow 0.2s, filter 0.2s;
            }

            .stButton > button:hover,
            div[data-testid="stFormSubmitButton"] button:hover {
                filter: brightness(1.08);
                box-shadow: 0 0 0 2px var(--bg-user), 0 4px 16px rgba(16,163,127,0.13);
                transform: translateY(-1px);
            }

            .results-panel {
                max-width: 600px;
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
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                }
                .chat-hero {
                    margin-top: 0.75rem;
                }
                div[data-testid="stForm"] > form {
                    border-radius: 12px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
