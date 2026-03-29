"""
Streamlit demo app for the Math Inquiries prototype.
Demonstrates: query input, retrieval, summarization, and reflection.
"""

import sys
import pathlib
import streamlit as st

# Ensure project root is on sys.path so `prototype` package imports work
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.db import KBClient
from prototype.agents import Planner, SearchAgent, SummarizerAgent, ReflectiveAgent


def main():
    """
    Main entry point for the Streamlit demo app.
    """
    st.title("Math Inquiries — Prototype Demo")
    st.write(
        "Research Analyst demo: enter a math query and see retrieval, summary, and reflection."
    )

    query = st.text_input("Query", "What is the quadratic formula?")
    k = st.slider(
        "Number of documents to retrieve (top-k)", min_value=1, max_value=10, value=5
    )

    max_tokens = st.slider(
        "Summary length (max tokens)", min_value=40, max_value=512, value=120, step=10
    )
    llm_provider = st.selectbox(
        "LLM Provider for Summarization",
        options=["openai", "gemini", "none"],
        index=0,
        format_func=lambda x: {
            "openai": "OpenAI (GPT)",
            "gemini": "Gemini (Google)",
            "none": "Naive (no LLM)",
        }[x],
    )

    if st.button("Run"):
        with st.spinner("Retrieving documents..."):
            kb = KBClient()
            planner = Planner(kb)
            result = planner.handle_query(query)
            docs = result["docs"][:k]

        st.subheader("Retrieved Documents")
        if docs:
            for d in docs:
                st.write(
                    f"**{d.get('title', 'Untitled')}** — {d.get('source', 'unknown')}"
                )
                st.write(d.get("text", ""))
        else:
            st.info("No documents found for this query.")

        with st.spinner("Summarizing..."):
            summarizer = SummarizerAgent(
                use_llm=llm_provider if llm_provider != "none" else None,
                max_tokens=max_tokens,
            )
            summary = summarizer.summarize(docs, max_tokens=max_tokens)
        st.subheader("Summary")
        st.write(summary)

        with st.spinner("Reflecting on summary..."):
            reflector = ReflectiveAgent()
            reflection = reflector.reflect(summary, docs)
        st.subheader("Reflection Metrics")
        st.write(f"Confidence: {reflection.get('confidence', 0)}%")
        st.write(f"Redundancy: {reflection.get('redundancy', 0)}%")
        st.write(f"Summary Length: {reflection.get('summary_length', 0)} words")
        st.write("Notes:")
        for note in reflection.get("notes", []):
            st.write(f"- {note}")


if __name__ == "__main__":
    main()
