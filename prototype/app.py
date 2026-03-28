"""Streamlit demo app for the Math Inquiries prototype."""

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
    st.title("Math Inquiries — Prototype Demo")
    st.write(
        "Research Analyst demo: enter a math query and see retrieval + summary + reflection."
    )

    query = st.text_input("Query", "What is the quadratic formula?")
    if st.button("Run"):
        kb = KBClient()
        planner = Planner(kb)
        result = planner.handle_query(query)

        st.subheader("Retrieved Documents")
        for d in result["docs"]:
            st.write(f"**{d.get('title')}** — {d.get('source')}")
            st.write(d.get("text"))

        summarizer = SummarizerAgent()
        summary = summarizer.summarize(result["docs"])
        st.subheader("Summary")
        st.write(summary)

        reflector = ReflectiveAgent()
        reflection = reflector.reflect(summary, result["docs"])
        st.subheader("Reflection")
        st.json(reflection)


if __name__ == "__main__":
    main()
