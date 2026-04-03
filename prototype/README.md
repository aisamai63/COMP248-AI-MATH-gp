# Prototype — Math Inquiries

Run the Streamlit demo that showcases: Planner -> Retriever -> SummarizerAgent -> ReflectiveAgent -> ToolAgent.

If `USE_CREWAI=true`, the same LangGraph workflow is executed through a thin CrewAI wrapper so the prototype satisfies the CrewAI requirement without changing the underlying reasoning pipeline. If you use the default Mistral configuration, install LiteLLM support as noted in `prototype/requirements.txt`.

Setup

1. Create a virtualenv and activate it.
2. Install dependencies:

```
pip install -r prototype/requirements.txt
```

1. Copy `.env.example` to `.env` and set `MISTRAL_API_KEY` if you want LLM summarization.

Run

```
streamlit run prototype/app.py
```

Notes

- Persistent ChromaDB retrieval is enabled through `db.py` and the configured collection.
- If the collection is empty, run `python prototype/ingest.py` from repo root or `python ingest.py` from `prototype/`.
- If `MISTRAL_API_KEY` is not set, the app falls back to the built-in naive summarizer.
- Set `USE_CREWAI=true` to exercise the CrewAI wrapper over the LangGraph workflow.
