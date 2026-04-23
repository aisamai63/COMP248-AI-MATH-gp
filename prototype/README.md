# Prototype — Math Inquiries

Run the Streamlit demo that showcases: Planner -> Retriever -> SummarizerAgent -> ReflectiveAgent -> ToolAgent.

If `USE_CREWAI=true`, the same LangGraph workflow is executed through a thin CrewAI wrapper so the prototype satisfies the CrewAI requirement without changing the underlying reasoning pipeline.

Setup

1. Create a virtualenv and activate it.
2. Install dependencies:

```
pip install -r prototype/requirements.txt
```

1. Copy `.env.example` to `.env` and set provider + key (`LLM_PROVIDER` with matching API key) for LLM summarization.

Run

```
.\.venv\Scripts\python.exe -m streamlit run prototype/app.py
```

If `langgraph` (or any dependency) is missing, install with the same interpreter:

```
.\.venv\Scripts\python.exe -m pip install -r prototype/requirements.txt
```

Notes

- Persistent ChromaDB retrieval is enabled through `db.py` and the configured collection.
- If the collection is empty, run `python prototype/ingest.py` from repo root or `python ingest.py` from `prototype/`.
- If the key for the active provider is not set, the app falls back to the built-in naive summarizer.
- Set `USE_CREWAI=true` to exercise the CrewAI wrapper over the LangGraph workflow.
