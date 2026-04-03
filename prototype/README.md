# Prototype — Math Inquiries

Run the Streamlit demo that showcases: Planner -> Retriever -> SummarizerAgent -> ReflectiveAgent -> ToolAgent.

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
