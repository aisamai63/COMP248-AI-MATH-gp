# Prototype — Math Inquiries

Run the Streamlit demo that showcases: Planner -> SearchAgent -> SummarizerAgent -> ReflectiveAgent.

Setup

1. Create a virtualenv and activate it.
2. Install dependencies:

```
pip install -r prototype/requirements.txt
```

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you intend to use OpenAI.

Run

```
streamlit run prototype/app.py
```

Notes

- This is a minimal demo for coursework. Chromadb integration is stubbed; replace with a running chroma instance for full RAG.
