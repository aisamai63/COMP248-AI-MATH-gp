# COMP248_math_inquiries

Math Inquiries — multi-agent research and summarization prototype for COMP248.

## Quick overview

This repository contains a design document and a minimal prototype demonstrating a Planner, SearchAgent (naïve RAG), SummarizerAgent, and ReflectiveAgent. The demo uses Streamlit for the frontend and Chromadb (optional) or local JSON/PDF files for the knowledge base.

## Quickstart (Windows)

1. Create and activate a virtual environment (from project root):

```powershell
python -m venv venv
& .\venv\Scripts\Activate.ps1
```

1. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r prototype/requirements.txt
```

1. Run the Streamlit demo:

```powershell
streamlit run prototype/app.py
```

Open the Local URL printed by Streamlit (usually <http://localhost:8501>).

## Adding data

- Place PDF files or `.jsonl` documents in `prototype/data/`. PDFs will be read if `pypdf` is installed.

## Project structure

- `design/` — design document, diagrams, architecture notes ([design/README_design.md](design/README_design.md))
- `prototype/` — prototype code and demo ([prototype/README.md](prototype/README.md))
- `report/` — report template for the course ([report/COMP248_report_template.md](report/COMP248_report_template.md))
- `docs/meeting_log.csv` — meeting register template

## Notes & next steps

- Chromadb integration is currently stubbed; to enable full RAG, configure a running Chromadb instance and update `prototype/db.py`.
- Set `OPENAI_API_KEY` in `prototype/.env` if you plan to call OpenAI models.

## GitHub

Repository pushed to: <https://github.com/aisamai63/COMP248_math_inquiries.git>

## Contact

For questions, open an issue on the repo or contact the project owner.
