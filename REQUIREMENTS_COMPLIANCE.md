# COMP248 Requirements Compliance Matrix

This checklist maps the assignment requirements to repository evidence and current status.

Legend:
- PASS: Implemented and evidenced in repo
- PARTIAL: Implemented in code but weak/missing explicit submission artifact
- MISSING: Not present in repo as a required final deliverable

## 1) Design Requirements

1. Multi-agent architecture (max 5 agents), responsibilities, planner/worker hierarchy:
- Status: PASS
- Evidence:
  - `design/README_design.md`
  - `prototype/workflow/graph.py`
  - `prototype/agents/planner.py`
  - `prototype/agents/retriever.py`
  - `prototype/agents/summarizer.py`
  - `prototype/agents/reflective_agent.py`
  - `prototype/agents/tool_agent.py`

2. RAG capability (naive), one field source, performance measures:
- Status: PASS
- Evidence:
  - `prototype/chroma_setup.py`
  - `prototype/agents/retriever.py`
  - `prototype/db.py`
  - `prototype/data/sample_docs.jsonl`
  - `prototype/data/*.pdf`
  - `prototype/evaluation.py` (mean top-k similarity, reflection confidence)

3. Reflection capability (introspection), metrics + workflow:
- Status: PASS
- Evidence:
  - `prototype/agents/reflective_agent.py`
  - `prototype/workflow/graph.py`
  - `design/README_design.md`

4. Main knowledge base details, storage usage, data model:
- Status: PARTIAL
- Evidence:
  - `prototype/chroma_setup.py`
  - `prototype/db.py`
  - `design/README_design.md`
- Gap:
  - Data model is described narratively but no explicit diagram/table schema artifact in report folder.

5. UML component + interaction (sequence) diagrams:
- Status: PARTIAL
- Evidence:
  - `design/README_design.md` (sequence in Mermaid)
  - `COMP248_AI_MATH_GP_Presentation_Diagrams.pdf` (diagram artifact)
- Gap:
  - Ensure final report PDF includes both component-level and interaction diagrams explicitly labeled.

6. Impact of design decisions on privacy/fairness/explainability/responsible AI:
- Status: PARTIAL
- Evidence:
  - `design/README_design.md` section "Responsible AI Notes"
- Gap:
  - Fairness and governance discussion should be expanded in final report.

## 2) Prototype Requirements

1. Use CrewAI (LangChain & LangGraph):
- Status: PASS
- Evidence:
  - `prototype/crewai_wrapper.py` (CrewAI runtime path)
  - `prototype/workflow/graph.py` (LangGraph orchestration)
  - `prototype/agents/summarizer.py` + `prototype/agents/reflective_agent.py` (LangChain prompt templates)

2. Use ChromaDB for naive RAG:
- Status: PASS
- Evidence:
  - `prototype/chroma_setup.py`
  - `prototype/db.py`
  - `prototype/ingest.py`

3. Use Streamlit for frontend:
- Status: PASS
- Evidence:
  - `prototype/app.py`
  - `prototype/ui/chat_ui.py`

4. Demonstrate one agent with tool use and the reflective agent:
- Status: PASS
- Evidence:
  - `prototype/agents/tool_agent.py`
  - `prototype/tools/builtin_tools.py`
  - `prototype/agents/reflective_agent.py`
  - `prototype/workflow/graph.py`

## 3) Report Requirements

Required sections:
- Cover page
- Table of contents
- Rationale and scope
- Research results
- Design documents
- Conclusion
- Assumptions
- References
- Appendix 1 meeting register

Status: MISSING in repo as a dedicated final report file.
Gap:
- No final report PDF found in repo root except diagram PDF.
- Add final report PDF before submission.

## 4) Presentation Requirements

1. Team participation in presentation:
- Status: OUT OF CODE SCOPE (team process item)

2. Present working code:
- Status: PASS (code present; runtime depends on valid API/network environment)

3. PowerPoint no more than 8 slides:
- Status: MISSING in repo
- Gap:
  - No `.ppt`/`.pptx` submission artifact found.

## 5) Submission Packaging Requirements

1. One zip, naming rule:
- Status: PARTIAL
- Evidence:
  - `COMP248_project_Research and summarization in the field of mathematical enquiries_001.zip`
- Gap:
  - Team number suffix appears missing (`_team#`).

2. Contents required in zip:
- Python scripts: PASS
- Supporting files: PASS
- PowerPoint slides: MISSING
- Project analysis report PDF: MISSING

## Immediate Action List Before Final Submission

1. Add final project report PDF including all required report sections.
2. Add final presentation `.pptx` (<= 8 slides).
3. Add Appendix 1 meeting register to report appendices.
4. Confirm/rename zip to include team number suffix per naming rule.
5. Validate demo runtime in a clean environment with working network and API keys.
