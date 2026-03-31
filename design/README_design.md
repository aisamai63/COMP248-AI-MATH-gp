# Math Inquiries - Design Document

## Overview

This document captures the multi-agent architecture, agent hierarchy, detailed workflows, RAG and reflection design, data model, metrics, diagrams, and responsible AI considerations for the Math Inquiries project (COMP248).

---

## 1. Multi-Agent Architecture

### Agent Hierarchy

| Level | Agent Name | Responsibility |
| --- | --- | --- |
| Planner | Planner | Receives user queries, routes to workers, composes tasks, and manages workflow |
| Worker | SearchAgent | Retrieves relevant documents from Chromadb (RAG) |
| Worker | SummarizerAgent | Summarizes retrieved documents using an LLM |
| Worker | ReflectiveAgent | Evaluates summary quality and suggests re-runs |
| Worker | ToolAgent (optional) | Interfaces with external tools and APIs |

**Hierarchy Diagram:**

```mermaid
graph TD
  Planner --> SearchAgent
  Planner --> SummarizerAgent
  Planner --> ReflectiveAgent
  Planner --> ToolAgent
```

### Agent Details

| Agent | LLM/Tool Choice and Justification | Workflow Diagram |
| --- | --- | --- |
| Planner | No LLM initially (rule-based); may use a small LLM for rephrasing. Tool: workflow router. | See below |
| SearchAgent | Chromadb, embedding model: `sentence-transformers/all-MiniLM-L6-v2` (fast, small footprint). | See below |
| SummarizerAgent | LLM: OpenAI (`gpt-4o` or `gpt-4`) or Llama2 (local, if budget constrained). Chosen for summarization quality. | See below |
| ReflectiveAgent | LLM-based microchain that evaluates summary coverage, factuality, redundancy, and length. | See below |
| ToolAgent | External APIs (web search, calculators) as needed. | See below |

#### Planner Workflow

```mermaid
flowchart TD
  UserQuery --> Planner
  Planner -->|route| SearchAgent
  SearchAgent --> Planner
  Planner -->|route| SummarizerAgent
  SummarizerAgent --> Planner
  Planner -->|quality check| ReflectiveAgent
  ReflectiveAgent --> Planner
  Planner -->|if needed| ToolAgent
  ToolAgent --> Planner
  Planner --> FinalResponse
```

#### SearchAgent Workflow

```mermaid
flowchart TD
  Planner --> SearchAgent
  SearchAgent --> EmbedQuery
  EmbedQuery --> Chromadb
  Chromadb --> TopKDocs
  TopKDocs --> Planner
```

#### SummarizerAgent Workflow

```mermaid
flowchart TD
  Planner --> SummarizerAgent
  SummarizerAgent --> PromptBuilder
  PromptBuilder --> LLM
  LLM --> DraftSummary
  DraftSummary --> Planner
```

#### ReflectiveAgent Workflow

```mermaid
flowchart TD
  Planner --> ReflectiveAgent
  ReflectiveAgent --> CheckCoverage
  ReflectiveAgent --> CheckFactuality
  ReflectiveAgent --> CheckRedundancy
  ReflectiveAgent --> CheckLength
  CheckCoverage --> ConfidenceScore
  CheckFactuality --> ConfidenceScore
  CheckRedundancy --> ConfidenceScore
  CheckLength --> ConfidenceScore
  ConfidenceScore --> Planner
```

#### ToolAgent Workflow

```mermaid
flowchart TD
  Planner --> ToolAgent
  ToolAgent --> SelectTool
  SelectTool --> ExternalAPI
  ExternalAPI --> ToolAgent
  ToolAgent --> Planner
```

---

## 2. RAG (Naive)

- **Index:** Chromadb vector index storing documents and metadata.
- **Retriever:** Top-k by cosine similarity (`k = 5`).
- **Generator:** SummarizerAgent condenses retrieved results.
- **Information Source:** `prototype/data/sample_docs.jsonl` (math-related documents).

**Performance Measures:**

- Recall@k for retrieval (measures whether relevant docs are returned).
- ROUGE and BLEU for summarization (if reference summaries exist).
- Factuality/hallucination rate (via question-answer checking).

**Justification:** These are standard measures in RAG and summarization literature.

---

## 3. Reflection Capability

- **Metrics Used:** Coverage, redundancy, length (conciseness), and factuality.
- **Workflow:** After summarization, ReflectiveAgent computes metrics and a confidence score. If below threshold, Planner triggers another retrieval or expands `k`.

---

## 4. Routing Capability

- **Routing Logic:** Planner uses rule-based logic to assign tasks to workers based on query type and workflow state.

---

## 5. Knowledge Base

- **Primary Store:** Chromadb (embeddings, documents, and metadata).
- **Backup:** JSONL file for demo/fallback (`prototype/data/sample_docs.jsonl`).

**Data Model:**

| Field | Type | Description |
| --- | --- | --- |
| id | str | Unique document ID |
| text | str | Document content |
| title | str | Document title |
| source | str | Source of document |
| embedding | vector | Embedding vector |

**ER Diagram:**

```mermaid
erDiagram
  DOCUMENT {
    string id PK
    string text
    string title
    string source
    vector embedding
  }
```

---

## 6. Diagrams

### UML Component Diagram

```mermaid
graph TD
  User[Research Analyst] -->|query| Planner
  Planner --> SearchAgent
  SearchAgent --> Chromadb[(Chromadb KB)]
  Planner --> SummarizerAgent
  SummarizerAgent --> ReflectiveAgent
  ReflectiveAgent --> Planner
  SummarizerAgent --> User
```

### UML Sequence Diagram

```mermaid
sequenceDiagram
  participant U as User
  participant P as Planner
  participant S as SearchAgent
  participant G as SummarizerAgent
  participant R as ReflectiveAgent

  U->>P: Submit query
  P->>S: Retrieve top-k
  S-->>P: Return documents
  P->>G: Summarize documents
  G-->>P: Return summary
  P->>R: Request reflection
  R-->>P: Return reflection score
  P->>U: Final summary and reflection
```

---

## 7. Responsible AI: Privacy, Fairness, Explainability

| Principle | Design Impact |
| --- | --- |
| Privacy | Minimize PII in KB; remove sensitive content from sources; redact logs. |
| Fairness | Use diverse sources; avoid bias in sample docs and LLM prompts. |
| Explainability | Provide provenance (`source`, `doc_id`) for every chunk; log agent decisions for review. |
| Responsible AI | Use only open, non-sensitive data; document all design choices and limitations. |

---

## 8. Files Referenced

- `prototype/agents.py` - agent implementations (simple prototypes).
- `prototype/db.py` - Chromadb interactions.
- `prototype/app.py` - Streamlit demo.

---

End of revised design document.
