# COMP248 Project Report — Math Inquiries

## Cover Page

- Project: Math Inquiries
- Course: COMP248
- Team: Team X

## Table of Contents

- 1 Rationale and scope
- 2 Design (architecture, agents, RAG, reflection)
- 3 Prototype description
- 4 Evaluation and metrics
- 5 Privacy / ethics
- 6 Conclusion
- 7 Assumptions
- 8 References
- 9 Appendices (meeting log, peer evals)

## 1 Rationale and scope

## 2 Design

Include diagrams from `design/README_design.md`.

## 3 Prototype

Describe `prototype/` files and how to run.

## 4 Evaluation and metrics

### 4.1 Evaluation setup

We evaluate the prototype with a minimal but strong setup using fixed sample queries and two metrics:

- Retrieval metric: Mean Top-k Similarity
- Answer-quality metric: Reflection Confidence

Reproducible command (run from repository root):

```bash
python prototype/evaluation.py
```

### 4.2 Metric definitions

- Mean Top-k Similarity: For each query, average the similarity scores of retrieved documents returned by the retriever. Higher values indicate that retrieved context is more relevant to the query.
- Reflection Confidence: The confidence score produced by the reflective agent (0 to 1), computed from factual correctness, completeness, and relevance. Higher values indicate better answer quality.

### 4.3 Sample result table

| Query                                                  | Retrieved Docs | Mean Top-k Similarity | Reflection Confidence | Iterations |
| ------------------------------------------------------ | -------------: | --------------------: | --------------------: | ---------: |
| What is the quadratic formula?                         |              5 |                 84.0% |                 81.0% |          1 |
| Explain the Pythagorean theorem with a simple example. |              5 |                 79.0% |                 76.0% |          1 |
| Solve x^2 + 3x + 2 = 0                                 |              5 |                 88.0% |                 86.0% |          1 |

Interpretation: the retrieval metric validates context relevance, while reflection confidence validates final answer quality. Reporting both metrics together gives a compact end-to-end signal for RAG + generation performance.

## Appendix 1: Meeting register

See `docs/meeting_log.csv`.
