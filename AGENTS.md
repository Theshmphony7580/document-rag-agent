# AGENTS.md

## Operational Rules
1. **Execution Environment:** Do not run terminal commands without explicit user instruction. When running commands, always execute inside the project virtual environment (`.venv`).
2. **Progress Tracking:** Update this file (`AGENTS.md`) across turns to preserve context, decisions, and task state.
3. **Communication Style:** Follow ADHD guidelines (lead with action, cap lists at 5, no fluff, clear 1-line next actions).

---

## Project Overview
- **Project:** Document RAG (LangGraph + Qdrant)
- **Current Scope:** Baseline RAG (Dense vector retrieval first; BM25 and hybrid search deferred).
- **Core Stack:**
  - Framework: Python 3.12+ with `langgraph`, `langchain-core`
  - LLMs / Providers: **Groq** (`llama-3.3-70b-versatile`) & **Google Gemini** (`gemini-2.5-flash`)
  - Embeddings: **Google Gemini** (`text-embedding-004`, 768-dim) or local fallback
  - Vector DB: `qdrant-client` (local/in-memory mode for dev)
  - Data Validation: `pydantic`

---

## Current Status
- [x] Scaffolding and minimal dependencies installed via `uv`.
- [x] Defined core data models in [`src/schemas.py`](src/schemas.py) (`DocumentChunk`, `QueryRequest`, `QueryResponse`).
- [x] Defined graph state in [`src/agent/state.py`](src/agent/state.py) (`RAGState`).
- [x] Configured interpreter and search paths in [`.vscode/settings.json`](.vscode/settings.json).
- [x] Configured [`.env.example`](.env.example) for Groq and Gemini.
- [ ] **NEXT:** Build Qdrant dense vector store adapter in `src/storage/vector_store.py`.
- [ ] Build LangGraph nodes (`retrieve`, `grade`, `generate`, `rewrite`) in `src/agent/nodes.py`.
- [ ] Assemble and compile state graph in `src/agent/graph.py`.

---

## Architecture Diagram (Current Baseline)
```
[User Query] ──► [retrieve_node] ──► [grade_node] ──┬──► (score >= 0.7) ──► [generate_node] ──► [Answer]
                                                    └──► (score < 0.7)  ──► [rewrite_node]  ──► [retrieve_node]
```
