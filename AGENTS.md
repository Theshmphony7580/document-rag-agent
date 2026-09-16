# AGENTS.md

## Operational Rules
1. **Execution Environment:** Do not run terminal commands without explicit user instruction. When running commands, always execute inside the project virtual environment (`.venv`).
2. **Progress Tracking:** Update this file (`AGENTS.md`) across every turn to preserve context, architectural decisions, and current task state.
3. **Communication Style:** Follow ADHD guidelines (lead with action, cap lists at 5, no fluff, clear 1-line next actions).

---

## Project Overview
- **Project:** Enterprise Document RAG Agent
- **Core Orchestrator:** LangGraph state graph with dynamic self-correction loops
- **Vector Database:** Qdrant (local `:memory:` / directory / remote server)
- **Embedding Model:** Google Gemini (`gemini-embedding-001` / `gemini-embedding-2`, 768-dim) with deterministic local test fallback
- **Inference Models:** Groq (`llama-3.3-70b-versatile`) & Google Gemini (`gemini-2.5-flash`)
- **Document Parsing:** IBM Docling (`HierarchicalChunker` + `TableFormer`) & Gemini Vision for figures
- **Package Manager:** `uv` with Python 3.12+

---

## Architecture Diagrams

### 1. Agent Reasoning Loop (LangGraph State Machine)
```
[User Query] ──► [retrieve_node] ──► [grade_node] ──┬──► (score >= 0.70 or max retries) ──► [generate_node] ──► [Answer]
                                                    └──► (score < 0.70 & retry < 1)     ──► [rewrite_node]  ──► [retrieve_node]
```

### 2. Document Ingestion Pipeline
```
[Document Upload] ──► [SHA-256 Check] ──┬──► (Already Exists) ──► [Skip / Fast Return]
                                        └──► (New Document)   ──► [Docling Parser (do_ocr=False)]
                                                                          │
                        ┌─────────────────────────────────────────────────┴─────────────────────────────────────────────────┐
                        ▼                                                                                                   ▼
             [Text & Table Elements]                                                                             [Cropped Charts / Visuals]
                        │                                                                                                   │
                        ▼                                                                                                   ▼
         [Docling Hierarchical Chunker]                                                                           [Gemini Vision (VLM)]
     (Preserves headers & whole tables)                                                                          (Generates text summary)
                        │                                                                                                   │
                        └─────────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                                                  │
                                                                  ▼
                                                      [Metadata Injector]
                                           (doc_hash, page_num, section_path, type)
                                                                  │
                                                                  ▼
                                                       [Gemini Dense Embedder]
                                                        (gemini-embedding-001)
                                                                  │
                                                                  ▼
                                                         [Qdrant Vector DB]
                                                        (Upsert PointStruct)
```

---

## Data Schemas & Standards

### Chunk Metadata Schema
```python
{
    "doc_id": str,          # Deterministic document identifier
    "doc_hash": str,        # SHA-256 hex digest of source file
    "chunk_id": str,        # Deterministic ID: {doc_hash}_{chunk_index}
    "source_file": str,     # Source filename
    "page_numbers": list,   # List of page integers [int]
    "section_path": str,    # Header breadcrumbs: "H1 > H2 > H3"
    "content_type": str,    # "text" | "table" | "diagram_vlm"
    "created_at": str       # ISO-8601 UTC timestamp
}
```

### LangGraph State ([`src/agent/state.py`](src/agent/state.py))
- `question: str`: Current user query.
- `chat_history: List[BaseMessage]`: Multi-turn conversational context.
- `retrieved_chunks: List[DocumentChunk]`: Top-k chunks fetched from Qdrant.
- `confidence_score: float`: Graded relevance score (0.0 to 1.0).
- `rewritten_question: Optional[str]`: Reformulated query if retrieval fails threshold.
- `answer: str`: Final grounded response with citations.
- `retry_count: int`: Number of query rewrites executed (capped at 1).

---

## Ingestion Pre-initialization & Key Requirements

| Resource / Key | Purpose | Required For Ingestion? | Default / Fallback |
|---|---|---|---|
| `GEMINI_API_KEY` | Dense embeddings (`gemini-embedding-001`) + VLM chart summaries | **Yes** | Local mock embedder available for offline testing |
| `GROQ_API_KEY` | Fast LLM inference (Llama 3.3) | No (Deferred to Agent phase) | None |
| `QDRANT_PATH` | Vector store backend on local disk | **Yes** | `"./qdrant_data"` (auto-created on disk) |
| `QDRANT_API_KEY` | Qdrant Cloud auth token | Optional | Empty for local disk |
| `QDRANT_COLLECTION_NAME` | Target collection name | **Yes** | `"knowledge_chunks"` (auto-created if missing) |
| Collection Specs | Cosine metric, 768 vector dimensions, keyword index on `doc_hash` | **Auto-initialized** | Handled on first run in `ensure_collection()` |

---

## File Structure & Map

```
document-rag/
├── .env.example                       # Environment variable templates
├── .gitignore                         # Git ignore file (excludes qdrant_data/, .venv)
├── AGENTS.md                          # Context, rules & progress tracking
├── list_gemini_models.py              # Root utility: list available Gemini models & types
├── prd.md                             # Original system PRD & technical architecture
├── pyproject.toml                     # uv project & dependency definitions
├── qdrant_data/                       # Local Qdrant disk storage (auto-created)
└── src/
    ├── __init__.py                    # Source root package
    ├── config.py                      # Pydantic-settings loader
    ├── schemas.py                     # DocumentChunk, QueryRequest, QueryResponse
    ├── agent/
    │   ├── __init__.py                # Agent module
    │   ├── state.py                   # LangGraph RAGState TypedDict
    │   ├── nodes.py                   # [Planned] retrieve, grade, generate, rewrite
    │   └── graph.py                   # [Planned] StateGraph assembly & compilation
    ├── ingestion/
    │   ├── __init__.py                # Ingestion module
    │   ├── embeddings.py              # Gemini text-embedding-004 + mock fallback
    │   ├── parser.py                  # Docling parser (do_ocr=False) + HierarchicalChunker
    │   ├── pipeline.py                # IngestionPipeline orchestrator (SHA-256 deduplication)
    │   └── vision.py                  # VLM figure captioner (prompt placeholder empty)
    └── storage/
        ├── __init__.py                # Storage module
        └── vector_store.py            # QdrantVectorStore disk adapter
```

### Component Status
|---|---|---|
| [`list_gemini_models.py`](list_gemini_models.py) | Done | Root utility to list all models & capabilities for GEMINI_API_KEY |
| [`pyproject.toml`](pyproject.toml) | Done | Project dependencies managed via `uv` |
| [`.env.example`](.env.example) | Done | Environment variable templates (Groq, Gemini, Qdrant on disk) |
| [`.vscode/settings.json`](.vscode/settings.json) | Done | Interpreter and import search paths |
| [`src/schemas.py`](src/schemas.py) | Done | Pydantic contracts (`DocumentChunk`, `QueryRequest`, `QueryResponse`) |
| [`src/agent/state.py`](src/agent/state.py) | Done | LangGraph `RAGState` TypedDict definition |
| [`src/config.py`](src/config.py) | Done | Settings loader via `pydantic-settings` |
| [`src/storage/vector_store.py`](src/storage/vector_store.py) | Done | Qdrant disk client adapter (ensure_collection, upsert, search) |
| [`src/ingestion/vision.py`](src/ingestion/vision.py) | Done | VLM figure captioner (Forensic Document Intelligence prompt + Gemini Vision) |
| [`src/ingestion/embeddings.py`](src/ingestion/embeddings.py) | Done | Gemini embeddings (`gemini-embedding-001`) + deterministic mock fallback |
| [`src/ingestion/parser.py`](src/ingestion/parser.py) | Done | Docling parser (`do_ocr=False`) + VLM diagram integration + text fallback |
| [`src/ingestion/pipeline.py`](src/ingestion/pipeline.py) | Done | Ingestion orchestrator (`ingest_file`) with SHA-256 deduplication |
| [`src/agent/nodes.py`](src/agent/nodes.py) | Planned | LangGraph node functions (`retrieve`, `grade`, `generate`, `rewrite`) |
| [`src/agent/graph.py`](src/agent/graph.py) | Planned | StateGraph compilation with conditional self-correction edge |

---

## Current Status Checklist

- [x] Scaffolding and minimal dependencies installed via `uv`.
- [x] Defined core data models in [`src/schemas.py`](src/schemas.py).
- [x] Defined graph state in [`src/agent/state.py`](src/agent/state.py).
- [x] Configured interpreter and search paths in [`.vscode/settings.json`](.vscode/settings.json).
- [x] Configured [`.env.example`](.env.example) with local disk Qdrant storage.
- [x] Architected document ingestion pipeline (Docling chunking, SHA-256 hashing, metadata schema, `do_ocr=False`).
- [x] Implemented [`src/config.py`](src/config.py) with Pydantic settings loading disk paths.
- [x] Implemented [`src/storage/vector_store.py`](src/storage/vector_store.py) with Qdrant disk storage and `doc_hash` index.
- [x] Implemented [`src/ingestion/vision.py`](src/ingestion/vision.py) with finalized Forensic Document Intelligence prompt.
- [x] Standardized root imports across `src/` to align with project layout and eliminate linter warnings.
- [x] Implemented [`src/ingestion/embeddings.py`](src/ingestion/embeddings.py) with live Gemini REST & offline mock.
- [x] Implemented [`src/ingestion/parser.py`](src/ingestion/parser.py) with Docling `HierarchicalChunker` & metadata.
- [x] Implemented [`src/ingestion/pipeline.py`](src/ingestion/pipeline.py) orchestrator with SHA-256 deduplication.
- [x] Ingestion pipeline smoke test executed and verified on local disk (parsing, deduplication, vector search).
- [ ] **NEXT:** Scaffold and implement LangGraph nodes in `src/agent/nodes.py`.
