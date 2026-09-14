# Technical Architecture Document & PRD: Enterprise Knowledge Agent

**Document Status:** Approved for Implementation

**Target Systems:** Multimodal Internal Knowledge Base, Relational Schemas, APIs

**Core Function:** Multi-hop reasoning, cross-format document retrieval, and end-to-end operational observability.

---

## 1. Product Overview & System Objectives

The Enterprise Knowledge Agent serves as a centralized intelligence layer connecting company documentation (PDFs, slide decks, technical architecture diagrams, wikis) with operational databases and codebases.

### Key Capabilities

* **Heterogeneous Ingestion:** Autonomous extraction and structured indexing of visual diagrams, complex financial/operational tables, raw code, and unstructured text.
* **Deterministic Intent Routing:** Separation of document Q&A from direct SQL/API telemetry calls to eliminate retrieval overhead.
* **Continuous System Telemetry:** 100% trace capture across all agent steps, tool executions, and LLM inference calls.

---

## 2. System Architecture & Component Diagram

```
                                [ CLIENT INTERFACES ]
                            (Web Portal, Slack, Internal API)
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION PIPELINE (Asynchronous / Event-Driven)                         │
│                                                                             │
│   Sources ──► [ Docling Engine ] ──┬──► [ Text Chunker ]   ──► Vector Index │
│   (PDF/Wiki/   (Layout + Tables)   ├──► [ VLM Captioner ]  ──► Vector Index │
│    Diagrams)                       ├──► [ Markdown Tables] ──► BM25 Index   │
│                                    └──► [ Entity Mapper ]  ──► Graph DB     │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. AGENT REASONING ENGINE (LangGraph State Machine)                         │
│                                                                             │
│                    ┌────────────────────────────────┐                       │
│                    │    Query Classifier / Router   │                       │
│                    └───────────────┬────────────────┘                       │
│                                    │                                        │
│          ┌─────────────────────────┼─────────────────────────┐              │
│          ▼                         ▼                         ▼              │
│   [ Hybrid RAG Tool ]      [ Text-to-SQL Tool ]     [ API Discovery Tool ]  │
│   (Dense + BM25 + Rerank)  (Schema Introspection)   (Live System Telemetry) │
│          │                         │                         │              │
│          └─────────────────────────┼─────────────────────────┘              │
│                                    ▼                                        │
│                    ┌────────────────────────────────┐                       │
│                    │     Response Synthesizer       │                       │
│                    └────────────────────────────────┘                       │
└───────────────────────────────────────────┬─────────────────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. OBSERVABILITY & METRICS PLATFORM                                         │
│                                                                             │
│   • Tracing: OpenTelemetry SDK ──► Langfuse Collector                       │
│   • Infrastructure: Prometheus Exporter ──► Grafana (Latency, Token, Cost)  │
│   • Quality Evals: Continuous Ragas Pipeline (Context Precision, Recall)    │
└─────────────────────────────────────────────────────────────────────────────┘

```

---

## 3. Data Ingestion & Storage Architecture

### Multi-Format Parsing (IBM Docling)

* **Text & Markdown:** Standard layout analysis with reading-order preservation.
* **Tables:** Processed via TableFormer to produce dual outputs: raw HTML for structural rendering and JSON representations for numerical evaluation.
* **Diagrams & Architecture Visuals:** Isolated bounding boxes are sent to a Vision-Language Model (`gpt-4o-mini` or `pixtral-12b`) to generate rich textual descriptions and system topology summaries.

### Hybrid Storage Topology

* **Vector Database (Qdrant):** Stores 1024-dimensional dense vectors using cosine distance over HNSW indexing. Houses text chunks, table summaries, and diagram descriptions.
* **Lexical Index (OpenSearch / Elasticsearch):** BM25 sparse index tracking exact entity names, function signatures, internal ticket identifiers, and error strings.
* **Graph Store (Neo4j):** Nodes represent services, teams, repositories, and documentation files; edges define dependencies (`DEPENDS_ON`, `MAINTAINED_BY`, `DOCUMENTED_IN`).

---

## 4. Agent Reasoning Engine (SOP)

The agent runs as a deterministic directed state machine using **LangGraph**.

```
[Start] ──► [Triage Intent] ──► [Select Tool] ──► [Execute Tool] ──► [Evaluate Confidence]
                                                                            │
                       ┌────────────────────────────────────────────────────┘
                       ▼ (Low Confidence: < 0.70)      ▼ (Pass: >= 0.70)
               [Rewrite Query]                 [Synthesize Response] ──► [Emit]

```

### Standard Operating Execution Steps

1. **State Initialization:** The incoming message is assigned a globally unique `trace_id` and loaded into state with conversation context.
2. **Intent Classification:**
* If the query targets internal policies, design docs, or architecture diagrams $\rightarrow$ route to `Hybrid_RAG_Tool`.
* If the query targets live usage metrics, inventory, or operational rows $\rightarrow$ route to `Text_to_SQL_Tool`.
* If ambiguous $\rightarrow$ invoke query decomposition to generate targeted parallel searches.


3. **Hybrid Retrieval:**
* Query dense and sparse indexes in parallel.
* Merge result lists using Reciprocal Rank Fusion (RRF):

$$RRF(d) = \sum_{m \in M} \frac{1}{60 + r_m(d)}$$


* Pass top-25 merged results to a cross-encoder reranker (e.g., `bge-reranker-large`) to retain the top-5 chunks.


4. **Self-Correction (Confidence Loop):**
* Check retrieved chunk relevance score. If the highest score falls below `0.70`, route to `Query_Rewriter` and re-retrieve once before falling back to manual escalation.


5. **Synthesis:** Combine retrieved context and source metadata into a structured response containing direct inline citations.

---

## 5. Observability, Telemetry & Evaluation Specification

Observability is embedded directly into the execution flow rather than handled post-hoc.

### Distributed Tracing (OpenTelemetry + Langfuse)

Every component is wrapped with an OpenTelemetry tracer. Spans must follow this hierarchy:

| Span Level | Captured Attributes |
| --- | --- |
| **Root (`agent.request`)** | `user_id`, `session_id`, `trace_id`, `total_duration_ms` |
| **Route (`agent.router`)** | `selected_tool`, `router_confidence`, `intent_label` |
| **Tool (`tool.execution`)** | `tool_name`, `retrieval_query`, `returned_chunk_ids`, `k_count` |
| **Rerank (`tool.rerank`)** | `pre_rerank_scores`, `post_rerank_scores`, `rerank_model` |
| **LLM (`llm.inference`)** | `model_name`, `input_tokens`, `output_tokens`, `time_to_first_token_ms`, `temperature` |

### Infrastructure Metrics (Prometheus + Grafana)

* **Latency:** Track P50, P90, and P99 response times. Standard: P90 under 3.5 seconds for document queries.
* **Financial Auditing:** Real-time token consumption meters computing exact spend per user, query type, and internal department.
* **Error Rate:** Track tool execution timeouts, database disconnects, and unparseable LLM output rates.

### Automated Evaluation (Ragas Pipeline)

A headless worker samples 10% of production interactions daily to compute the RAG Triad:

| Metric | Measurement Goal | Target SLA |
| --- | --- | --- |
| **Context Relevance** | Precision of retrieved context relative to the input query | $\ge 0.85$ |
| **Faithfulness** | Percentage of claims in generated output grounded directly in retrieved context | $\ge 0.95$ |
| **Answer Relevance** | Completeness and direct relevance of the final response to the user's prompt | $\ge 0.90$ |

---

## 6. Technical Stack & Deployment Specification

| Tier | Technology Selection | Configuration Notes |
| --- | --- | --- |
| **Agent Orchestration** | LangGraph (Python 3.11+) | Stateful multi-node workflow hosted on FastAPI |
| **Document Parsing** | IBM Docling (`docling-core`) | GPU-accelerated worker pool using TableFormer |
| **Vector Database** | Qdrant | Distributed cluster, HNSW indexing enabled |
| **Lexical Engine** | OpenSearch 2.x | BM25 with custom domain synonym dictionaries |
| **LLM Engine** | Dedicated Enterprise Endpoints | Primary: High-parameter model; Router: Small parameter model |
| **Tracing Backend** | Langfuse (Self-Hosted / Cloud) | Connected via standard OpenTelemetry HTTP collector |
| **Container Platform** | Kubernetes (EKS / GKE) | Horizontal Pod Autoscalers (HPA) triggered on inference queue depth |

---

