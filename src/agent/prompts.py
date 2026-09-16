"""Centralized prompt templates for the LangGraph Document RAG agent.

Provides structured prompt templates for:
- grade_node: Relevance grading and confidence scoring in strict JSON
- rewrite_node: Search query optimization and keyword sharpening
- generate_node: Grounded, citation-enforced answer synthesis
- Fallback refusal: Standard deterministic explanation on repeated retrieval failure
"""

from typing import List
from schemas import DocumentChunk

# --- Relevance Grader Prompts ---
GRADE_SYSTEM_PROMPT = """You are an expert document relevance evaluator in an enterprise RAG system.
Your task is to determine whether the provided retrieved document excerpts contain sufficient, accurate information to answer the user's question.

Evaluate the content objectively:
1. Score relevance between 0.00 and 1.00.
2. If the excerpts directly answer the question or contain key relevant facts, score >= 0.70.
3. If the excerpts are irrelevant, tangential, or missing the core information, score < 0.70.

You must respond ONLY with a valid JSON object matching this schema:
{
    "is_relevant": true,
    "confidence": 0.85,
    "reason": "Brief one-sentence explanation of the grading rationale"
}"""

GRADE_USER_TEMPLATE = """Question: {question}

Retrieved Excerpts:
{context}

JSON Evaluation:"""

# --- Query Rewriter Prompts ---
REWRITE_SYSTEM_PROMPT = """You are a search query reformulation expert for vector and semantic search engines.
Your task is to rephrase the user's question into an optimized, keyword-dense search query.

Rules:
1. Strip conversational filler (e.g., "can you tell me", "what is", "please explain").
2. Focus on technical terminology, domain entities, and core semantic keywords.
3. Return ONLY the reformulated query string. Do not add quotes, markdown, or commentary."""

REWRITE_USER_TEMPLATE = """Original Question: {question}
Attempt: {attempt_number}

Optimized Search Query:"""

# --- Grounded Answer Generation Prompts ---
GENERATE_SYSTEM_PROMPT = """You are an authoritative enterprise knowledge assistant. Answer the user's question using ONLY the provided document context.

Strict Grounding Rules:
1. Every factual statement must be backed by the provided excerpts.
2. For every factual claim, include an inline citation in the format: [Source: <filename>, Page <page_number>].
3. Do NOT extrapolate, speculate, or introduce external knowledge.
4. If a detail is missing from the context, explicitly state that it is not covered.
5. Use clear, professional formatting (concise paragraphs, bullet points when appropriate)."""

GENERATE_USER_TEMPLATE = """Question: {question}

Context Excerpts:
{context}

Answer:"""

# --- Fallback Refusal Template ---
FALLBACK_REFUSAL_TEMPLATE = (
    "I do not have sufficient information in the indexed documents to answer "
    "your question with confidence (relevance score: {confidence_score:.2f} < {threshold:.2f}). "
    "Please refine your query or ensure the relevant documents are ingested."
)


def format_chunk_context(chunks: List[DocumentChunk]) -> str:
    """Format a list of DocumentChunks into a clean string for prompt injection."""
    if not chunks:
        return "[No document excerpts available]"

    formatted_parts = []
    for idx, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        source_file = meta.get("source_file", "unknown_document")
        page_nums = meta.get("page_numbers", [])
        page_str = ", ".join(str(p) for p in page_nums) if page_nums else "1"
        section = meta.get("section_path", "General")
        content_type = meta.get("content_type", "text")

        header = f"--- [Excerpt {idx}] Source: {source_file} | Page(s): {page_str} | Section: {section} | Type: {content_type} ---"
        formatted_parts.append(f"{header}\n{chunk.text.strip()}")

    return "\n\n".join(formatted_parts)
