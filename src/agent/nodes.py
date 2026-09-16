"""LangGraph Node implementations for the self-correcting Document RAG pipeline.

Nodes:
- retrieve_node: Fetches top-k document chunks from Qdrant vector store.
- grade_node: Evaluates chunk relevance and computes a confidence score (0.0 - 1.0).
- rewrite_node: Re-engineers failing queries into high-density search terms and tracks retries.
- generate_node: Synthesizes grounded answers with citations or triggers graceful refusal.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from schemas import DocumentChunk
from config import get_settings
from storage.vector_store import QdrantVectorStore
from ingestion.embeddings import HuggingFaceEmbedder, GeminiEmbedder, get_embedder
from agent.state import RAGState
from agent.prompts import (
    GRADE_SYSTEM_PROMPT,
    GRADE_USER_TEMPLATE,
    REWRITE_SYSTEM_PROMPT,
    REWRITE_USER_TEMPLATE,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_USER_TEMPLATE,
    FALLBACK_REFUSAL_TEMPLATE,
    format_chunk_context,
)
from agent.llm import LLMClient

logger = logging.getLogger(__name__)


def retrieve_node(
    state: RAGState,
    vector_store: Optional[QdrantVectorStore] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retrieve top-k relevant document chunks from the vector store using dense vector similarity."""
    settings = get_settings()
    active_query = state.get("rewritten_question") or state.get("question", "")

    if not active_query:
        logger.warning("[retrieve_node] No active query found in state.")
        return {"retrieved_chunks": []}

    vs = vector_store or QdrantVectorStore()
    emb = embedder or get_embedder()

    try:
        query_vector = emb.embed_text(active_query, is_query=True)
        chunks = vs.search(query_vector=query_vector, top_k=settings.RETRIEVAL_TOP_K)
        return {"retrieved_chunks": chunks}
    except Exception as e:
        logger.error(f"[retrieve_node] Retrieval failed: {e}")
        return {"retrieved_chunks": []}


def grade_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Grade relevance of retrieved chunks against the active question and assign confidence score."""
    question = state.get("rewritten_question") or state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    client = llm or LLMClient()

    if not chunks:
        logger.info("[grade_node] Zero chunks retrieved; assigning baseline zero confidence.")
        return {"confidence_score": 0.0}

    context_str = format_chunk_context(chunks)
    prompt = GRADE_USER_TEMPLATE.format(question=question, context=context_str)

    raw_response = client.generate(
        prompt=prompt,
        system_prompt=GRADE_SYSTEM_PROMPT,
        json_mode=True,
    )

    # Robust JSON parsing with fallback regex extraction
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    confidence = float(data.get("confidence", 0.0))
    # Constrain confidence between 0.0 and 1.0
    confidence = max(0.0, min(1.0, confidence))

    logger.info(f"[grade_node] Graded confidence: {confidence:.2f} (Reason: {data.get('reason', 'N/A')})")
    return {"confidence_score": confidence}


def rewrite_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Reformulate the search query into high-density semantic search keywords and increment retry count."""
    orig_question = state.get("question", "")
    current_retries = state.get("retry_count", 0) + 1
    client = llm or LLMClient()

    prompt = REWRITE_USER_TEMPLATE.format(
        question=orig_question,
        attempt_number=current_retries,
    )

    rewritten_raw = client.generate(
        prompt=prompt,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        json_mode=False,
    )

    # Sanitize reformulated query
    rewritten_clean = rewritten_raw.strip('"` \n\r\t')
    logger.info(f"[rewrite_node] Retry #{current_retries}: '{orig_question}' -> '{rewritten_clean}'")

    return {
        "rewritten_question": rewritten_clean,
        "retry_count": current_retries,
    }


def generate_node(
    state: RAGState,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Generate a grounded response with page/source citations or output a standard refusal."""
    settings = get_settings()
    confidence = state.get("confidence_score", 0.0)
    retries = state.get("retry_count", 0)
    threshold = settings.CONFIDENCE_THRESHOLD

    # If confidence is below threshold and we have exhausted retries, refuse gracefully
    if confidence < threshold and retries >= 2:
        logger.info(f"[generate_node] Retrieval failed after {retries} retries ({confidence:.2f} < {threshold:.2f}). Triggering refusal.")
        refusal_msg = FALLBACK_REFUSAL_TEMPLATE.format(
            confidence_score=confidence,
            threshold=threshold,
        )
        return {"answer": refusal_msg}

    # Otherwise synthesize grounded answer
    question = state.get("question", "")
    chunks = state.get("retrieved_chunks", [])
    context_str = format_chunk_context(chunks)
    client = llm or LLMClient()

    prompt = GENERATE_USER_TEMPLATE.format(question=question, context=context_str)
    answer = client.generate(
        prompt=prompt,
        system_prompt=GENERATE_SYSTEM_PROMPT,
        json_mode=False,
    )

    return {"answer": answer}
