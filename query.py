from fastapi import APIRouter, HTTPException
from app.core.schemas import QueryRequest
from app.services.retrieval_service import hybrid_query

router = APIRouter(prefix="/query", tags=["Retrieval"])


@router.post("", summary="Ask a question (hybrid RAG + GraphRAG)")
def query(req: QueryRequest):
    """Fuses semantic (Qdrant/MXBAI) + BM25 chunk retrieval via RRF, extracts entities from
    the question to pull a relevant Neo4j subgraph, and generates a grounded answer with Gemma."""
    if not req.question.strip():
        raise HTTPException(400, "question cannot be empty")
    return hybrid_query(req.question, req.top_k, req.graph_depth)
