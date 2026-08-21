from fastapi import APIRouter
from app.database.qdrant_client import vector_store
from app.database.neo4j_client import graph_store
from app.services.bm25_service import bm25_service
from app.core.config import settings

router = APIRouter(tags=["Admin"])


@router.delete("/reset", summary="Wipe the Qdrant collection, Neo4j graph, and BM25 index")
def reset():
    vector_store.reset()
    graph_store.reset()
    bm25_service.build([])
    return {"status": "reset complete"}


@router.get("/health", summary="Health/config check")
def health():
    return {
        "status": "ok",
        "mxbai_api_url": settings.MXBAI_API_URL,
        "gemma_api_url": settings.GEMMA_API_URL,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "neo4j_uri": settings.NEO4J_URI,
    }
