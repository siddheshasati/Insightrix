from fastapi import FastAPI
from app.api.routes import ingest, query, graph, admin
from app.services.bm25_service import bm25_service

app = FastAPI(
    title="KGRA -- Knowledge Graph & Retrieval Augmented",
    description=(
        "Hybrid Retrieval-Augmented Generation combining semantic search + BM25 (Qdrant) "
        "with a Graph RAG layer (Neo4j) whose knowledge graph is built entirely by an LLM "
        "(Gemma 4B) doing entity/relation extraction - no manual/regex patterns. "
        "Embeddings: MXBAI. Both models are served locally and reached via plain API URLs, no API keys."
    ),
    version="1.0.0",
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)
app.include_router(admin.router)


@app.on_event("startup")
def _startup():
    # Recover BM25 keyword index from Qdrant (source of truth) after a restart.
    try:
        bm25_service.rebuild_from_qdrant()
    except Exception as e:
        print(f"[startup] could not rebuild bm25 index yet: {e}")
