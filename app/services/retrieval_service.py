from app.services.embedding_service import embedding_service
from app.database.qdrant_client import vector_store
from app.services.bm25_service import bm25_service
from app.database.neo4j_client import graph_store
from app.services.extraction_service import extract_graph
from app.services.llm_service import llm_service
from app.core.prompts import ANSWER_SYSTEM
from app.core.config import settings


def _rrf_merge(vector_results: list[dict], bm25_results: list[dict], k: int = 60, top_k: int = 5) -> list[dict]:
    """Reciprocal Rank Fusion of semantic + BM25 result lists, deduped by chunk_id."""
    scores, payloads = {}, {}
    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        payloads[cid] = r
    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank + 1)
        payloads[cid] = r
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [payloads[cid] for cid, _ in ranked]


def _graph_context_text(subgraph: dict) -> str:
    if not subgraph["edges"]:
        return "No related graph facts found."
    lines = [f"{e['source']} --[{e['rtype']}]--> {e['target']}" + (f" ({e['rdesc']})" if e.get("rdesc") else "") for e in subgraph["edges"]]
    return "\n".join(lines)


def hybrid_query(question: str, top_k: int = None, graph_depth: int = None) -> dict:
    top_k = top_k or settings.TOP_K_FINAL
    graph_depth = graph_depth if graph_depth is not None else settings.GRAPH_DEPTH

    # 1. Hybrid chunk retrieval (semantic + BM25 fused via RRF)
    qvec = embedding_service.embed_one(question)
    vector_hits = vector_store.search(qvec, settings.TOP_K_VECTOR)
    bm25_hits = bm25_service.search(question, settings.TOP_K_BM25)
    merged_chunks = _rrf_merge(vector_hits, bm25_hits, top_k=top_k)

    # 2. Graph retrieval: extract entities from the query via the LLM,
    #    fall back to fulltext entity search if none were found.
    query_graph = extract_graph(question)
    entity_names = [e["name"] for e in query_graph["entities"]]
    if not entity_names:
        entity_names = [f["name"] for f in graph_store.find_entities_fulltext(question)]
    subgraph = graph_store.get_subgraph(entity_names, depth=graph_depth)

    # 3. Compose context and generate the final grounded answer
    chunk_context = "\n\n---\n\n".join(c["text"] for c in merged_chunks) or "No document chunks retrieved."
    graph_context = _graph_context_text(subgraph)
    prompt = f"""Question: {question}

Retrieved document chunks:
{chunk_context}

Knowledge graph facts:
{graph_context}

Answer the question using only the above context."""
    answer = llm_service.generate(prompt, system=ANSWER_SYSTEM, temperature=0.2)

    return {
        "answer": answer,
        "chunks": merged_chunks,
        "graph_entities_used": entity_names,
        "subgraph": subgraph,
    }
