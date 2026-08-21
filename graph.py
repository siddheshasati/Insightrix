from fastapi import APIRouter, HTTPException
from app.core.schemas import CypherRequest
from app.database.neo4j_client import graph_store

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/entity/{name}", summary="Get the neighborhood subgraph of a specific entity")
def graph_entity(name: str, depth: int = 2):
    subgraph = graph_store.get_subgraph([name], depth=depth)
    if not subgraph["nodes"]:
        raise HTTPException(404, f"Entity '{name}' not found in the graph")
    return subgraph


@router.get("/search", summary="Full-text search over graph entities")
def graph_search(q: str, limit: int = 5):
    return graph_store.find_entities_fulltext(q, limit)


@router.post("/cypher", summary="[DEV] Run a raw read-only Cypher query against Neo4j")
def graph_cypher(req: CypherRequest):
    forbidden = ["DELETE", "CREATE", "MERGE", " SET ", "REMOVE", "DROP"]
    if any(k in f" {req.query.upper()} " for k in forbidden):
        raise HTTPException(400, "Only read-only Cypher is allowed via this endpoint")
    with graph_store.driver.session() as s:
        return [dict(r) for r in s.run(req.query, **req.parameters)]
