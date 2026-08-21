import re
from neo4j import GraphDatabase
from app.core.config import settings


def _sanitize_label(label: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_]", "_", (label or "Entity").strip()) or "Entity"
    return label[:50]


def _sanitize_rel(rel: str) -> str:
    rel = re.sub(r"[^a-zA-Z0-9_]", "_", (rel or "RELATED_TO").strip().upper()) or "RELATED_TO"
    return rel[:50]


class GraphStore:
    def __init__(self):
        self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        self._ensure_constraints()

    def close(self):
        self.driver.close()

    def _ensure_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            s.run("CREATE FULLTEXT INDEX entity_fts IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]")

    def upsert_chunk_meta(self, chunk_id: str, doc_id: str, text: str, source: str):
        with self.driver.session() as s:
            s.run(
                "MERGE (c:Chunk {id: $chunk_id}) SET c.doc_id=$doc_id, c.source=$source, c.text=$text",
                chunk_id=chunk_id, doc_id=doc_id, source=source, text=text[:500],
            )

    def upsert_entity(self, name: str, etype: str, description: str, doc_id: str, chunk_id: str):
        label = _sanitize_label(etype)
        with self.driver.session() as s:
            s.run(
                f"""
                MERGE (e:Entity {{name: $name}})
                ON CREATE SET e.type = $etype, e.description = $description, e.doc_ids = [$doc_id]
                ON MATCH SET e.description = coalesce(e.description, $description),
                              e.doc_ids = CASE WHEN NOT $doc_id IN e.doc_ids THEN e.doc_ids + $doc_id ELSE e.doc_ids END
                SET e:`{label}`
                WITH e
                MATCH (c:Chunk {{id: $chunk_id}})
                MERGE (e)-[:MENTIONED_IN]->(c)
                """,
                name=name, etype=etype, description=description, doc_id=doc_id, chunk_id=chunk_id,
            )

    def upsert_relation(self, source: str, target: str, rtype: str, description: str, chunk_id: str):
        rel = _sanitize_rel(rtype)
        with self.driver.session() as s:
            s.run(
                f"""
                MERGE (a:Entity {{name: $source}})
                MERGE (b:Entity {{name: $target}})
                MERGE (a)-[r:`{rel}`]->(b)
                SET r.description = coalesce(r.description, $description), r.chunk_id = $chunk_id
                """,
                source=source, target=target, description=description, chunk_id=chunk_id,
            )

    def find_entities_fulltext(self, query_text: str, limit: int = 5) -> list[dict]:
        with self.driver.session() as s:
            res = s.run(
                "CALL db.index.fulltext.queryNodes('entity_fts', $q) YIELD node, score "
                "RETURN node.name AS name, node.type AS type, score ORDER BY score DESC LIMIT $limit",
                q=query_text, limit=limit,
            )
            return [dict(r) for r in res]

    def get_subgraph(self, entity_names: list[str], depth: int = 2, limit: int = 80) -> dict:
        if not entity_names:
            return {"nodes": [], "edges": []}
        with self.driver.session() as s:
            res = s.run(
                f"""
                MATCH path = (e:Entity)-[*0..{max(0, min(depth, 4))}]-(:Entity)
                WHERE e.name IN $names
                UNWIND relationships(path) AS rel
                WITH DISTINCT startNode(rel) AS a, endNode(rel) AS b, type(rel) AS rtype, rel.description AS rdesc
                RETURN a.name AS source, a.type AS source_type, b.name AS target, b.type AS target_type,
                       rtype, rdesc
                LIMIT $limit
                """,
                names=entity_names, limit=limit,
            )
            edges = [dict(r) for r in res]
        nodes = {}
        for e in edges:
            nodes[e["source"]] = e["source_type"]
            nodes[e["target"]] = e["target_type"]
        return {"nodes": [{"name": n, "type": t} for n, t in nodes.items()], "edges": edges}

    def reset(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")


graph_store = GraphStore()
