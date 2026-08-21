from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from app.core.config import settings


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(settings.QDRANT_COLLECTION):
            self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=qm.VectorParams(size=settings.EMBED_DIM, distance=qm.Distance.COSINE),
            )

    def upsert(self, points: list[dict]):
        """points: [{id, vector, payload}]"""
        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points],
        )

    def search(self, vector: list[float], top_k: int) -> list[dict]:
        res = self.client.search(collection_name=settings.QDRANT_COLLECTION, query_vector=vector, limit=top_k)
        return [{**r.payload, "score": r.score} for r in res]

    def scroll_all(self) -> list[dict]:
        """Pull every stored chunk back out - used to rebuild BM25 on startup (Qdrant = source of truth)."""
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION, limit=256, offset=offset, with_payload=True, with_vectors=False
            )
            out.extend([p.payload for p in points])
            if offset is None:
                break
        return out

    def reset(self):
        self.client.delete_collection(settings.QDRANT_COLLECTION)
        self._ensure_collection()


vector_store = VectorStore()
