from rank_bm25 import BM25Okapi
from app.database.qdrant_client import vector_store


class BM25Service:
    def __init__(self):
        self.payloads: list[dict] = []
        self.bm25: BM25Okapi | None = None

    def build(self, chunks: list[dict]):
        """chunks: [{payload: {...}}] where payload has a 'text' key."""
        payloads = [c.get("payload", c) for c in chunks]
        self.payloads = payloads
        tokenized = [p["text"].lower().split() for p in payloads if p.get("text")]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def rebuild_from_qdrant(self):
        """Recover BM25 state after a process restart (Qdrant is the source of truth)."""
        payloads = vector_store.scroll_all()
        self.build([{"payload": p} for p in payloads])

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [{**self.payloads[i], "score": float(scores[i])} for i in ranked if scores[i] > 0]


bm25_service = BM25Service()
