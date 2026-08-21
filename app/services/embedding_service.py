import httpx
from app.core.config import settings


class EmbeddingService:
    """Calls your local MXBAI server at MXBAI_API_URL. No API key needed.

    Tries a couple of common local-server request/response shapes so it works
    out of the box with most self-hosted embedding servers (text-embeddings-inference,
    a custom FastAPI wrapper, OpenAI-compatible /v1/embeddings, etc).
    If your server uses a different schema, adjust `_request`/`_parse` below.
    """

    def __init__(self):
        self.url = settings.MXBAI_API_URL

    def _request(self, text: str) -> dict:
        with httpx.Client(timeout=60) as client:
            # OpenAI-compatible style body; most local embedding servers accept "input" or "text"
            r = client.post(self.url, json={"input": text, "text": text})
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _parse(data: dict) -> list[float]:
        if "embedding" in data:
            return data["embedding"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"][0]["embedding"]
        if "embeddings" in data:
            return data["embeddings"][0] if isinstance(data["embeddings"][0], list) else data["embeddings"]
        if "vector" in data:
            return data["vector"]
        raise ValueError(f"Unrecognized embedding response shape: {list(data.keys())}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._parse(self._request(t)) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


embedding_service = EmbeddingService()
