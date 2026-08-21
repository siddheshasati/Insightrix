from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Local model servers: plain API URLs, NO api key required ---
    MXBAI_API_URL: str = "http://localhost:8000/embed"       # local MXBAI embedding server
    GEMMA_API_URL: str = "http://localhost:8001/v1/chat/completions"  # local Gemma 4B server
    EMBED_DIM: int = 1024

    # --- Qdrant (vector DB, local docker) ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "kgra_chunks"

    # --- Neo4j (graph DB, local docker) ---
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "changeme123"

    # --- Paths ---
    PDF_DIR: str = "./data/pdfs"
    PROCESSED_DIR: str = "./data/processed"

    # --- Chunking / retrieval tuning ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    TOP_K_VECTOR: int = 5
    TOP_K_BM25: int = 5
    TOP_K_FINAL: int = 5
    GRAPH_DEPTH: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
