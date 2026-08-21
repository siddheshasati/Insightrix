### KGRA -- Knowledge Graph & Retrieval Augmented

#### System Architecture
```
data/pdfs (.pdf/.docx/.txt/.md)
        │
        ▼
Document Parser (pypdf / python-docx)
        │
        ▼
Chunking (Recursive, size/overlap configurable)
        │
        ├──────────────► Entity/Relation Extraction (Gemma 4B, local API)
        │                          │
        │                          ▼
        │                     Neo4j Graph
        │
        ▼
MXBAI Embeddings (local API)
        │
        ▼
Qdrant Vector Store
        │
        ▼
Hybrid Retrieval (BM25 + Vector + Graph, RRF fusion)
        │
        ▼
Gemma 4B generates grounded answer
        │
        ▼
processed files moved to data/processed
```

## Project layout
```
KGRA/
├── app/
│   ├── api/routes/     ingest.py, query.py, graph.py, admin.py
│   ├── core/           config.py, prompts.py, schemas.py
│   ├── database/       qdrant_client.py, neo4j_client.py
│   └── services/       embedding_service.py, llm_service.py, extraction_service.py,
│                       bm25_service.py, ingestion_service.py, retrieval_service.py
├── data/
│   ├── pdfs/           drop source files here
│   └── processed/      moved here after ingestion
├── ui/
│   └── streamlit_app.py
├── main.py
├── .env / .env.example
├── docker-compose.yml  (Neo4j + Qdrant only)
└── requirements.txt
```

## 1. Models
MXBAI and Gemma 4B run on your own local servers, reached via plain API URLs -
**no API key**. Set the endpoints in `.env`:
```
MXBAI_API_URL=http://localhost:8000/embed
GEMMA_API_URL=http://localhost:8001/v1/chat/completions
```
`app/services/embedding_service.py` and `app/services/llm_service.py` each try a
couple of common local-server response shapes (OpenAI-style, `{"embedding": [...]}`,
etc). If your servers use a different schema, tweak `_request`/`_parse` in those
two files — that's the only place request/response format is defined.

## 2. Start infra
```bash
cp .env.example .env      # edit URLs/creds if needed
docker compose up -d      # Neo4j + Qdrant
```

## 3. Install & run
```bash
pip install -r requirements.txt
# drop pdf/docx/txt/md files into data/pdfs/
uvicorn main:app --reload
```
Open **http://localhost:8000/docs** — every feature is testable there.

Optional UI:
```bash
streamlit run ui/streamlit_app.py
```

## Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Parse data/pdfs -> embed -> Qdrant, LLM-extract graph -> Neo4j (background) |
| `GET /ingest/status` | Poll progress/result |
| `POST /query` | Hybrid RAG + GraphRAG question answering |
| `GET /graph/entity/{name}` | Entity neighborhood subgraph |
| `GET /graph/search` | Fulltext search graph entities |
| `POST /graph/cypher` | Read-only Cypher (dev tool) |
| `DELETE /reset` | Wipe Qdrant + Neo4j + BM25 |
| `GET /health` | Config/health check |

Neo4j browser: http://localhost:7474 · Qdrant dashboard: http://localhost:6333/dashboard
