import os
import shutil
import uuid
import glob
from pypdf import PdfReader
import docx
from app.core.config import settings
from app.services.embedding_service import embedding_service
from app.database.qdrant_client import vector_store
from app.database.neo4j_client import graph_store
from app.services.bm25_service import bm25_service
from app.services.extraction_service import extract_graph


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(path: str) -> str:
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def _read_txt(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return f.read()


LOADERS = {".pdf": _read_pdf, ".docx": _read_docx, ".txt": _read_txt, ".md": _read_txt}


def _chunk(text: str, size: int = None, overlap: int = None) -> list[str]:
    size = size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def load_documents(pdf_dir: str = None) -> list[dict]:
    pdf_dir = pdf_dir or settings.PDF_DIR
    docs = []
    for path in glob.glob(os.path.join(pdf_dir, "**", "*"), recursive=True):
        ext = os.path.splitext(path)[1].lower()
        if ext in LOADERS and os.path.isfile(path):
            try:
                text = LOADERS[ext](path)
                if text.strip():
                    docs.append({"id": str(uuid.uuid4()), "source": path, "text": text})
            except Exception as e:
                print(f"[ingest] skipping {path}: {e}")
    return docs


def ingest_all(pdf_dir: str = None) -> dict:
    """Loads every pdf/docx/txt/md in PDF_DIR, chunks it, embeds + stores in Qdrant,
    extracts a knowledge graph per chunk via the LLM (Gemma) into Neo4j, rebuilds BM25,
    then moves processed source files into PROCESSED_DIR. No manual entity/relation
    patterns are used anywhere - extraction is 100% LLM-driven."""
    pdf_dir = pdf_dir or settings.PDF_DIR
    docs = load_documents(pdf_dir)
    bm25_chunks = []
    stats = {"documents": len(docs), "chunks": 0, "entities": 0, "relations": 0, "errors": []}

    for doc in docs:
        for idx, chunk_text in enumerate(_chunk(doc["text"])):
            chunk_id = str(uuid.uuid4())
            try:
                vector = embedding_service.embed_one(chunk_text)
                payload = {
                    "text": chunk_text, "source": doc["source"], "doc_id": doc["id"],
                    "chunk_id": chunk_id, "chunk_index": idx,
                }
                vector_store.upsert([{"id": chunk_id, "vector": vector, "payload": payload}])
                bm25_chunks.append({"payload": payload})
                graph_store.upsert_chunk_meta(chunk_id, doc["id"], chunk_text, doc["source"])

                graph_data = extract_graph(chunk_text)
                for ent in graph_data["entities"]:
                    graph_store.upsert_entity(ent["name"], ent.get("type", "Entity"), ent.get("description", ""), doc["id"], chunk_id)
                    stats["entities"] += 1
                for rel in graph_data["relations"]:
                    graph_store.upsert_relation(rel["source"], rel["target"], rel.get("type", "RELATED_TO"), rel.get("description", ""), chunk_id)
                    stats["relations"] += 1
                stats["chunks"] += 1
            except Exception as e:
                stats["errors"].append(f"{doc['source']} chunk {idx}: {e}")

        # move fully-processed source file out of the inbox folder
        try:
            os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
            shutil.move(doc["source"], os.path.join(settings.PROCESSED_DIR, os.path.basename(doc["source"])))
        except Exception as e:
            stats["errors"].append(f"could not move {doc['source']}: {e}")

    bm25_service.build(bm25_chunks)
    return stats
