from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.core.schemas import IngestRequest
from app.core.config import settings
from app.services.ingestion_service import ingest_all

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

_status = {"running": False, "last_result": None}


def _run(pdf_dir: str | None):
    _status["running"] = True
    try:
        _status["last_result"] = ingest_all(pdf_dir)
    except Exception as e:
        _status["last_result"] = {"error": str(e)}
    finally:
        _status["running"] = False


@router.post("", summary="Ingest all PDF/DOCX/TXT/MD files from data/pdfs")
def ingest(background_tasks: BackgroundTasks, req: IngestRequest = IngestRequest()):
    """Runs asynchronously: parses files -> chunks -> MXBAI-embeds into Qdrant ->
    Gemma extracts entities/relations into Neo4j -> rebuilds BM25 -> moves files to data/processed.
    Poll /ingest/status for progress."""
    if _status["running"]:
        raise HTTPException(409, "Ingestion already running")
    background_tasks.add_task(_run, req.pdf_dir)
    return {"status": "started", "pdf_dir": req.pdf_dir or settings.PDF_DIR}


@router.get("/status", summary="Check ingestion progress/result")
def ingest_status():
    return _status
