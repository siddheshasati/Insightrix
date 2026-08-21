from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question", examples=["Who founded the company and where is it based?"])
    top_k: Optional[int] = Field(None, description="Override number of fused chunks used as context")
    graph_depth: Optional[int] = Field(None, description="Override graph traversal depth (hops)")


class IngestRequest(BaseModel):
    pdf_dir: Optional[str] = Field(None, description="Folder to scan for pdf/docx/txt/md files. Defaults to PDF_DIR in .env")


class CypherRequest(BaseModel):
    query: str = Field(..., description="Read-only Cypher query", examples=["MATCH (e:Entity) RETURN e.name, e.type LIMIT 25"])
    parameters: Optional[Dict[str, Any]] = {}
