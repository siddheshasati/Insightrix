import json
import re
from app.core.prompts import EXTRACTION_SYSTEM
from app.services.llm_service import llm_service


def _safe_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        return {"entities": [], "relations": []}


def extract_graph(text: str) -> dict:
    """Calls Gemma to extract {entities, relations} JSON from arbitrary text (chunk or query).
    Entity/relation types are fully inferred by the LLM - zero manual/regex patterns."""
    prompt = f'Text:\n"""\n{text[:4000]}\n"""\n\nExtract the knowledge graph JSON now.'
    raw = llm_service.generate(prompt, system=EXTRACTION_SYSTEM, json_mode=True, temperature=0.0)
    data = _safe_json(raw)
    data.setdefault("entities", [])
    data.setdefault("relations", [])
    data["entities"] = [e for e in data["entities"] if isinstance(e, dict) and e.get("name")]
    data["relations"] = [r for r in data["relations"] if isinstance(r, dict) and r.get("source") and r.get("target")]
    return data
