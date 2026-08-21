import httpx
from app.core.config import settings


class LLMService:
    """Calls your local Gemma 4B server at GEMMA_API_URL. No API key needed.

    Defaults to an OpenAI-compatible chat/completions body, which is what most
    local model servers (llama.cpp server, LM Studio, vLLM, text-generation-webui)
    expose. If your server uses a different schema, adjust `_request`/`_parse` below.
    """

    def __init__(self):
        self.url = settings.GEMMA_API_URL

    def _request(self, prompt: str, system: str, temperature: float, json_mode: bool) -> dict:
        body = {
            "model": "gemma-4b",
            "messages": [
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        with httpx.Client(timeout=180) as client:
            r = client.post(self.url, json=body)
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _parse(data: dict) -> str:
        if "choices" in data:
            choice = data["choices"][0]
            return choice.get("message", {}).get("content") or choice.get("text", "")
        if "response" in data:
            return data["response"]
        if "content" in data:
            return data["content"]
        raise ValueError(f"Unrecognized LLM response shape: {list(data.keys())}")

    def generate(self, prompt: str, system: str = None, json_mode: bool = False, temperature: float = 0.1) -> str:
        data = self._request(prompt, system, temperature, json_mode)
        return self._parse(data)


llm_service = LLMService()
