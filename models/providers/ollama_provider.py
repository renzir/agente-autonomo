from models.ollama_client import OllamaClient
from .base import LLMProvider

class OllamaProvider(LLMProvider):
    def __init__(self, client: OllamaClient = None):
        # Reutiliza la instancia existente de OllamaClient
        self._client = client or OllamaClient()

    async def chat(self, messages, model=None, tools=None, stream=False, **params):
        # Delegación directa al cliente existente
        return await self._client.chat(
            messages=messages,
            model=model,
            tools=tools,
            stream=stream,
            **params
        )

    def extract_tool_calls(self, response: dict) -> list[dict]:
        # Usa el método estático ya existente
        return OllamaClient.extract_tool_calls(response)

    def get_token_counts(self, response: dict) -> tuple[int, int]:
        # Usa el método estático ya existente
        return OllamaClient.get_token_counts(response)

    async def close(self):
        await self._client.close()