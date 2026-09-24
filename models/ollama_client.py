"""Cliente REST asíncrono para Ollama local. Se conecta a http://localhost:11434 por defecto.
El endpoint es configurable vía config o parámetro init. Correspondiente a models/ en la estructura del proyecto."""

from __future__ import annotations

import json
from typing import Any, Optional
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

import httpx

# Ruta al archivo de config de modelos (para descubrir el endpoint)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
CONFIG_PATHS = [
    Path(__file__).parent.parent / "config" / "ollama_endpoint.yaml",  # prioriza config dedicada
    Path(__file__).parent.parent / "config" / "agent.yaml",           # fallback
]


class OllamaClient:
    """Cliente asíncrono para la API REST de Ollama.
    
    El endpoint (base_url) se resolve en este orden:
      1) Parámetro explícito `base_url` en el constructor
      2) Valor de config/ollama_endpoint.yaml si existe → clave "endpoint"
      3) Default hardcodeado: http://127.0.0.1:11434
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._resolved_url: str = OllamaClient.resolve_endpoint(base_url)
        self.timeout = timeout
        self.model_name = model_name  # Almacena el nombre del modelo
        self.max_tokens = max_tokens   # Almacena los tokens máximos
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------ public helpers
    @staticmethod
    def resolve_endpoint(url_override: str | None) -> str:
        """Devuelve el endpoint base de Ollama tras seguir la política de priorización."""
        if url_override:
            return url_override.rstrip("/")
        # buscar config dedicada
        for p in CONFIG_PATHS:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                ep = (data.get("ollama_endpoint") or data.get("endpoint"))
                if ep:
                    return ep.rstrip("/")
            except FileNotFoundError:
                continue
        # fallback
        return DEFAULT_OLLAMA_URL

    # ------------------------------------------------------------------ lifecycle
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._resolved_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client
    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------ API wrappers
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,  # CORREGIDO §Tool calling: tools como parámetro explícito
        **params: Any,
    ) -> dict[str, Any]:
        """Envía un chat con la API /api/chat de Ollama.
        
        CORREGIDO §Tool calling: Accepta y envía 'tools' en el payload según spec oficial.
        """
        client = await self._get_client()
        payload: dict[str, Any] = {
            "model": model or self.model_name or "",
            "messages": messages,
        }
        
        # CORREGIDO §Tool calling: Incluir tools en el payload si se proporcionan
        if tools:
            payload["tools"] = tools
        
        payload.update(params)

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> dict[str, Any]:
        """Lista los modelos instalados en Ollama (/api/tags)."""
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        return response.json()

    # CORREGIDO §Tool calling: Extraer tool_calls de la respuesta oficial de Ollama
    @staticmethod
    def extract_tool_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extrae tool_calls de la respuesta de Ollama.
        
        CORREGIDO §Tool calling: Maneja tanto respuestas con como sin tool_calls.
        La API oficial retorna: {"message": {"content": "...", "tool_calls": [...]}}
        
        Returns:
            Lista de tool_calls (vacía si no hay)
        """
        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])
        
        # Si no hay tool_calls, devolver lista vacía (no romper)
        if not tool_calls:
            return []
        
        return tool_calls
        # CORREGIDO §11: Exponer token counts reales del response de Ollama
    @staticmethod
    def get_token_counts(response: dict[str, Any]) -> tuple[int, int]:
        """Obtiene prompt_eval_count y eval_count reales de la respuesta.
        
        CORREGIDO §11: Usa conteos reales en vez de estimaciones.
        
        Returns:
            Tuple (prompt_tokens, completion_tokens) - (0, 0) si no disponibles
        """
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        
        # Fallback por si viene como string
        try:
            prompt_tokens = int(prompt_tokens) if prompt_tokens else 0
            completion_tokens = int(completion_tokens) if completion_tokens else 0
        except (ValueError, TypeError):
            prompt_tokens = 0
            completion_tokens = 0
        
        return (prompt_tokens, completion_tokens)

# ------------------------------------------------------------------ instancia global por conveniencia
DEFAULT_CLIENT: Optional[OllamaClient] = None


def get_default_client() -> OllamaClient:
    """Devuelve una instancia única y perezosa del cliente con el endpoint resuelto."""
    global DEFAULT_CLIENT
    if DEFAULT_CLIENT is None:
        DEFAULT_CLIENT = OllamaClient()
    return DEFAULT_CLIENT
