from typing import Protocol, AsyncGenerator, Any, Optional, Tuple

class LLMProvider(Protocol):
    """
    Contrato mínimo para interactuar con un proveedor de LLM.
    Reemplaza el uso directo de OllamaClient en el orquestador y el agente.
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = False,
        **params: Any,
    ) -> dict[str, Any] | AsyncGenerator[tuple[str, bool], None]:
        """
        Envía un mensaje al modelo.
        
        Contratos de retorno reales basados en OllamaClient:
        - Si stream=False: devuelve dict con estructura {'message': {'content': str}, 'prompt_eval_count': int, ...}
        - Si stream=True: devuelve AsyncGenerator que yield (chunk_text: str, is_final: bool)
        
        Lanza excepciones httpx.HTTPStatusError o similares en fallos de red/autenticación.
        """
        ...

    def extract_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extrae tool_calls de la respuesta NO STREAMING.
        
        Entrada: El diccionario retornado por chat() con stream=False.
        Salida: Lista de tool_calls (vacía si no hay).
        Formato interno del item: {'function': {'name': str, 'arguments': str}} u objeto completo.
        """
        ...

    def get_token_counts(self, response: dict[str, Any]) -> tuple[int, int]:
        """
        Obtiene (prompt_tokens, completion_tokens) de la respuesta NO STREAMING.
        
        Entrada: El diccionario retornado por chat() con stream=False.
        Salida: Tuple[int, int] o (0, 0) si no disponibles.
        """
        ...

    async def close(self) -> None:
        """
        Cierra recursos del proveedor (ej. connection pools).
        Llamado desde LocalAgent.__del__ o cleanup.
        """
        ...