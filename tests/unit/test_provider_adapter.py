from unittest.mock import AsyncMock, MagicMock

import pytest

from models.ollama_client import OllamaClient
from models.providers.ollama_provider import OllamaProvider


class TestOllamaProviderAdapter:

    @pytest.mark.asyncio
    async def test_delegates_chat_to_client(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.chat = AsyncMock(return_value={"message": {"content": "Hola"}})

        provider = OllamaProvider(client=mock_client)
        messages = [{"role": "user", "content": "Hola"}]

        result = await provider.chat(messages, stream=False)

        mock_client.chat.assert_awaited_once_with(
            messages=messages,
            model=None,
            tools=None,
            stream=False
        )        
        assert result == {"message": {"content": "Hola"}}

    def test_delegates_extract_tool_calls(self):
        mock_client = MagicMock(spec=OllamaClient)
        provider = OllamaProvider(client=mock_client)

        response = {"message": {"content": "Hola"}}
        expected = OllamaClient.extract_tool_calls(response)

        assert provider.extract_tool_calls(response) == expected

    @pytest.mark.asyncio
    async def test_close_delegates(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.close = AsyncMock()

        provider = OllamaProvider(client=mock_client)

        await provider.close()

        mock_client.close.assert_awaited_once()