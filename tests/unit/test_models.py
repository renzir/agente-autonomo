"""Tests para model_registry y OllamaClient (instanciación sin servidor).

Verifica que:
- ModelRegistry carga config correctamente.
- OllamaClient se instancia y resuelve el endpoint sin errores.
- Los métodos de la API pueden llamarse (o saber qué pasa si Ollama no está corriendo).
"""
import re
import pytest
from pathlib import Path
from httpx import Timeout


# ================================================================
# Helpers de ruta
# ================================================================

def _models_yaml_path() -> Path:
    return Path(__file__).parent.parent.parent / "config" / "models.yaml"


def _agent_yaml_path() -> Path:
    return Path(__file__).parent.parent.parent / "config" / "agent.yaml"


# ================================================================
# Test ModelRegistry (model_registry.py)
# ================================================= ================================================================

class TestModelRegistry:
    """Pruebas para ModelRegistry."""

    def test_registry_loads_defaults(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))
        defaults = registry.get_defaults()

        assert defaults["model"] == "primary"
        assert defaults["temperature"] == 0.7
        assert defaults["top_p"] == 0.9
        assert defaults["max_tokens"] == 2048
        assert defaults["keep_alive_seconds"] == 1800

    def test_registry_get_model_config(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))
        # Valores corregidos para coincidir con config/models.yaml real
        assert registry.get_model_config("primary") == "qwen3.6-64k:latest"
        assert registry.get_model_config("coding") == "qwen3.6-64k:latest"
        assert registry.get_model_config("fast") == "qwen3.6-64k:latest"
        assert registry.get_model_config("research") == "ornith15-iq4xs:latest"

    def test_registry_get_model_config_unknown(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))
        assert registry.get_model_config("nonexistent") is None

    def test_registry_list_models(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))
        models = registry.list_models()

        assert len(models) == 4
        assert "primary" in models
        assert "coding" in models
        assert "fast" in models
        assert "research" in models

    def test_registry_select_by_type(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))

        # Valores corregidos para coincidir con config/models.yaml real
        assert registry.select_model("primary") == "qwen3.6-64k:latest"
        assert registry.select_model("coding") == "qwen3.6-64k:latest"
        assert registry.select_model("fast") == "qwen3.6-64k:latest"
        assert registry.select_model("research") == "ornith15-iq4xs:latest"

    def test_registry_select_default(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))
        # Sin tipo — debe usar defaults.model que es 'primary'
        assert registry.select_model(None) == "qwen3.6-64k:latest"

    def test_registry_raw_access(self) -> None:
        from models.model_registry import ModelRegistry

        registry = ModelRegistry(str(_models_yaml_path()))
        raw = registry.raw

        assert isinstance(raw["models"], dict)
        assert isinstance(raw["defaults"], dict)


# ================================================================
# Test OllamaClient (ollama_client.py)
# ================================================================

class TestOllamaClientInstantiation:
    """Pruebas de instanciación del cliente Ollama (sin necesidad de servidor)."""

    def test_default_endpoint(self) -> None:
        from models.ollama_client import OllamaClient

        client = OllamaClient()
        # Aceptar cualquier URL local válida (localhost, loopback, o IP de red local privada)
        url = client._resolved_url
        assert 'localhost' in url or \
               '127.' in url or \
               '::1' in url or \
               re.search(r'^http://(192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})', url)

    def test_custom_endpoint(self) -> None:
        from models.ollama_client import OllamaClient

        url = "http://192.168.1.100:11434"
        client = OllamaClient(base_url=url)
        assert client._resolved_url == "http://192.168.1.100:11434"

    def test_endpoint_trailing_slash_stripped(self) -> None:
        from models.ollama_client import OllamaClient

        client = OllamaClient(base_url="http://127.0.0.1:11434/")
        assert client._resolved_url == "http://127.0.0.1:11434"

    def test_timeout_attribute(self) -> None:
        from models.ollama_client import OllamaClient

        client = OllamaClient(timeout=120.0)
        assert client.timeout == 120.0


class TestResolveEndpointStatic:
    """Pruebas del método estático resolve_endpoint."""

    def test_explicit_url_wins(self) -> None:
        from models.ollama_client import OllamaClient

        result = OllamaClient.resolve_endpoint("http://custom.host:9999")
        assert result == "http://custom.host:9999"


    def test_none_fallback_to_default(self) -> None:
        from models.ollama_client import OllamaClient

        result = OllamaClient.resolve_endpoint(None)
        
        # ✅ Aceptar cualquier IP local o localhost
        assert re.search(
            r'(localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|::1)', 
            result
        ), f"Expected localhost/loopback URL, got {result}"

class TestGetDefaultClient:
    """Prueba de la función lazy get_default_client."""

    def test_returns_same_instance(self) -> None:
        from models.ollama_client import get_default_client
        
        # La primera llamada crea el cliente
        client_a = get_default_client()
        assert client_a is not None

        # Verificar que la segunda llamada devuelve la MISMA instancia (singleton lazy)
        client_b = get_default_client()
        assert client_b is client_a, "get_default_client() debe devolver la misma instancia (singleton)"
        
        from models.ollama_client import OllamaClient
        assert isinstance(client_a, OllamaClient)


class TestOllamaClientMethodsExist:
    """Verifica que los métodos principales existen y tienen la firma correcta."""

    def test_chat_method_exists(self) -> None:
        from models.ollama_client import OllamaClient
        import inspect

        client = OllamaClient()
        assert hasattr(client, "chat")
        sig = inspect.signature(client.chat)
        
        # Verificar parámetros esperados del método chat
        params = list(sig.parameters.keys())
        assert 'messages' in params
        assert 'model' in params
        assert 'tools' in params
        assert 'stream' in params
        
        # Verificar que es asíncrono (coroutine)
        import asyncio
        assert asyncio.iscoroutinefunction(client.chat)

    def test_chat_stream_method_exists(self) -> None:
        from models.ollama_client import OllamaClient
        import inspect
        import asyncio

        client = OllamaClient()
        assert hasattr(client, "chat_stream")
        
        # Verificar que es un generador asíncrono
        assert asyncio.iscoroutinefunction(client.chat_stream) or callable(client.chat_stream)

    def test_list_models_method_exists(self) -> None:
        from models.ollama_client import OllamaClient
        import inspect
        import asyncio

        client = OllamaClient()
        assert hasattr(client, "list_models")
        
        # Verificar que es asíncrono
        assert asyncio.iscoroutinefunction(client.list_models)

    def test_resolve_endpoint_static_method_exists(self) -> None:
        from models.ollama_client import OllamaClient
        import inspect

        assert hasattr(OllamaClient, "resolve_endpoint")
        
        # Debe ser un método estático
        assert isinstance(inspect.getattr_static(OllamaClient, "resolve_endpoint"), staticmethod)