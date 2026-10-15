"""Tests para la carga de configuraciones (agent.yaml y models.yaml).

Verifica que los archivos YAML se cargan correctamente y validan contra Pydantic.
Corresponde a la sección tests/ del proyecto.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError


def _agent_config_path() -> Path:
    """Devuelve la ruta absoluta al archivo agent.yaml (raíz del proyecto)."""
    return Path(__file__).parent.parent.parent / "config" / "agent.yaml"


def _model_config_path() -> Path:
    """Devuelve la ruta absoluta al archivo models.yaml (raíz del proyecto)."""
    return Path(__file__).parent.parent.parent / "config" / "models.yaml"



class TestAgentConfig:
    """Pruebas para AgentConfig (config/agent.yaml)."""

    def test_agent_yaml_exists(self) -> None:
        """Verifica que el archivo agent.yaml existe en la ruta esperada."""
        assert _agent_config_path().exists()

    def test_load_agent_config(self) -> None:
        """Carga y valida la configuración del agente desde YAML."""
        from models.schemas import AgentConfig

        config = AgentConfig.from_file(str(_agent_config_path()))

        # Verifica valores de agent (depende de los datos reales en config/agent.yaml)
        # Nota: Asegúrate de que tu YAML tenga estos valores o ajusta aquí.
        # Si el YAML es vacío o usa defaults, estos deberían coincidir con los defaults de Pydantic:
        assert config.max_iterations == 12
        assert config.max_tool_calls == 30
        assert config.max_execution_time == 900.0  # Default de schemas.py

        # Valores de context (defaults de Pydantic)
        assert config.context.hard_limit == 32768
        assert config.context.target_input == 14000
        assert config.context.warning == 18000
        assert config.context.compression == 20000
        assert config.context.emergency_limit == 24000

        # Valores de safety (defaults de Pydantic)
        assert config.safety.sandbox_enabled is True
        assert config.safety.allowed_dirs == ["."]
        assert config.safety.command_timeout == 30.0  # float en schemas.py
        assert config.safety.red_network_by_default is True

    def test_agent_config_defaults(self) -> None:
        """Verifica que AgentConfig usa valores por defecto al no recibir parámetros."""
        from models.schemas import AgentConfig

        # Sin archivos — usa los defaults de Pydantic
        config = AgentConfig()
        assert config.max_iterations == 12
        assert config.context.hard_limit == 32768
        assert config.safety.sandbox_enabled is True

    def test_agent_config_invalid_max_iterations(self) -> None:
        """Verifica que se lanza ValidationError si max_iterations < 1."""
        from models.schemas import AgentConfig

        with pytest.raises(ValidationError):
            AgentConfig(max_iterations=0)


class TestModelConfig:
    """Pruebas para ModelConfig (config/models.yaml)."""

    def test_model_yaml_exists(self) -> None:
        """Verifica que el archivo models.yaml existe en la ruta esperada."""
        assert _model_config_path().exists()

    def test_load_model_config(self) -> None:
        """Carga y valida la configuración de modelos desde YAML."""
        from models.schemas import ModelConfig

        config = ModelConfig.from_file(str(_model_config_path()))

        # Verifica que todos los modelos esperados existen
        # Nota: Estos nombres deben coincidir con las claves en tu config/models.yaml
        assert "primary" in config.models
        assert "coding" in config.models
        assert "fast" in config.models
        assert "research" in config.models

        # Valores de los modelos (ajusta si tu YAML tiene otros identificadores)
        assert config.models["primary"] == "qwen3.6-64k:latest"
        assert config.models["coding"] == "qwen3.6-64k:latest"
        assert config.models["fast"] == "qwen3.6-64k:latest"
        assert config.models["research"] == "ornith15-iq4xs:latest"

    def test_model_config_defaults(self) -> None:
        """Verifica que los valores por defecto son correctos."""
        from models.schemas import ModelConfig

        config = ModelConfig()  # sin archivos
        assert config.defaults["model"] == "primary"
        assert config.defaults["temperature"] == 0.7
        assert config.defaults["top_p"] == 0.9
        assert config.defaults["max_tokens"] == 2048
        assert config.defaults["keep_alive_seconds"] == 1800

    def test_model_config_overrides(self) -> None:
        """Verifica que se pueden sobreescribir los defaults."""
        from models.schemas import ModelConfig

        custom_defaults = {
            "model": "coding",
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 4096,
        }
        config = ModelConfig(defaults=custom_defaults)
        assert config.defaults["model"] == "coding"
        assert config.defaults["temperature"] == 0.3


class TestChatMessage:
    """Pruebas para ChatMessage (sección de comunicación)."""

    def test_chat_message_creation(self) -> None:
        from models.schemas import ChatMessage

        msg = ChatMessage(role="user", content="Hola")
        assert msg.role == "user"
        assert msg.content == "Hola"


class TestToolCallRequest:
    """Pruebas para ToolCallRequest."""

    def test_tool_call_defaults(self) -> None:
        from models.schemas import ToolCallRequest

        tool = ToolCallRequest(name="filesystem", arguments={"path": "/tmp"})
        assert tool.name == "filesystem"
        assert tool.arguments["path"] == "/tmp"


class TestToolResult:
    """Pruebas para ToolResult."""

    def test_tool_result_with_error(self) -> None:
        from models.schemas import ToolResult

        result = ToolResult(tool_name="shell", content="", error="Command failed")
        assert result.error == "Command failed"


class TestToolMetadata:
    """Pruebas para ToolMetadata."""

    def test_tool_metadata_defaults(self) -> None:
        from models.schemas import ToolMetadata

        # Crear sin input_schema para probar default (debería usar Field(default_factory=dict))
        meta = ToolMetadata(
            name="test_tool",
            description="Una herramienta de prueba",
        )
        assert meta.permission == "read"
        assert meta.risk == "low"
        assert meta.input_schema == {}  # Default vacío
        
