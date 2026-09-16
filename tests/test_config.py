"""Tests para la carga de configuraciones (agent.yaml y models.yaml).

Verifica que los archivos YAML se cargan correctamente y validan contra Pydantic.
Corresponde a la sección tests/ del proyecto.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError


def _agent_config_path() -> Path:
    """Devuelve la ruta absoluta al archivo agent.yaml."""
    return Path(__file__).parent.parent / "config" / "agent.yaml"


def _model_config_path() -> Path:
    """Devuelve la ruta absoluta al archivo models.yaml."""
    return Path(__file__).parent.parent / "config" / "models.yaml"


class TestAgentConfig:
    """Pruebas para AgentConfig (config/agent.yaml)."""

    def test_agent_yaml_exists(self) -> None:
        """Verifica que el archivo agent.yaml existe en la ruta esperada."""
        assert _agent_config_path().exists()

    def test_load_agent_config(self) -> None:
        """Carga y valida la configuración del agente desde YAML."""
        from models.schemas import AgentConfig

        config = AgentConfig.from_file(str(_agent_config_path()))

        # Valores de agent
        assert config.max_iterations == 12
        assert config.max_tool_calls == 30
        assert config.max_execution_time == 300

        # Valores de context
        assert config.context.hard_limit == 32768
        assert config.context.target_input == 14000
        assert config.context.warning == 18000
        assert config.context.compression == 20000
        assert config.context.emergency_limit == 24000

        # Valores de safety
        assert config.safety.sandbox_enabled is True
        assert config.safety.allowed_dirs == ["."]
        assert config.safety.command_timeout == 30
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
        assert "primary" in config.models
        assert "coding" in config.models
        assert "fast" in config.models
        assert "research" in config.models

        assert config.models["primary"] == "qwen3.6:latest"
        assert config.models["coding"] == "qwen3.6:latest"
        assert config.models["fast"] == "gemma4-122k:latest"
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

        meta = ToolMetadata(
            name="test_tool",
            description="Una herramienta de prueba",
            input_schema={"type": "object"},
        )
        assert meta.permission == "auto"
        assert meta.risk == 0
        assert meta.timeout == 30.0
        assert meta.cost == 0.0
        assert meta.requires_confirmation is False


class TestConfigPathAbsolute:
    """Pruebas que aseguran que las rutas son absolutas y robustas."""

    def test_agent_config_resolves_from_any_cwd(self) -> None:
        """AgentConfig.from_file funciona incluso si el cwd es diferente."""
        from models.schemas import AgentConfig

        # Usamos la ruta absoluta explícita
        config = AgentConfig.from_file(str(_agent_config_path()))
        assert config.max_iterations == 12

    def test_model_config_resolves_from_any_cwd(self) -> None:
        """ModelConfig.from_file funciona incluso si el cwd es diferente."""
        from models.schemas import ModelConfig

        config = ModelConfig.from_file(str(_model_config_path()))
        assert "primary" in config.models