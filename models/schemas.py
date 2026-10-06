"""Pydantic v2 schemas para el agente personal local.

Esquemas utilizados en toda la aplicación para validación tipada.
Corresponde a la sección models/schemas.py del proyecto.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ======================================================================
# Mensajes y herramientas (sección de comunicación / tools)
# ======================================================================

class ChatMessage(BaseModel):
    """Representa un mensaje en el chat con el agente.
    
    Correspondiente a la sección de comunicación del agente.
    """
    role: str  # "system", "user", "assistant"
    content: str


class ToolCallRequest(BaseModel):
    """Solicitud de llamada a una herramienta."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Resultado de la ejecución de una herramienta."""
    tool_name: str
    content: str
    error: Optional[str] = None




class ToolMetadata(BaseModel):
    """Metadatos descriptivos para cada herramienta disponible."""
    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    permission: str = "read"  # read, write, execute, admin
    risk: str = "low"  # 'low', 'medium', 'high'
    timeout: float = 30.0  # segundos máximos de ejecución
    cost: float = 0.0  # costo estimado en tokens
    requires_confirmation: bool = False


# ======================================================================
# Esquemas adicionales para Planner (agregados para corregir ImportError)
# ======================================================================

class Step(BaseModel):
    """Representa un paso en el plan de ejecución."""
    id: int = Field(default=0)
    description: str = ""
    depends_on: list[int] = Field(default_factory=list)
    tool: str = ""  # Nombre de la herramienta
    args: dict[str, Any] = Field(default_factory=dict)
    expected_output: str = ""

class PlannerRequest(BaseModel):
    """Solicitud al planner."""
    task: str
    context: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 2048

class PlannerResponse(BaseModel):
    """Respuesta del planner con la lista de pasos."""
    steps: list[Step] = Field(default_factory=list)
    estimated_tokens: int = 0
    max_iterations: int = 0


class InteractionMode(str, Enum):
    CONVERSATIONAL = "conversational"  # Modo /chat
    AUTONOMOUS = "autonomous"          # Modo /agente (predeterminado)

# ======================================================================
# Configuración del agente (sección config/agent.yaml)
# ======================================================================

class AgentConfig(BaseModel):
    """Configuración del agente cargada desde config/agent.yaml.
    
    Correspondiente a la sección de configuración del agente (config/).
    """
    max_iterations: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=30, ge=1)
    max_execution_time: float = Field(default=300.0, gt=0)  # segundos

    class ContextConfig(BaseModel):
        hard_limit: int = Field(default=32768, gt=0)
        target_input: int = Field(default=14000, gt=0)
        warning: int = Field(default=18000, gt=0)
        compression: int = Field(default=20000, gt=0)
        emergency_limit: int = Field(default=24000, gt=0)

    context: ContextConfig = Field(default_factory=ContextConfig)
    
    class SafetyConfig(BaseModel):
        sandbox_enabled: bool = True
        allowed_dirs: list[str] = Field(default_factory=lambda: ["."])
        command_timeout: float = 30.0
        red_network_by_default: bool = True

    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    @classmethod
    def from_file(cls, filepath: str | Path = "config/agent.yaml") -> AgentConfig:
        """Carga la configuración desde un archivo YAML."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        # Combina agent + context + safety del YAML en un solo dict para Pydantic
        combined = {}
        combined.update(data.get("agent", {}))
        combined["context"] = data.get("context", {})
        combined["safety"] = data.get("safety", {})
        
        return cls(**combined)


# ======================================================================
# Configuración de modelos (sección config/models.yaml)
# ======================================================================

class ModelConfig(BaseModel):
    """Configuración de modelos cargada desde config/models.yaml.
    
    Correspondiente a la sección de configuración de modelos (config/).
    """
    models: dict[str, str] = Field(default_factory=dict)  # nombre -> identificador
    defaults: dict[str, Any] = Field(default_factory=lambda: {
        "model": "primary",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
        "keep_alive_seconds": 1800,
    })

    @classmethod
    def from_file(cls, filepath: str | Path = "config/models.yaml") -> ModelConfig:
        """Carga la configuración desde un archivo YAML."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Archivo de config de modelos no encontrado: {filepath}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        # asegura estructura mínima esperada
        return cls(
            models=data.get("models", {}),
            defaults=data.get("defaults", {}),
        )