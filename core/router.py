"""
core/router.py - Enrutador simplificado del agente.
Objetivo: Reducir la predicción temprana de herramientas y hacer el enrutamiento robusto.
El clasificador de intención ahora es estricto con prefijos (/chat, /agente).
El router ya no adivina la herramienta basada en keywords complejas.
"""
import re
import logging
from typing import Tuple, Callable, Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Importamos el modo de interacción directo si está disponible, sino fallback
try:
    from models.schemas import InteractionMode, ToolMetadata
except ImportError:
    # Fallback para compatibilidad en tests o entornos sin imports completos
    class InteractionMode:
        CONVERSATIONAL = "conversational"
        AUTONOMOUS = "autonomous"

class IntentClassifier:
    """
    Clasificador LIGERO y ERECTO de intención basado SOLO en prefijos explícitos.
    No usa heurísticas de palabras clave para evitar falsos positivos.
    
    Reglas estrictas:
    1. Input que empieza con '/chat' (o '/chat ') -> CONVERSATIONAL.
    2. Input que empieza con '/agente' (o '/agente ') -> AUTONOMOUS.
    3. Cualquier otro input (sin prefijo) -> AUTONOMOUS por defecto.
    """
    
    @classmethod
    def classify(cls, text: str) -> str:
        """
        Clasifica el input del usuario según comandos explícitos al inicio.
        
        Args:
            text: Input crudo del usuario
            
        Returns:
            InteractionMode.CONVERSATIONAL o InteractionMode.AUTONOMOUS
        """
        if not text or not isinstance(text, str):
            return InteractionMode.AUTONOMOUS
        
        text_stripped = text.strip().lower()
        
        # Detección estricta de prefijo al inicio para evitar activaciones erróneas
        # Ej: "hola /chat" NO debe activar conversacional. Solo "/chat hola sí" SÍ.
        
        if text_stripped.startswith('/chat') and (len(text_stripped) == 5 or text_stripped[5] in (' ', '\n', '\t')):
            return InteractionMode.CONVERSATIONAL
            
        if text_stripped.startswith('/agente') and (len(text_stripped) == 7 or text_stripped[7] in (' ', '\n', '\t')):
            return InteractionMode.AUTONOMOUS
        
        # Default: Si no hay prefijo explícito, asumir modo autónomo (comportamiento original seguro)
        return InteractionMode.AUTONOMOUS


class RouterDecision(BaseModel):
    """Resultado de la decisión del enrutador simplificado."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    
    def to_dict(self) -> dict:
        return {
            'tool_name': self.tool_name,
            'args': self.args,
            'confidence': self.confidence,
            'reason': self.reason
        }


class Router:
    """
    Enrutador simplificado.
    
    Ya no intenta adivinar qué herramienta (filesystem, search, shell) necesita el agente
    basándose en el contenido semántico del input para evitar falsos positivos.
    
    Ahora delega la selección fina de herramientas al flujo posterior del agente (native tool calling / planner).
    Su rol se reduce a validar disponibilidad y aplicar prioridades básicas si es necesario para el fallback.
    """
    
    def __init__(self, tool_registry: Dict[str, ToolMetadata] = None):
        self.logger = logging.getLogger(__name__)
        self.tool_registry = tool_registry or {}

    def route(self, step) -> RouterDecision:
        print("[TRACE] ROUTER START")

        tool_name = step.tool
        args = step.args or {}

        if not tool_name:
            return RouterDecision(
                tool_name="",
                confidence=0.0,
                reason=f"El step {step.id} no especifica herramienta"
            )

        try:
            tool = self.tool_registry.get_tool(tool_name)

            self.tool_registry.validate_input(tool_name, args)

            decision = RouterDecision(
                tool_name=tool_name,
                args=args,
                confidence=1.0,
                reason=f"Herramienta '{tool_name}' resuelta desde ToolRegistry"
            )

            self.logger.debug(
                f"Router resolvió step {step.id}: {tool_name}"
            )

            return decision

        except KeyError:
            return RouterDecision(
                tool_name="",
                args={},
                confidence=0.0,
                reason=f"Herramienta '{tool_name}' no existe en ToolRegistry"
            )

        except ValueError as e:
            return RouterDecision(
                tool_name="",
                args={},
                confidence=0.0,
                reason=f"Argumentos inválidos para '{tool_name}': {e}"
            )
    
    
    def update_registry(self, tools: Dict[str, ToolMetadata]):
        """Actualiza el registry de herramientas."""
        self.tool_registry.update(tools)