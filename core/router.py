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
        # Prioridad de fallback solo para selección inicial, no para predicción temprana
        self.priority_order = ['filesystem', 'shell', 'search', 'git'] 
    
    def route(self, intention: str, available_tools: Dict[str, ToolMetadata] = None) -> RouterDecision:
        """
        Decide qué herramienta usar.
        
        Simplificado: No usa keywords complejas para predecir la herramienta.
        Si hay una intención clara o contexto previo, podría usarse, pero por defecto
        devuelve un tool_name basado en prioridad/simple match para mantener compatibilidad
        sin adivinar.
        
        Args:
            intention: Texto de la intención (ya procesado opcionalmente por IntentClassifier)
            available_tools: Diccionario de herramientas disponibles
            
        Returns:
            RouterDecision con la herramienta elegida (por prioridad o default)
        """
        print("[TRACE] ROUTER START")
        tools_to_use = available_tools or self.tool_registry
        
        if not tools_to_use:
            return RouterDecision(
                tool_name="",
                confidence=0.0,
                reason="No hay herramientas disponibles"
            )
        
        # Simplificación: Ya no hacemos keyword matching complejo aquí.
        # Devolvemos la herramienta de mayor prioridad disponible para mantener el flujo.
        # La selección real se hará más tarde en el agente.
        decision = self._simple_match(tools_to_use)
        
        self.logger.debug(f"Decisión del router simplificado: {decision.tool_name}")
        print(f"[TRACE] ROUTER DECISION: {decision}")
        return decision
    
    def _simple_match(self, available_tools: Dict[str, ToolMetadata]) -> RouterDecision:
        """
        Selección simple por prioridad. Elimina la lógica de adivinanza.
        """
        # 1. Intentar encontrar una herramienta en el orden de prioridad estándar
        for tool_name in self.priority_order:
            if tool_name in available_tools:
                return RouterDecision(
                    tool_name=tool_name,
                    confidence=0.5, # Confianza base ya que no hay predicción semántica
                    reason="Selección por prioridad de fallback"
                )
        
        # 2. Si ninguna está en la lista de prioridad, tomar la primera disponible
        for tool_name in available_tools.keys():
            return RouterDecision(
                tool_name=tool_name,
                confidence=0.3,
                reason="Selección por disponibilidad (fallback)"
            )

        # 3. Si no hay absolutamente nada
        return RouterDecision(
            tool_name="",
            confidence=0.0,
            reason="No hay herramientas disponibles en el registry"
        )
    
    def update_registry(self, tools: Dict[str, ToolMetadata]):
        """Actualiza el registry de herramientas."""
        self.tool_registry.update(tools)