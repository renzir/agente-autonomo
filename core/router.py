"""
core/router.py - Enrutador de herramientas del agente
Recibe la intención del usuario y decide qué herramienta usar.
Basado en metadatos de herramientas del registry.
"""

import logging
from typing import Callable, Dict, List, Optional, Any
from pydantic import BaseModel, Field

# Importar modelos existentes para compatibilidad
try:
    from models.schemas import ToolMetadata
except ImportError:
    # Fallback si los modelos no están disponibles
    class ToolMetadata(BaseModel):
        name: str
        description: str
        input_schema: Dict[str, Any] = Field(default_factory=dict)
        permission: str = "default"  # read, write, execute, admin
        risk: str = "low"  # low, medium, high, critical
        timeout: int = 30
        cost: float = 0.0


class RouterDecision(BaseModel):
    """Resultado de la decisión del enrutador."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    
    def to_dict(self) -> dict:
        """Convierte a diccionario."""
        return {
            'tool_name': self.tool_name,
            'args': self.args,
            'confidence': self.confidence,
            'reason': self.reason
        }


class Router:
    """Enrutador que decide qué herramienta usar según la intención del usuario."""
    
    def __init__(self, tool_registry: Dict[str, ToolMetadata] = None):
        self.logger = logging.getLogger(__name__)
        self.tool_registry = tool_registry or {}
        self.priority_order = [
            'filesystem', 'search', 'shell', 'git', 'browser', 'code_editor'
        ]
    
    def route(self, intention: str, available_tools: Dict[str, ToolMetadata] = None) -> RouterDecision:
        """
        Decide qué herramienta usar basándose en la intención del usuario.
        
        Args:
            intention: Descripción de lo que el usuario quiere hacer
            available_tools: Diccionario de herramientas disponibles {name: metadata}
            
        Returns:
            RouterDecision con la herramienta y args elegidos
        """
        self.logger.info(f"Enrutando intención: {intention[:50]}...")
        
        tools_to_use = available_tools or self.tool_registry
        
        # Si no hay herramientas, devolver decisión por defecto
        if not tools_to_use:
            return RouterDecision(
                tool_name="",
                confidence=0.0,
                reason="No hay herramientas disponibles"
            )
        
        # Analizar la intención para determinar la herramienta
        decision = self._match_intention(intention, tools_to_use)
        
        self.logger.info(f"Decisión del router: {decision.tool_name} "
                        f"(confianza: {decision.confidence:.2f})")
        return decision
    
    def _match_intention(self, intention: str, 
                        available_tools: Dict[str, ToolMetadata]) -> RouterDecision:
        """Empareja la intención del usuario con herramientas disponibles."""
        
        # Palabras clave para cada tipo de herramienta
        keyword_mapping = {
            'filesystem': ['leer', 'escribir', 'archivo', 'crear', 'eliminar', 
                          'renombrar', 'copiar', 'mover', 'carpeta', 'directorio',
                          'read', 'write', 'file', 'folder', 'directory'],
            'search': ['buscar', 'encontrar', 'buscar en internet', 'google',
                      'search', 'find', 'query'],
            'shell': ['ejecutar', 'command', 'cmd', 'terminal', 'bash', 'sh', 
                     'run', 'execute', 'proceso'],
            'git': ['git', 'commit', 'push', 'pull', 'branch', 'repositorio',
                   'repo', 'merge', 'clonar'],
            'browser': ['navegar', 'web', 'url', 'abrir página', 'scraper',
                       'browse', 'visit', 'website'],
            'code_editor': ['editar código', 'programar', 'refactorizar', 
                           'debug', 'compilar', 'code', 'edit']
        }
        
        # Calcular score para cada herramienta
        scores = {}
        intention_lower = intention.lower()
        
        for tool_name, keywords in keyword_mapping.items():
            if tool_name not in available_tools:
                continue
            
            score = 0.0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in intention_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Bonus por riesgo permitido si está en config
            tool_meta = available_tools[tool_name]
            if tool_meta.permission == "execute" and "ejecutar" in intention_lower:
                score += 0.5
            
            scores[tool_name] = {
                'score': score,
                'matched_keywords': matched_keywords
            }
        
        # Seleccionar la herramienta con mayor score
        if not scores:
            # Si no hay match por keywords, usar prioridad
            for tool_name in self.priority_order:
                if tool_name in available_tools:
                    return RouterDecision(
                        tool_name=tool_name,
                        confidence=0.3,
                        reason="Selección por prioridad (sin keywords coincidentes)"
                    )
        
        best_tool = max(scores.items(), key=lambda x: x[1]['score'])
        tool_name = best_tool[0]
        score = best_tool[1]['score']
        
        # Normalizar confianza basado en el score
        confidence = min(1.0, score / 2.0)  # Escala normalizada
        
        matched_keywords = best_tool[1]['matched_keywords']
        reason = f"Keywords coincidentes: {', '.join(matched_keywords)}" if matched_keywords else "Selección por análisis semántico"
        
        return RouterDecision(
            tool_name=tool_name,
            confidence=confidence,
            reason=reason
        )
    
    def update_registry(self, tools: Dict[str, ToolMetadata]):
        """Actualiza el registry de herramientas."""
        self.tool_registry.update(tools)
        self.logger.info(f"Registry actualizado con {len(tools)} herramientas")