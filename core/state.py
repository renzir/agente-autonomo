"""
core/state.py - Estado de sesión del agente
Almacena historial de mensajes, tokens usados y estado del ciclo.
Persistible a JSON para recuperación futura.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Representa un mensaje en el historial de conversación."""
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convierte el mensaje a diccionario para serialización."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


class SessionState(BaseModel):
    """Estado de una sesión del agente. Serializable a JSON."""
    
    # Identificador de sesión
    session_id: str = Field(default_factory=lambda: f"session_{int(time.time())}")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    # Historial de mensajes
    messages: List[Message] = Field(default_factory=list)
    
    # Métricas del ciclo actual
    tokens_used: int = 0
    iterations_completed: int = 0
    current_phase: str = "idle"  # idle, understand, plan, execute, observe, verify, reflect, done
    
    # Estado del ciclo
    current_task: Optional[str] = None
    pending_actions: List[Dict[str, Any]] = Field(default_factory=list)
    results_cache: Dict[str, Any] = Field(default_factory=dict)
    
    # Configuración actual
    config_snapshot: Optional[Dict[str, Any]] = None
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Agrega un mensaje al historial."""
        msg = Message(
            role=role, 
            content=content, 
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg
    
    def update_tokens(self, tokens: int):
        """Actualiza el contador de tokens usados."""
        self.tokens_used += tokens
        self.updated_at = time.time()
    
    def set_phase(self, phase: str):
        """Establece la fase actual del ciclo."""
        self.current_phase = phase
        self.updated_at = time.time()
    
    def to_dict(self) -> dict:
        """Convierte el estado a diccionario para serialización."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'messages': [msg.to_dict() for msg in self.messages],
            'tokens_used': self.tokens_used,
            'iterations_completed': self.iterations_completed,
            'current_phase': self.current_phase,
            'current_task': self.current_task,
            'pending_actions': self.pending_actions,
            'results_cache': self.results_cache,
            'config_snapshot': self.config_snapshot
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serializa el estado a JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SessionState':
        """Crea una instancia desde un diccionario."""
        messages = [Message(**msg_data) for msg_data in data.get('messages', [])]
        return cls(
            session_id=data.get('session_id', f"session_{int(time.time())}"),
            created_at=data.get('created_at', time.time()),
            updated_at=data.get('updated_at', time.time()),
            messages=messages,
            tokens_used=data.get('tokens_used', 0),
            iterations_completed=data.get('iterations_completed', 0),
            current_phase=data.get('current_phase', 'idle'),
            current_task=data.get('current_task'),
            pending_actions=data.get('pending_actions', []),
            results_cache=data.get('results_cache', {}),
            config_snapshot=data.get('config_snapshot')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SessionState':
        """Crea una instancia desde un string JSON."""
        return cls.from_dict(json.loads(json_str))