"""
tests/test_state.py - Tests para core/state.py
Verifica la serialización y manejo del estado de sesión.
"""

import json
import pytest
from core.state import SessionState, Message


class TestMessage:
    """Tests para el modelo Message."""
    
    def test_create_message(self):
        """Crear un mensaje básico."""
        msg = Message(role='user', content='Hola')
        assert msg.role == 'user'
        assert msg.content == 'Hola'
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}
    
    def test_message_to_dict(self):
        """Convertir mensaje a diccionario."""
        msg = Message(role='assistant', content='Respuesta de prueba')
        msg_dict = msg.to_dict()
        
        assert msg_dict['role'] == 'assistant'
        assert msg_dict['content'] == 'Respuesta de prueba'
        assert 'timestamp' in msg_dict
        assert isinstance(msg_dict['metadata'], dict)


class TestSessionState:
    """Tests para el modelo SessionState."""
    
    def test_create_session(self):
        """Crear una sesión básica."""
        state = SessionState()
        assert state.session_id.startswith('session_')
        assert len(state.messages) == 0
        assert state.tokens_used == 0
        assert state.iterations_completed == 0
        assert state.current_phase == 'idle'
    
    def test_add_message(self):
        """Agregar mensajes al historial."""
        state = SessionState()
        
        msg1 = state.add_message('user', 'Mensaje 1')
        assert len(state.messages) == 1
        assert msg1.role == 'user'
        
        msg2 = state.add_message('assistant', 'Mensaje 2', metadata={'step': 1})
        assert len(state.messages) == 2
        assert msg2.metadata == {'step': 1}
    
    def test_update_tokens(self):
        """Actualizar contador de tokens."""
        state = SessionState()
        
        state.update_tokens(100)
        assert state.tokens_used == 100
        
        state.update_tokens(50)
        assert state.tokens_used == 150
    
    def test_set_phase(self):
        """Cambiar fase del ciclo."""
        state = SessionState()
        
        phases = ['understand', 'plan', 'execute', 'observe', 'verify', 'reflect']
        for phase in phases:
            state.set_phase(phase)
            assert state.current_phase == phase
    
    def test_to_dict_serialization(self):
        """Serializar estado a diccionario."""
        state = SessionState()
        state.add_message('user', 'Prueba')
        state.update_tokens(200)
        state.iterations_completed = 3
        
        state_dict = state.to_dict()
        
        assert 'session_id' in state_dict
        assert 'messages' in state_dict
        assert len(state_dict['messages']) == 1
        assert state_dict['tokens_used'] == 200
        assert state_dict['iterations_completed'] == 3
    
    def test_to_json_serialization(self):
        """Serializar estado a JSON."""
        state = SessionState()
        state.add_message('user', 'JSON Test')
        
        json_str = state.to_json()
        assert isinstance(json_str, str)
        
        # Verificar que es JSON válido
        data = json.loads(json_str)
        assert data['messages'][0]['content'] == 'JSON Test'
    
    def test_from_json_deserialization(self):
        """Deserializar estado desde JSON."""
        state1 = SessionState()
        state1.add_message('user', 'Mensaje original')
        state1.update_tokens(500)
        state1.iterations_completed = 5
        
        json_str = state1.to_json()
        state2 = SessionState.from_json(json_str)
        
        assert state2.session_id == state1.session_id
        assert len(state2.messages) == 1
        assert state2.messages[0].content == 'Mensaje original'
        assert state2.tokens_used == 500
        assert state2.iterations_completed == 5
    
    def test_from_dict_deserialization(self):
        """Deserializar estado desde diccionario."""
        state1 = SessionState()
        state1.add_message('assistant', 'Hola de vuelta')
        
        state_dict = state1.to_dict()
        state2 = SessionState.from_dict(state_dict)
        
        assert len(state2.messages) == 1
        assert state2.messages[0].role == 'assistant'


class TestSessionPersistence:
    """Tests de persistencia del estado."""
    
    def test_roundtrip_serialization(self):
        """Verificar roundtrip completo (serializar y deserializar)."""
        original = SessionState()
        original.add_message('user', 'Pregunta inicial')
        original.add_message('assistant', 'Respuesta generada', metadata={'token_count': 150})
        original.update_tokens(200)
        original.current_task = 'Tarea de prueba'
        original.iterations_completed = 3
        
        # Serializar a JSON
        json_str = original.to_json(indent=4)
        
        # Deserializar desde JSON
        restored = SessionState.from_json(json_str)
        
        # Verificar integridad
        assert restored.session_id == original.session_id
        assert len(restored.messages) == len(original.messages)
        assert restored.messages[0].content == 'Pregunta inicial'
        assert restored.messages[1].metadata['token_count'] == 150
        assert restored.tokens_used == 200
        assert restored.current_task == 'Tarea de prueba'
        assert restored.iterations_completed == 3


class TestEdgeCases:
    """Tests para casos límite."""
    
    def test_empty_session(self):
        """Crear sesión vacía."""
        state = SessionState()
        assert len(state.messages) == 0
        assert state.tokens_used == 0
    
    def test_many_messages(self):
        """Sesión con muchos mensajes."""
        state = SessionState()
        
        for i in range(100):
            state.add_message('user' if i % 2 == 0 else 'assistant', f'Mensaje {i}')
        
        assert len(state.messages) == 100
    
    def test_large_tokens(self):
        """Manejar conteo grande de tokens."""
        state = SessionState()
        state.update_tokens(999999)
        assert state.tokens_used == 999999