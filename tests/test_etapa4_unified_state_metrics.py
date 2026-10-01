"""
tests/test_etapa4_unified_state_metrics.py - Tests para ETAPA 4: Unificar estado y métricas

Verifica:
1. LocalAgent usa el MetricsLogger real (no el stub)
2. SessionState.add_tool_result() registra correctamente el resultado
3. Las métricas se persisten en JSONL tras ejecución autónoma
"""

import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock, patch, AsyncMock

# ============================================================================
# TEST 1: Verificar que LocalAgent usa MetricsLogger real
# ============================================================================

class TestLocalAgentUsesRealMetricsLogger:
    """Tests para verificar que LocalAgent no usa el stub local de MetricsLogger."""
    
    async def test_agent_uses_real_metrics_logger(self):
        """Verifica que LocalAgent instancia el MetricsLogger de metrics.logger, no el stub local."""
        
        # Primero, importar ambas clases para comparar tipos
        from metrics.logger import MetricsLogger as RealMetricsLogger
        
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            # Importar LocalAgent
            from core.agent import LocalAgent
            
            agent = LocalAgent(model_name='test_model')
            
            # Verificar que metrics_logger es una instancia del logger real
            assert isinstance(agent.metrics_logger, RealMetricsLogger), \
                f"Se esperaba MetricsLogger de metrics.logger, pero se obtuvo {type(agent.metrics_logger)}"
            
            # Verificar que tiene el atributo 'path' (propiedad del logger real)
            assert hasattr(agent.metrics_logger, 'path'), \
                "El logger debe tener un atributo 'path'"
            
            # Verificar que path no es None (debería ser la ruta absoluta a metrics/run.jsonl)
            assert agent.metrics_logger.path is not None, \
                "La ruta del logger no debería ser None"


# ============================================================================
# TEST 2: SessionState.add_tool_result() registra correctamente
# ============================================================================

class TestSessionStateAddToolResult:
    """Tests para el método add_tool_result de SessionState."""
    
    def test_add_tool_result_dict(self):
        """Verifica que add_tool_result con un dict funciona correctamente."""
        from core.state import SessionState
        
        state = SessionState()
        
        # Agregar resultado como dict
        result = {"status": "success", "data": [1, 2, 3]}
        msg = state.add_tool_result("step_001", result)
        
        # Verificar que se guardó en results_cache
        assert "step_001" in state.results_cache
        assert state.results_cache["step_001"] == result
        
        # Verificar que se agregó al historial de mensajes
        assert len(state.messages) == 1
        assert msg.role == "tool_result"
        assert "Tool step_001 result:" in msg.content
        assert json.loads(msg.content.split("result: ")[1])["status"] == "success"
        
        # Verificar metadata
        assert msg.metadata["tool_call_id"] == "step_001"
    
    def test_add_tool_result_string(self):
        """Verifica que add_tool_result con un string se normaliza correctamente."""
        from core.state import SessionState
        
        state = SessionState()
        
        # Agregar resultado como string (debería ser envuelto en dict)
        result_str = "Operación completada"
        msg = state.add_tool_result("step_002", result_str)
        
        # Verificar que se normalizó a dict
        assert "step_002" in state.results_cache
        assert state.results_cache["step_002"] == {"result": result_str}
        
        # Verificar que se agregó al historial
        assert len(state.messages) == 1
        assert msg.role == "tool_result"
        assert json.loads(msg.content.split("result: ")[1])["result"] == result_str
    
    def test_add_tool_result_multiple(self):
        """Verifica que múltiples resultados se agregan correctamente."""
        from core.state import SessionState
        
        state = SessionState()
        
        # Agregar varios resultados
        state.add_tool_result("step_001", {"output": "primer resultado"})
        state.add_tool_result("step_002", {"output": "segundo resultado"})
        state.add_tool_result("step_003", "tercer resultado en string")
        
        # Verificar que todos están en results_cache
        assert len(state.results_cache) == 3
        assert "step_001" in state.results_cache
        assert "step_002" in state.results_cache
        assert "step_003" in state.results_cache
        
        # Verificar que todos están en messages
        assert len(state.messages) == 3
        
        # Verificar que cada mensaje tiene role="tool_result"
        for msg in state.messages:
            assert msg.role == "tool_result"
    
    def test_add_tool_result_preserves_history(self):
        """Verifica que add_tool_result no rompe el historial existente."""
        from core.state import SessionState
        
        state = SessionState()
        
        # Agregar mensajes existentes
        state.add_message("user", "Hola")
        state.add_message("assistant", "¿Cómo puedo ayudarte?")
        
        count_before = len(state.messages)
        
        # Agregar resultado de herramienta
        state.add_tool_result("step_001", {"status": "ok"})
        
        # Verificar que se mantuvieron los mensajes anteriores
        assert len(state.messages) == count_before + 1
        assert state.messages[0].role == "user"
        assert state.messages[0].content == "Hola"
        assert state.messages[1].role == "assistant"
        assert state.messages[1].content == "¿Cómo puedo ayudarte?"
        
        # Verificar que el nuevo mensaje es tool_result
        assert state.messages[2].role == "tool_result"


# ============================================================================
# TEST 3: Métricas persistidas en JSONL tras ejecución autónoma
# ============================================================================

class TestMetricsPersistenceInAutonomousMode:
    """Tests para verificar que las métricas se persisten correctamente en JSONL."""
    
    def test_metrics_jsonl_created_and_valid(self):
        """Verifica que al crear MetricsLogger, se crea el archivo JSONL y es válido."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "test_persistence.jsonl")
            
            from metrics.logger import MetricsLogger
            
            logger = MetricsLogger(path=jsonl_path)
            
            # Escribir algunas métricas
            logger.log("test_metric_1", {"value": 42})
            logger.log("test_metric_2", {"status": "ok"})
            logger.record_task(task_success=True, latency_seconds=1.5)
            
            # Verificar que el archivo existe
            assert os.path.exists(jsonl_path), "El archivo JSONL debería haberse creado"
            
            # Leer y verificar validez de cada línea
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                
            # Debería haber 3 líneas (2 log + 1 record_task)
            assert len(lines) == 3, f"Se esperaban 3 líneas, se obtuvieron {len(lines)}"
            
            # Verificar que cada línea es JSON válido y tiene la estructura correcta
            for line in lines:
                entry = json.loads(line)
                assert 'timestamp' in entry, "Cada entrada debe tener 'timestamp'"
                assert 'metric' in entry, "Cada entrada debe tener 'metric'"
                assert 'value' in entry, "Cada entrada debe tener 'value'"
    
    def test_autonomous_mode_persists_metrics(self):
        """Verifica que una ejecución autónoma finaliza con métricas en JSONL."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "autonomous_test.jsonl")
            
            from metrics.logger import MetricsLogger
            from core.state import SessionState
            
            # Crear logger real con ruta en temp dir
            logger = MetricsLogger(path=jsonl_path)
            
            # Simular escritura de métricas típicas de un ciclo autónomo
            logger.log("loop_start", {"task": "Tarea de prueba"})
            logger.log("tool_call", {"tool": "filesystem", "iteration": 1})
            logger.log("orchestration_summary", {
                "latency": 1.5,
                "iterations_used": 3,
                "total_tool_calls_executed": 2
            })
            
            # Verificar persistencia
            assert os.path.exists(jsonl_path)
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                lines = [json.loads(line.strip()) for line in f if line.strip()]
            
            assert len(lines) == 3
            
            # Verificar que las métricas esperadas están presentes
            metrics_list = [entry['metric'] for entry in lines]
            assert 'loop_start' in metrics_list
            assert 'tool_call' in metrics_list
            assert 'orchestration_summary' in metrics_list


# ============================================================================
# TEST 4: Integración completa - LocalAgent + Logger Real + Persistencia
# ============================================================================

class TestFullIntegration:
    """Tests de integración completa de la ETAPA 4."""
    
    async def test_agent_instantiation_uses_real_logger(self):
        """Verifica que al instanciar LocalAgent, se usa el MetricsLogger real."""
        
        # Importar clases
        from metrics.logger import MetricsLogger as RealMetricsLogger
        
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            from core.agent import LocalAgent
            
            agent = LocalAgent(model_name='test_model')
            
            # Verificar tipo correcto
            assert isinstance(agent.metrics_logger, RealMetricsLogger)
            
            # Verificar que tiene los métodos del logger real
            assert hasattr(agent.metrics_logger, 'log')
            assert hasattr(agent.metrics_logger, 'record_task')
            assert hasattr(agent.metrics_logger, 'get_metrics')
            
            # Verificar que no es el stub (el stub no tiene path ni record_task)
            stub_attrs = {'path', 'record_task'}
            for attr in stub_attrs:
                assert hasattr(agent.metrics_logger, attr), \
                    f"El logger debería tener atributo '{attr}' del logger real"
    
    def test_session_state_tool_result_integration(self):
        """Verifica que SessionState y add_tool_result funcionan sin errores."""
        
        from core.state import SessionState
        
        state = SessionState()
        
        # Simular flujo completo: usuario -> herramienta -> resultado
        state.add_message("user", "Crear un archivo de prueba")
        
        # El LLM decide usar una herramienta, obtiene el step_id
        step_id = "call_abc123"
        tool_result = {"file_created": "test.txt", "path": "/tmp/test.txt"}
        
        # Registrar el resultado
        msg = state.add_tool_result(step_id, tool_result)
        
        # Verificar integridad
        assert step_id in state.results_cache
        
        # El estado debería ser serializable
        state_dict = state.to_dict()
        assert len(state_dict['messages']) == 2  # user + tool_result
        
        # Debería deserializarse correctamente
        restored_state = SessionState.from_dict(state_dict)
        assert len(restored_state.messages) == 2
        assert restored_state.results_cache[step_id] == tool_result


# ============================================================================
# TEST DE REGRESIÓN: Verificar que no rompimos nada existente
# ============================================================================

class TestRegression:
    """Tests de regresión para asegurar que los cambios no rompieron funcionalidad existente."""
    
    def test_session_state_backward_compatible(self):
        """Verifica que SessionState sigue funcionando con los métodos existentes."""
        
        from core.state import SessionState
        
        state = SessionState()
        
        # Métodos existentes deben seguir funcionando
        state.add_message("user", "Prueba")
        state.update_tokens(100)
        state.set_phase("plan")
        
        assert state.tokens_used == 100
        assert state.current_phase == "plan"
        assert len(state.messages) == 1
        
        # Serialización debe funcionar
        json_str = state.to_json()
        restored = SessionState.from_json(json_str)
        
        assert restored.session_id == state.session_id
        assert restored.tokens_used == state.tokens_used
        assert restored.current_phase == state.current_phase
    
    def test_real_logger_backward_compatible(self):
        """Verifica que el MetricsLogger real sigue funcionando con sus métodos existentes."""
        
        from metrics.logger import MetricsLogger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = os.path.join(temp_dir, "compat_test.jsonl")
            
            logger = MetricsLogger(path=jsonl_path)
            
            # Métodos existentes deben funcionar
            logger.log("test", 42)
            logger.record_task(task_success=True, latency_seconds=1.0)
            
            metrics = logger.get_metrics()
            assert len(metrics) == 2
            
            # El archivo debe existir y ser válido
            assert os.path.exists(jsonl_path)


# ============================================================================
# IMPORTACIÓN DE PYTEST (Necesario para asserts y fixtures)
# ============================================================================

import pytest  # noqa: F401