"""
tests/test_agent.py - Tests de integración para core/agent.py
Verifica que la capa superior conecta correctamente config, models y core.
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import MagicMock, patch

# Importar el módulo bajo prueba
from core.agent import LocalAgent, MetricsLogger, create_agent


class TestMetricsLogger:
    """Tests para la clase MetricsLogger."""
    
    def test_log_simple(self):
        """Loguear una métrica simple."""
        logger = MetricsLogger()
        logger.log('test_metric', 100)
        
        assert len(logger.metrics_history) == 1
        assert logger.metrics_history[0]['metric'] == 'test_metric'
        assert logger.metrics_history[0]['value'] == 100
    
    def test_log_complex(self):
        """Loguear una métrica compleja (dict)."""
        logger = MetricsLogger()
        data = {'status': 'ok', 'latency': 0.5}
        logger.log('perf_test', data)
        
        assert len(logger.metrics_history) == 1
        assert logger.metrics_history[0]['value'] == data
    
    def test_log_multiple(self):
        """Loguear múltiples métricas."""
        logger = MetricsLogger()
        
        for i in range(5):
            logger.log(f'metric_{i}', i)
        
        assert len(logger.metrics_history) == 5


class TestCreateAgent:
    """Tests para la fábrica create_agent."""
    
    def test_create_agent_default(self):
        """Crear agente con configuración por defecto."""
        agent = create_agent(model_name='qwen3.6:latest')
        
        assert isinstance(agent, LocalAgent)
        # Verificar que componentes críticos existen
        assert hasattr(agent, 'orchestrator')
        assert hasattr(agent, 'planner')
        assert hasattr(agent, 'router')
        assert hasattr(agent, 'metrics_logger')


class TestLocalAgentInit:
    """Tests para la inicialización de LocalAgent."""
    
    def test_init_with_config_file(self):
        """Inicializar agente con archivo de config."""
        # Crear archivo temporal con config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'agent': {'max_iterations': 20},
                'security': {'strict_mode': False}
            }, f)
            config_path = f.name
        
        try:
            agent = LocalAgent(config_path=config_path, model_name='test_model')
            
            # Verificar que cargó la config (max_iterations debería ser 20)
            assert agent.config.get('agent', {}).get('max_iterations') == 20
        finally:
            os.unlink(config_path)
    
    def test_init_without_config_file(self):
        """Inicializar agente con defaults si no hay config."""
        agent = LocalAgent(config_path='/nonexistent/path.yaml', model_name='test_model')
        
        # Debería usar defaults
        assert agent.config.get('agent', {}).get('max_iterations') == 10
    
    def test_init_components_exist(self):
        """Verificar que todos los componentes se inicializan."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            assert agent.ollama_client is not None
            assert agent.model_registry is not None
            assert agent.planner is not None
            assert agent.router is not None
            assert agent.orchestrator is not None


class TestLocalAgentRun:
    """Tests para el método run del agente."""
    
    def test_run_returns_message(self):
        """run() debe devolver un Message."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            # Mockeamos el orchestrator para evitar llamadas reales a LLM
            mock_response = MagicMock()
            mock_response.content = "Respuesta de prueba"
            # Reemplazar completamente el objeto run con un MagicMock
            agent.orchestrator.run = MagicMock(return_value=mock_response)
            
            response = agent.run("Hola mundo")
            
            assert response is not None
            assert hasattr(response, 'content') or isinstance(response, str)

    def test_run_creates_session(self):
        """run() debe crear una sesión."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            # Mockeamos el orchestrator para evitar llamadas reales a LLM
            mock_response = MagicMock()
            mock_response.content = "Respuesta"
            agent.orchestrator.run = MagicMock(return_value=mock_response)
            
            agent.run("Prueba de sesión")
            
            assert agent.current_session is not None

    def test_run_with_empty_input(self):
        """run() debe manejar input vacío sin romper."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            mock_response = MagicMock()
            mock_response.content = "Empty response"
            agent.orchestrator.run = MagicMock(return_value=mock_response)
            
            response = agent.run("")
            
            # No debería levantar excepción
            assert response is not None

class TestLocalAgentLifecycle:
    """Tests del ciclo de vida del agente."""
    
    def test_get_metrics_returns_list(self):
        """get_metrics() debe devolver una lista."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            metrics = agent.get_metrics()
            
            assert isinstance(metrics, list)
    
    def test_reset_session_clears_state(self):
        """reset_session() debe limpiar el estado."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            # Simular actividad
            agent.current_session = MagicMock()
            agent.metrics_logger.metrics_history.append({'test': 'data'})
            
            # Resetear
            agent.reset_session()
            
            assert agent.current_session is None
            assert len(agent.metrics_logger.metrics_history) == 0


class TestIntegrationToolMap:
    """Tests específicos para el mapa de herramientas (_get_tools_map)."""
    
    def test_get_tools_map_returns_callable(self):
        """_get_tools_map debe devolver un dict con callables."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            tools = agent._get_tools_map()
            
            assert isinstance(tools, dict)
            
            # Verificar que las herramientas clave existen y son funciones
            for tool_name in ['filesystem', 'search', 'shell', 'git']:
                assert tool_name in tools
                assert callable(tools[tool_name])


class TestEdgeCases:
    """Tests para casos límite."""
    
    def test_agent_init_with_invalid_yaml(self):
        """Inicializar con YAML inválido debe usar defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content:::")  # YAML inválido
            config_path = f.name
        
        try:
            agent = LocalAgent(config_path=config_path, model_name='test_model')
            
            # Debería haber caído en defaults
            assert agent.config.get('agent', {}).get('max_iterations') == 10
        finally:
            os.unlink(config_path)
    
    def test_run_with_mock_failures(self):
        """run() debe manejar fallos internos gracefully."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            # Simular fallo en el orchestrator
            def raise_error(*args, **kwargs):
                raise Exception("Orchestrator error")
            
            agent.orchestrator.run = MagicMock(side_effect=raise_error)
            
            # run() debería propagar la excepción o manejarla según implementación
            with pytest.raises(Exception):
                agent.run("Tarea que falla")
                
class TestMetricsPropagation:
    """Tests para verificar que las métricas se propagan correctamente."""
    
    def test_metrics_logged_during_run(self):
        """Verificar que las métricas se loguean al ejecutar run()."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            mock_response = MagicMock()
            mock_response.content = "Ok"

            # Spy para verificar que logger_callback fue pasado al orchestrator
            captured_callbacks = []
            original_run = agent.orchestrator.run
            
            def track_callback(*args, **kwargs):
                if 'logger_callback' in kwargs:
                    callback = kwargs['logger_callback']
                    captured_callbacks.append(callback)
                return mock_response

            with patch.object(agent.orchestrator, 'run', side_effect=track_callback):
                agent.run("Prueba métricas")
            
            # Verificar que logger_callback fue pasado (el orchestrator intenta loguear con él)
            assert len(captured_callbacks) == 1
            
    def test_metrics_structure(self):
        """Verificar la estructura de las métricas logueadas."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            agent = LocalAgent(model_name='test_model')
            
            mock_response = MagicMock()
            mock_response.content = "Ok"
            agent.orchestrator.run = MagicMock(return_value=mock_response)
            
            agent.run("Estructura métricas")
            
            if agent.metrics_logger.metrics_history:
                metric = agent.metrics_logger.metrics_history[0]
                
                assert 'timestamp' in metric
                assert 'metric' in metric
                assert 'value' in metric