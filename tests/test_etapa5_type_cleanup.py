"""Tests para la ETAPA 5: limpieza final de tipos, esquemas y validaciones."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import logging


# --- Test A: Única definición de ToolMetadata ---
class TestSingleToolMetadataDefinition:
    """Verifica que no exista más de una definición de ToolMetadata."""
    
    def test_single_tool_metadata_definition(self):
        """Solo models.schemas debe definir ToolMetadata y router usa la misma."""
        from models.schemas import ToolMetadata as SchemasMetadata
        
        # Importar desde router.py (que lo trae de schemas via try/except)
        from core.router import ToolMetadata as RouterMetadata
        
        # Deben ser exactamente el mismo objeto (no dos clases distintas)
        assert SchemasMetadata is RouterMetadata, \
            "ToolMetadata en router debe ser idéntico al de schemas (misma clase)"


# --- Test B: Resultados de herramientas aceptados por Orchestrator ---
class TestOrchestratorAcceptsMixedResults:
    """Verifica que el Orchestrator acepte resultados str y dict de herramientas."""

    async def test_orchestrator_handles_str_result(self):
        """El orchestrator debe aceptar un resultado str de una herramienta."""
        from core.state import SessionState
        from core.orchestrator import Orchestrator
        
        mock_llm = MagicMock()
        # Mock para que el LLM devuelva intención y respuesta final
        mock_llm.chat = MagicMock(side_effect=[
            {"message": {"content": "test intent"}},  # Understand
            {"message": {"content": "done"}}          # Final response
        ])

        orchestrator = Orchestrator(
            llm_client=mock_llm,
            config={'agent': {'max_iterations': 2}}
        )
        
        state = SessionState()
        
        def tool_returns_string(*args, **kwargs):
            return "Resultado de la herramienta como string"

        tools_map = {'filesystem': tool_returns_string}

        # Mock del plan con campos válidos para validate_plan
        from core.planner import PlannerResponse, Step
        valid_plan = PlannerResponse(
            steps=[Step(id='1', description='test', tool='', args={}, depends_on=[], expected_output='')],
            estimated_tokens=10,
            max_iterations=2
        )
        
        with patch.object(orchestrator.planner, 'plan', return_value=valid_plan):
            response = await orchestrator.run("Tarea de prueba", state, tools_map)
        
        # Verifica que no hubo excepción y que se devolvió un Message
        assert response is not None
        assert hasattr(response, 'content')

    async def test_orchestrator_handles_dict_result(self):
        """El orchestrator debe aceptar un resultado dict de una herramienta."""
        from core.state import SessionState
        from core.orchestrator import Orchestrator
        
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(side_effect=[
            {"message": {"content": "test intent"}},
            {"message": {"content": "done"}}
        ])

        orchestrator = Orchestrator(
            llm_client=mock_llm,
            config={'agent': {'max_iterations': 2}}
        )
        
        state = SessionState()
        
        def tool_returns_dict(*args, **kwargs):
            return {"status": "success", "data": "info"}

        tools_map = {'filesystem': tool_returns_dict}

        # Mock del plan con campos válidos para validate_plan
        from core.planner import PlannerResponse, Step
        valid_plan = PlannerResponse(
            steps=[Step(id='2', description='test', tool='', args={}, depends_on=[], expected_output='')],
            estimated_tokens=10,
            max_iterations=2
        )

        with patch.object(orchestrator.planner, 'plan', return_value=valid_plan):
            response = await orchestrator.run("Tarea de prueba", state, tools_map)
        
        assert response is not None
        assert hasattr(response, 'content')


# --- Test C: LocalAgent.run() devuelve Message cuando Orchestrator lo proporciona ---
class TestLocalAgentReturnsMessage:
    """Verifica que LocalAgent.run() devuelva correctamente un Message."""


    async def test_agent_returns_message_when_orchestrator_succeeds(self):
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            from core.agent import LocalAgent
            from core.state import Message
            
            agent = LocalAgent(model_name='test')
            
            # Simular que el handler autónomo devuelve un Message válido
            mock_message = Message(role="assistant", content="Todo bien")
            
            # Mockeamos directamente el método interno que llama al orchestrator
            with patch.object(agent, '_handle_autonomous_mode', new=AsyncMock(return_value=mock_message)):
                result = await agent.run("Crear archivo")
            
            assert result is not None
            assert isinstance(result, Message)
            assert result.content == "Todo bien"



# --- Test D: LocalAgent.run() NO devuelve None cuando Orchestrator falla/devuelve None ---
class TestLocalAgentFallbackOnOrchestratorNone:
    """Verifica que LocalAgent.run() no devuelva None si el orchestrator falla."""

    async def test_agent_fallback_when_orchestrator_returns_none(self):
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            from core.agent import LocalAgent
            
            agent = LocalAgent(model_name='test')
            
            # Simular que el orchestrator devuelve None (error interno)
            agent.orchestrator.run = AsyncMock(return_value=None)
            
            result = await agent.run("Tarea que causa error")
            
            assert result is not None, "El agente NO debe devolver None"
            assert hasattr(result, 'content')
            assert len(result.content) > 0

    async def test_fallback_logged_on_orchestrator_none(self):
        """Verifica que el fallback quede registrado en logs."""
        with patch('core.agent.OllamaClient'), \
             patch('core.agent.ModelRegistry'):
            
            from core.agent import LocalAgent
            
            agent = LocalAgent(model_name='test')
            agent.orchestrator.run = AsyncMock(return_value=None)
            
            # Capturar logs
            import io
            log_stream = io.StringIO()
            handler = logging.StreamHandler(log_stream)
            handler.setLevel(logging.ERROR)
            logger = logging.getLogger('core.agent')
            logger.addHandler(handler)
            logger.setLevel(logging.ERROR)
            
            result = await agent.run("Tarea con error")
            
            # Verificar que se registró el error
            log_contents = log_stream.getvalue()
            assert 'Orchestrator devolvió None' in log_contents, \
                "El error debe quedar registrado en logs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])