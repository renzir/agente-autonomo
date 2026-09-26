"""
test_integration_flow_close.py — Prueba de cierre de flujo del agente (§15 - Integración Final)

Cubre:
1. Flujo completo: UNDERSTAND → PLAN → EXECUTE (con tools reales stubbed) → VERIFY → REFLECT → FINAL RESPONSE.
2. Verifica que la respuesta final se genera correctamente después de las herramientas.
3. Verifica que las métricas finales sean consistentes con el flujo completado.

Ejecutar con: pytest tests/test_integration_flow_close.py -v
"""

import json
import os
import sys
import time
from unittest.mock import MagicMock, patch, ANY
from datetime import datetime, timezone

# Asegurar path para imports relativos si se ejecuta standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from core.state import SessionState, Message
    from core.orchestrator import Orchestrator
    from core.planner import Planner, PlannerResponse, Step
    from metrics.logger import MetricsLogger, SimpleMetricsLogger

except ImportError:
    pass


class TestFlowCloseIntegration:
    """Pruebas de integración para el cierre del flujo del agente."""


    async def test_agent_generates_final_response_after_tool_execution(self):
        """Verifica que el agente genera una respuesta final tras ejecutar herramientas con éxito."""
        
        # 1. Configurar logger en memoria
        mem_logger = SimpleMetricsLogger(path=None)
        
        # 2. Mock del LLM para simular:
        #    - Llamada 1 (UNDERSTAND): Extrae la intención del usuario
        #    - Llamada 2 (_generate_response): Respuesta textual final
        mock_llm_client = MagicMock()
        
        call_count = {'value': 0}
        
        def side_effect_chat(messages, **kwargs):
            call_count['value'] += 1
            
            if call_count['value'] == 1:
                # Primera llamada (UNDERSTAND): devolver la intención extraída
                return {
                    "message": {
                        "role": "assistant",
                        "content": "Crear un archivo de texto con el contenido 'Hola Mundo'"
                    }
                }
            else:
                # Segunda llamada (_generate_response): respuesta final al usuario
                return {
                    "message": {
                        "role": "assistant", 
                        "content": "Archivo 'test_final.txt' creado con éxito. El contenido es: Hola Mundo."
                    }
                }

        mock_llm_client.chat = MagicMock(side_effect=side_effect_chat)

        # 3. Mock del Planner para devolver 1 paso en la iteración 1
        plan_call_count = {'value': 0}
        
        def stubbed_plan(self, intention, context=None):
            plan_call_count['value'] += 1
            
            if plan_call_count['value'] == 1:
                # Primer paso: crear archivo
                return PlannerResponse(
                    steps=[Step(
                        id=1,
                        description="Crear archivo de prueba",
                        depends_on=[],
                        tool="create_file", 
                        args={"path": "./test_final.txt", "content": "Hola Mundo"},
                        expected_output="Archivo creado"
                    )],
                    estimated_tokens=50,
                    max_iterations=2
                )
            else:
                # Segundo paso (final): no hay más tareas
                return PlannerResponse(
                    steps=[],
                    estimated_tokens=0,
                    max_iterations=0
                )

        # 4. Mock del Router para asegurar que enruta a 'create_file'
        def stubbed_router(intention, available_tools=None):
            from core.router import RouterDecision
            return RouterDecision(
                tool_name="create_file",
                confidence=1.0,
                reason="Intención clara de crear archivo"
            )

        # 5. Mock de create_file para simular éxito sin IO real
        mock_create_file_call_count = {'value': 0}

        def mock_create_file(args_dict, permission='default', risk_level='low'):
            mock_create_file_call_count['value'] += 1
            return {"status": "success", "result": "Archivo creado en ./test_final.txt"}

        with patch.object(Planner, 'plan', side_effect=stubbed_plan):
            with patch('core.router.Router.route', side_effect=stubbed_router):
                # Inyectar la herramienta mockeada en tools_map
                tools_map = {
                    "create_file": mock_create_file,
                    "read_file": lambda a: {"status": "ok", "content": ""},
                    "list_files": lambda a: {"status": "ok", "files": []}
                }

                # 6. Ejecutar el agente
                orchestrator = Orchestrator(
                    llm_client=mock_llm_client,
                    planner=Planner(llm_client=mock_llm_client),
                    config={'agent': {'max_iterations': 2, 'max_tool_calls': 5}},
                    metrics_logger=mem_logger
                )

                state = SessionState()
                
                # Mock de _reflect para forzar la terminación tras la segunda iteración (resp final)
                with patch.object(orchestrator, '_reflect', return_value=False):
                    response = await orchestrator.run("Crea un archivo con 'Hola Mundo'", state, tools_map)

        # --- Verificaciones ---
        
        # ✅ 1. El response no es None
        assert response is not None, "El agente debe generar una respuesta final."
        
        # ✅ 2. La respuesta tiene contenido textual (no es solo un tool call)
        assert hasattr(response, 'content'), "La respuesta final debe tener un campo 'content'."
        assert len(response.content) > 0, "La respuesta final debe tener texto descriptivo."
        
        # ✅ 3. El LLM fue llamado exactamente 2 veces (1 UNDERSTAND + 1 resp final)
        assert call_count['value'] == 2, f"Se esperaban 2 llamadas al LLM; se hicieron {call_count['value']}."
        
        # ✅ 4. La herramienta fue llamada una vez
        assert mock_create_file_call_count['value'] == 1, f"Se esperaba que create_file se llamara 1 vez; se llamó {mock_create_file_call_count['value']} veces."
        
        # ✅ 5. Las métricas finales son consistentes
        final_metrics_found = False
        for entry in mem_logger.metrics_history:
            if entry.get('metric') == 'orchestration_summary':
                val = entry.get('value', {})
                assert 'iterations_used' in val, "La métrica de resumen debe incluir iterations_used."
                assert 'total_tool_calls_executed' in val, "La métrica de resumen debe incluir total_tool_calls_executed."
                
                # Debería haber 1 iteración (o 2 dependiendo de cómo cuente el orchestrator) y 1 tool call
                assert val.get('iterations_used', 0) >= 1
                assert val.get('total_tool_calls_executed', 0) >= 1
                
                final_metrics_found = True
                break
        
        assert final_metrics_found, "Debería existir una métrica de resumen de orquestación al final del flujo."

    async def test_agent_handles_llm_failure_during_final_response(self):
        """Verifica que el agente maneja un fallo en la generación de la respuesta final."""
        
        mem_logger = SimpleMetricsLogger(path=None)
        
        call_count = {'value': 0}
        
        def side_effect_chat_failing(messages, **kwargs):
            call_count['value'] += 1
            
            if call_count['value'] == 1:
                # Primera llamada: solicitar herramienta
                return {
                    "message": {
                        "role": "assistant", 
                        "content": "Voy a crear un archivo.",  
                        "tool_calls": [
                            {
                                "id": "call_002", 
                                "type": "function", 
                                "function": {
                                    "name": "create_file", 
                                    "arguments": '{"path": "./test_fail.txt", "content": "Fail"}'
                                }
                            }
                        ]
                    }
                }
            else:
                # Segunda llamada (final): falla
                raise Exception("Ollama connection timeout during final response")

        mock_llm_client = MagicMock()
        mock_llm_client.chat = MagicMock(side_effect=side_effect_chat_failing)

        def stubbed_plan_fail(self, intention, context=None):
            return PlannerResponse(
                steps=[Step(id=1, description="Crear archivo", depends_on=[], tool="create_file", args={}, expected_output="Archivo")],
                estimated_tokens=50, max_iterations=2
            )

        def stubbed_router_fail(intention, available_tools=None):
            from core.router import RouterDecision
            return RouterDecision(tool_name="create_file", confidence=1.0, reason="Stub")

        def mock_create_file_fail(args_dict, permission='default', risk_level='low'):
            return {"status": "success", "result": "Archivo creado"}

        with patch.object(Planner, 'plan', side_effect=stubbed_plan_fail):
            with patch('core.router.Router.route', side_effect=stubbed_router_fail):
                tools_map = {"create_file": mock_create_file_fail}
                orchestrator = Orchestrator(
                    llm_client=mock_llm_client,
                    planner=Planner(llm_client=mock_llm_client),
                    config={'agent': {'max_iterations': 2, 'max_tool_calls': 5}},
                    metrics_logger=mem_logger
                )

                state = SessionState()
                
                # Mock de _reflect para forzar la terminación tras la segunda iteración
                with patch.object(orchestrator, '_reflect', return_value=False):
                    try:
                        response = await orchestrator.run("Crea un archivo con 'Fail'", state, tools_map)
                        # Si llega aquí, significa que el handler de excepciones generó una respuesta fallback
                        assert response is not None, "Debe haber una respuesta fallback incluso si falla la última llamada al LLM."
                    except Exception as e:
                        # Si el orchestrator no maneja bien la excepción del LLM final, esto puede fallar.
                        # En un diseño robusto, debe tener un try/except en la generación de respuesta final.
                        assert False, f"El orchestrator debería manejar excepciones del LLM final y generar un fallback. Error: {str(e)}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])