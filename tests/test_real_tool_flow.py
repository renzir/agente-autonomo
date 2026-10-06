"""
Prueba mínima del flujo real de herramientas.
Obliga a que LocalAgent use tools reales vía Orchestrator.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Asegurar que el root está en path
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


class TestRealToolFlow:
    """Prueba mínima del flujo completo con herramientas reales."""

    async def test_filesystem_tool_executed_via_orchestrator(self):
        """Verifica que una tool real (filesystem) se ejecuta correctamente vía Orchestrator."""
        
        from core.agent import LocalAgent
        from core.state import SessionState
        from tools.registry import ToolRegistry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear un sandbox fake que acepte paths en temp_dir
            class FakeSandbox:
                allowed_dirs = [temp_dir]
                
                def validate_path(self, p): 
                    return True
            
            # Patch the _get_sandbox function to return our fake sandbox
            with patch('tools.filesystem._get_sandbox', return_value=FakeSandbox()):
                with patch('tools.shell._get_sandbox', return_value=FakeSandbox()):
                    with patch('tools.search._get_sandbox', return_value=FakeSandbox()):
                        
                        # Crear agente con mocks
                        with patch('core.agent.OllamaClient'), \
                             patch('core.agent.ModelRegistry'):
                            
                            agent = LocalAgent(model_name='test_model')
                            
                            # Mock del LLM para que devuelva una intención de crear archivo
                            async def mock_llm_chat(*args, **kwargs):
                                return {
                                    "message": {
                                        "role": "assistant",
                                        "content": "create_file test.txt con contenido de prueba"
                                    }
                                }
                            
                            agent.ollama_client.chat = AsyncMock(side_effect=mock_llm_chat)
                            
                            # Mock del planner para que devuelva un paso concreto
                            from core.planner import PlannerResponse, Step
                            
                            async def mock_plan(intention, context=None):
                                return PlannerResponse(
                                    steps=[Step(
                                        id=1, 
                                        description="Crear archivo de prueba", 
                                        tool="filesystem", 
                                        args={'operation': 'create_file', 'path': os.path.join(temp_dir, 'test.txt'), 'content': 'Hola desde tool real'}
                                    )],
                                    estimated_tokens=50,
                                    max_iterations=1
                                )
                            
                            agent.planner.plan = mock_plan
                            
                            # Mock del router para que siempre seleccione filesystem
                            from core.router import RouterDecision
                            
                            def always_route_filesystem(intention, available_tools=None):
                                return RouterDecision(
                                    tool_name='filesystem',
                                    confidence=0.95,
                                    reason="Forced filesystem for real tool test"
                                )
                            
                            agent.router.route = always_route_filesystem
                            
                            # Mock de _reflect para terminar tras una iteración
                            with patch.object(agent.orchestrator, '_reflect', return_value=False):
                                response = await agent.run("/agente crear un archivo de prueba")
                            
                            # Verificar que se obtuvo respuesta
                            assert response is not None, "La respuesta no debería ser None"
                            
                            # Verificar que el archivo fue creado realmente (si la tool funcionó)
                            test_file = os.path.join(temp_dir, 'test.txt')
                            assert os.path.exists(test_file), f"El archivo {test_file} debería existir si create_file_tool funcionó"
                            
                            if os.path.exists(test_file):
                                with open(test_file, 'r') as f:
                                    content = f.read()
                                assert 'Hola desde tool real' in content, "El contenido del archivo debería coincidir"


if __name__ == "__main__":
    asyncio.run(TestRealToolFlow().test_filesystem_tool_executed_via_orchestrator())
    print("✅ Prueba de flujo real PASSED")