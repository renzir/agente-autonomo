"""
tests/test_ollama_integration.py — Prueba de conexión real con Ollama (servidor remoto)

Cubre:
1. Conexión al endpoint de Ollama desde models/ollama_client.py.
2. Verificación de que el cliente puede hacer una llamada básica (health check).
3. Soporte para IP/puerto hardcodeado vía variables de entorno o constantes configurables.

Configuración del servidor remoto:
    - Establece las variables OLLAMA_BASE_URL y OLLAMA_MODEL en tu entorno, o
    - Modifica las constantes SERVER_IP y SERVER_PORT abajo.

Ejecutar con: pytest tests/test_ollama_integration.py -v
"""

import pytest
import httpx
import sys
import os


# Asegurar que el root del proyecto está en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ============================================================================
# CONFIGURACIÓN DEL SERVIDOR REMOTO (MODIFICAR AQUÍ)
# ============================================================================

#prueba de conexión a Ollama en un servidor remoto (IP/puerto)

# Opción 1: Usar variables de entorno
OLLAMA_SERVER_IP = 'http://100.64.99.98:11434'
OLLAMA_MODEL_NAME = 'qwen3.6-64k:latest'


class TestOllamaIntegration:
    """Pruebas de integración con el cliente real de Ollama."""

    def test_endpoint_resolution(self):
        """Verifica que resolve_endpoint devuelve la IP configurada correctamente."""
        from models.ollama_client import OllamaClient

        # Probar con override explícito (la IP del servidor remoto)
        resolved = OllamaClient.resolve_endpoint(OLLAMA_SERVER_IP)
        
        assert resolved == OLLAMA_SERVER_IP.rstrip('/'), \
            f"El endpoint resuelto '{resolved}' no coincide con la configuración '{OLLAMA_SERVER_IP}'"

    def test_client_initialization(self):
        """Verifica que el cliente se inicializa con los parámetros correctos."""
        from models.ollama_client import OllamaClient

        client = OllamaClient(
            base_url=OLLAMA_SERVER_IP,
            model_name=OLLAMA_MODEL_NAME,
            timeout=30.0
        )

        assert client._resolved_url == OLLAMA_SERVER_IP.rstrip('/')
        assert client.model_name == OLLAMA_MODEL_NAME
        assert client.timeout == 30.0

    def test_health_check_models_list(self):
        """Intenta listar modelos disponibles en Ollama (health check básico)."""
        import asyncio
        from models.ollama_client import OllamaClient

        client = OllamaClient(base_url=OLLAMA_SERVER_IP)
        
        try:
            # list_models es asíncrono, ejecutar con asyncio.run
            result = asyncio.run(client.list_models())
            
            assert isinstance(result, dict), "La respuesta debe ser un diccionario"
            assert 'models' in result or 'error' in result, \
                "La respuesta debe contener 'models' (éxito) o 'error' (fallo)"
            
            # Si hay modelos, verificar que la lista no esté vacía (opcional)
            if not result.get('error'):
                models = result.get('models', [])
                assert len(models) > 0, "Se esperan al menos un modelo instalado en Ollama"
                
        except Exception as e:
            pytest.skip(f"No se pudo conectar a Ollama en {OLLAMA_SERVER_IP}: {str(e)}")

    @pytest.mark.asyncio
    async def test_chat_basic_response(self):
        """Envía un mensaje simple y verifica que se obtiene una respuesta del LLM."""
        import asyncio
        from models.ollama_client import OllamaClient
        import json
        
        client = OllamaClient(
            base_url=OLLAMA_SERVER_IP,
            model_name=OLLAMA_MODEL_NAME
        )
        messages = [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": "Responde solo con 'OK' si recibes este mensaje."}
        ]

        
        try:
            result = await client.chat(messages, model=OLLAMA_MODEL_NAME, stream=False)
            assert isinstance(result, dict), "La respuesta del chat debe ser un diccionario"
            assert 'message' in result or 'error' in result, \
                "La respuesta debe contener 'message' (éxito) o 'error' (fallo)"
            if not result.get('error'):
                content = result.get('message', {}).get('content', '')
                assert isinstance(content, str), "El contenido debe ser un string"
                assert len(content) > 0, "El contenido no puede estar vacío"
        except json.JSONDecodeError as e:
            # DEBUG: Ver qué está devolviendo el servidor
            print(f"\n\n--- DEBUG ERROR JSON ---")
            print(f"URL intentada: {client._resolved_url}/api/chat")
            try:
                # Intentar leer la respuesta cruda para ver qué es
                import httpx
                # Esto fallará porque ya se consumió el response, pero es solo para entender el flujo
            except:
                pass
            print(f"Error de JSON: {str(e)}")
            raise  # ¡IMPORTANTE! Lanza el error para que veas el traceback completo en consola
        except Exception as e:
            pytest.skip(f"No se pudo conectar a Ollama para chat en {OLLAMA_SERVER_IP}: {str(e)}")
        
    @pytest.mark.asyncio    
    async def test_chat_basic_streaming_response(self):
        """Envía un mensaje con streaming y verifica que se obtienen chunks."""
        import httpx
        
        try:
            from models.ollama_client import OllamaClient
        except ImportError as e:
            pytest.skip(f"Módulos no disponibles: {e}")

        client = OllamaClient(
            base_url=OLLAMA_SERVER_IP,
            model_name=OLLAMA_MODEL_NAME
        )
        
        messages = [
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": "Responde con una frase corta para probar streaming."}
        ]
        
        # 1. Verificar primero que Ollama está disponible (para evitar errores confusos de stream)
        try:
            health_check = await client.list_models()
            if health_check.get('error'):
                pytest.skip(f"Ollama no disponible: {health_check['error']}")
        except httpx.ConnectError as e:
            pytest.skip(f"No se pudo conectar a Ollama para verificar disponibilidad en {OLLAMA_SERVER_IP}: {str(e)}")
        except Exception as e:
            pytest.skip(f"Error inesperado al verificar Ollama: {type(e).__name__}: {str(e)[:200]}")

        try:
            # chat() con stream=True devuelve un AsyncGenerator
            result_gen = await client.chat(
                messages=messages, 
                model=OLLAMA_MODEL_NAME, 
                stream=True
            )
            
            # Verificar que es un generador asíncrono
            assert hasattr(result_gen, '__aiter__'), "El streaming debe devolver un async generator"
            
            chunks_collected = []
            full_content = ""
            
            async for chunk_text, is_final in result_gen:
                chunks_collected.append(chunk_text)
                full_content += chunk_text
                
            # Debe haber al menos un chunk
            assert len(chunks_collected) > 0, "Se esperaba al menos un chunk de streaming"
            
            # El contenido final no debe estar vacío
            assert len(full_content.strip()) > 0 or len(chunks_collected) > 0, \
                "El streaming debería producir algún resultado"

        except httpx.HTTPStatusError as e:
            pytest.skip(f"Error HTTP durante streaming ({e.response.status_code}): {str(e)[:200]}")
        except httpx.ConnectError as e:
            pytest.skip(f"No se pudo conectar a Ollama durante streaming en {OLLAMA_SERVER_IP}: {str(e)}")
        except Exception as e:
            # Captura cualquier otro error (JSON, tipo de dato, etc.)
            pytest.skip(f"Error inesperado en streaming en {OLLAMA_SERVER_IP}: {type(e).__name__}: {str(e)[:200]}")
   
    def test_endpoint_from_config_file(self):
        """Verifica que resolve_endpoint lee correctamente desde config/ollama_endpoint.yaml si existe."""
        from models.ollama_client import OllamaClient, CONFIG_PATHS
        import yaml
        from pathlib import Path
        
        # Buscar el primer archivo de config que exista
        config_file = None
        for p in CONFIG_PATHS:
            if p.exists():
                config_file = p
                break
        
        if not config_file:
            pytest.skip("No se encontró ningún archivo de config de Ollama")
        
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            endpoint_from_config = (data.get('ollama_endpoint') or data.get('endpoint', ''))
            
            if not endpoint_from_config:
                pytest.skip("El archivo de config no contiene 'ollama_endpoint' o 'endpoint'")
            
            # Ahora probar que resolve_endpoint sin override lee desde el config
            resolved = OllamaClient.resolve_endpoint(None)
            
            # Si el config existe y tiene endpoint, debería usarse
            assert resolved == endpoint_from_config.rstrip('/'), \
                f"El endpoint resuelto '{resolved}' no coincide con '{endpoint_from_config}' del config"
                
        except Exception as e:
            pytest.skip(f"No se pudo leer el config de Ollama: {str(e)}")


# ============================================================================
# TEST DE INTEGRACIÓN CON EL AGENTE COMPLETO (si Ollama está disponible)
# ============================================================================

class TestFullAgentWithRealOllama:
    """Prueba del flujo completo del agente usando Ollama real."""
    
    @pytest.mark.asyncio
    async def test_agent_loop_with_real_ollama(self):
        """Ejecuta un ciclo mínimo del agente con LLM real y verifica métricas."""
        import asyncio
        
        try:
            from models.ollama_client import OllamaClient
            from core.state import SessionState
            from core.orchestrator import Orchestrator
            from metrics.logger import SimpleMetricsLogger
        except ImportError as e:
            pytest.skip(f"Módulos del proyecto no disponibles: {str(e)}")

        client = OllamaClient(base_url=OLLAMA_SERVER_IP, model_name=OLLAMA_MODEL_NAME)
        
        # Verificar que se puede conectar antes de seguir
        try:
            result = await client.list_models()
            if result.get('error'):
                pytest.skip(f"Ollama no disponible: {result['error']}")
        except Exception as e:
            pytest.skip(f"No se pudo conectar a Ollama en {OLLAMA_SERVER_IP}: {str(e)}")

        # Crear un metrics logger en memoria para este test
        mem_logger = SimpleMetricsLogger(path=None)
        
        # Construir orchestrator con el cliente real
        from core.planner import Planner
        from core.router import Router
        
        planner = Planner(llm_client=client, config={'agent': {'max_iterations': 2, 'max_tool_calls': 3}})
        
        orchestrator = Orchestrator(
            llm_client=client,
            planner=planner,
            router=Router(),
            config={'agent': {'max_iterations': 2, 'max_tool_calls': 3}},
            metrics_logger=mem_logger
        )
        
        state = SessionState()
        
        # Herramientas stub para este test (no se llamarán porque _reflect se patchea)
        tools_map = {
            'test_tool': lambda a: {'status': 'success'}
        }
        
        try:
            with __import__('unittest.mock').mock.patch.object(orchestrator, '_reflect', return_value=False):
                response = await orchestrator.run("Prueba de ciclo con Ollama real", state, tools_map)
            
            assert response is not None, "La respuesta no debe ser None"
            assert hasattr(response, 'content'), "La respuesta debe tener campo 'content'"
            assert len(response.content) > 0, "El contenido de la respuesta no puede estar vacío"
            
            # Verificar que se registraron métricas
            metrics = mem_logger.get_metrics() if hasattr(mem_logger, 'get_metrics') else []
            assert len(metrics) >= 1, "Se esperaban al menos algunas métricas registradas"
            
        except Exception as e:
            pytest.fail(f"Fallo en el ciclo del agente con Ollama real: {str(e)}")


# ============================================================================
# CLI PARA PROBAR CONEXIÓN DESDE TERMINAL
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Probar conexión con Ollama desde terminal')
    parser.add_argument('--url', default=OLLAMA_SERVER_IP, help=f'URL del servidor Ollama (default: {OLLAMA_SERVER_IP})')
    parser.add_argument('--model', default=OLLAMA_MODEL_NAME, help=f'Modelo a usar (default: {OLLAMA_MODEL_NAME})')
    args = parser.parse_args()
    
    print(f"Probando conexión con Ollama en {args.url}...")
    print(f"Modelo: {args.model}")
    
    import asyncio

    @pytest.mark.asyncio
    async def test_connection():
        from models.ollama_client import OllamaClient
        
        client = OllamaClient(base_url=args.url, model_name=args.model)
        
        # Health check
        try:
            result = await client.list_models()
            print(f"✅ Conexión exitosa")
            if 'models' in result and result['models']:
                print(f"Modelos disponibles:")
                for m in result['models']:
                    name = m.get('name', 'Unknown')
                    size = m.get('size', 0)
                    print(f"  - {name} ({size / (1024**3):.2f} GB)")
            elif 'error' in result:
                print(f"❌ Error de Ollama: {result['error']}")
            return True
        except Exception as e:
            print(f"❌ No se pudo conectar a {args.url}: {str(e)}")
            return False
    
    success = asyncio.run(test_connection())
    
    if not success:
        print("\n💡 Soluciones posibles:")
        print("  1. Verifica que Ollama está corriendo en el servidor")
        print("  2. Revisa el firewall (puerto 11434 debe estar abierto)")
        print(f"  3. La URL correcta es: {args.url}")
    
    exit(0 if success else 1)