"""
core/agent.py - Punto de entrada del agente (§12)
Conecta config, models, core modules y expone la API pública.
Filosofía: NO hardcode nada; usa las capas inyectadas.
Modificado: Clasificación estricta de modo (/chat vs /agente vs default).
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from tools.registry import ToolRegistry

# Agregar el directorio raíz al path para imports relativos
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from models.ollama_client import OllamaClient
    from models.model_registry import ModelRegistry
    from models.schemas import AgentConfig, ToolMetadata, InteractionMode
except ImportError as e:
    logging.error(f"Error importing models: {e}")
    sys.exit(1)

# Core imports
from core.state import SessionState, Message
from core.planner import Planner
from core.router import Router 
# Importamos el clasificador aquí para usarlo en la lógica de enrutamiento estricta
from core.router import IntentClassifier 
from core.orchestrator import Orchestrator

# Herramientas reales (no implementadas aún native tool calling, pero disponibles)
from tools.filesystem import read_file_tool, list_files_tool, create_file_tool, modify_file_tool, project_tree_tool
from tools.search import search_text_tool
from tools.shell import execute_command_tool


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import yaml

class LocalAgent:
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        sandbox_enabled: bool = True
    ):
        self.logger = logging.getLogger(__name__)

        # 1. Cargar configuración desde archivos YAML en config/
        self.config = self._load_agent_config(config_path)
        models_config = self._load_models_config()
        endpoint_config = self._load_endpoint_config()

        # 2. Resolver modelo y endpoint
        resolved_model = model_name or self._resolve_model_name(models_config)
        resolved_base_url = base_url or self._resolve_endpoint(endpoint_config)

        self.logger.info(f"Modelo resuelto: {resolved_model}")
        self.logger.info(f"Endpoint resuelto: {resolved_base_url}")

        # 3. Inicializar cliente y proveedor LLM
        self.model_registry = ModelRegistry()

        ollama_client = OllamaClient(
            base_url=resolved_base_url,
            model_name=resolved_model,
            max_tokens=models_config.get("defaults", {}).get("max_tokens", 2048)
        )

        from models.providers.ollama_provider import OllamaProvider
        self.llm_provider = OllamaProvider(client=ollama_client)

        # Mantener la referencia al cliente para compatibilidad con código existente
        self.ollama_client = ollama_client

        # 4. Configurar Sandbox desde config/agent.yaml
        safety_config = self.config.get("safety", {})

        if sandbox_enabled and safety_config.get("sandbox_enabled", True):
            allowed_dirs = safety_config.get("allowed_dirs", ["."])
            command_timeout = safety_config.get("command_timeout", 30)

            from safety.sandbox import Sandbox
            self.sandbox = Sandbox(
                allowed_dirs=allowed_dirs,
                timeout=command_timeout
            )
            sandbox_config = self.sandbox.config
        else:
            self.sandbox = None
            sandbox_config = {"sandbox_enabled": False}

        # 5. Configurar el estado de sesión
        self.state_factory = SessionState

        # 6. Inicializar Planner con el proveedor LLM
        self.planner = Planner(
            llm_client=self.llm_provider,
            config=self.config
        )

        # 7. Catálogo central de herramientas y Router
        self.tool_registry = ToolRegistry()
        self.router = Router(tool_registry=self.tool_registry)

        # 8. Configurar métricas
        from metrics.logger import MetricsLogger
        self.metrics_logger = MetricsLogger(path="metrics/run.jsonl")

        # 9. Inicializar Orchestrator con dependencias inyectadas
        self.orchestrator = Orchestrator(
            llm_client=self.llm_provider,
            planner=self.planner,
            router=self.router,
            config=self.config,
            metrics_logger=self.metrics_logger
        )

        self.current_session: Optional[SessionState] = None

        self.logger.info(
            "Agente inicializado correctamente con proveedor LLM desacoplado"
        )

    async def run(self, user_input: str):
        print(f"[TRACE] RUN START: {user_input}")
        self.logger.info(f"Agente ejecutando: {user_input[:50]}...")

        command = user_input.strip().split(maxsplit=1)[0].lower()

        if command == "/agente":
            mode = InteractionMode.AUTONOMOUS
        elif command == "/analiza":
            mode = InteractionMode.ANALYSIS
        else:
            mode = InteractionMode.CONVERSATIONAL

        print(f"[TRACE] MODE: {mode}")

        if mode == InteractionMode.CONVERSATIONAL:
            print("[TRACE] -> CONVERSATIONAL")
            return await self._handle_conversational_mode(user_input)

        if mode == InteractionMode.ANALYSIS:
            print("[TRACE] -> ANALYSIS")
            return await self._handle_analysis_mode(user_input)

        print("[TRACE] -> AUTONOMOUS")
        return await self._handle_autonomous_mode(user_input)


    def _load_agent_config(self, config_path: Optional[str]) -> dict:
        """Carga configuración desde agent.yaml o usa defaults."""
        if config_path is None:
            config_path = "config/agent.yaml"
        
        default_config = {
            'agent': {
                'max_iterations': 10,
                'max_tool_calls': 20,
                'max_execution_time': 900
            },
            'context': {
                'hard_limit': 32768,
                'target_input': 14000
            },
            'safety': {
                'sandbox_enabled': True,
                'allowed_dirs': ['.'],
                'command_timeout': 30
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    for key in yaml_config:
                        default_config[key] = yaml_config[key]
                self.logger.info(f"Configuración de agente cargada desde {config_path}")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar config/agent.yaml: {e}, usando defaults")
        
        return default_config

    def _load_models_config(self) -> dict:
        """Carga configuración de modelos desde models.yaml."""
        models_path = "config/models.yaml"
        default_config = {
            'models': {'primary': 'qwen3.6:latest'},
            'defaults': {
                'model': 'primary',
                'temperature': 0.7,
                'max_tokens': 2048
            }
        }
        
        if os.path.exists(models_path):
            try:
                with open(models_path, 'r') as f:
                    return yaml.safe_load(f) or default_config
            except Exception as e:
                self.logger.warning(f"No se pudo cargar models.yaml: {e}")
        
        return default_config

    def _load_endpoint_config(self) -> dict:
        """Carga configuración del endpoint desde ollama_endpoint.yaml."""
        endpoint_path = "config/ollama_endpoint.yaml"
        default_config = {'endpoint': 'http://localhost:11434'}
        
        if os.path.exists(endpoint_path):
            try:
                with open(endpoint_path, 'r') as f:
                    loaded = yaml.safe_load(f)
                    return loaded or default_config
            except Exception as e:
                self.logger.warning(f"No se pudo cargar ollama_endpoint.yaml: {e}")
        
        return default_config

    def _resolve_model_name(self, models_config: dict) -> str:
        """Resuelve el nombre del modelo usando la config de models.yaml."""
        defaults = models_config.get('defaults', {})
        model_key = defaults.get('model', 'primary')
        models = models_config.get('models', {})
        
        # Si model_key existe en models, usar ese; sino fallback a default
        return models.get(model_key, 'qwen3.6:latest')

    def _resolve_endpoint(self, endpoint_config: dict) -> str:
        """Resuelve el endpoint de Ollama."""
        return endpoint_config.get('endpoint', 'http://localhost:11434')

# LOCALIZACIÓN: core/agent.py -> Método _handle_conversational_mode

    async def _handle_conversational_mode(self, user_input: str):
        """Maneja el modo conversacional reutilizando la sesión y manteniendo el streaming."""
        print("[TRACE] CONVERSATIONAL START")

        # 1. Crear una sesión solo si no existe una activa.
        if self.current_session is None:
            self.current_session = self.state_factory()
            self.logger.info("Nueva sesión conversacional creada")

        state = self.current_session

        # 2. Añadir el mensaje del usuario una sola vez por petición.
        state.add_message("user", user_input)

        system_prompt = (
            "Eres un asistente útil y conciso. Responde a las preguntas del usuario "
            "directamente sin usar herramientas externas ni realizar planes complejos."
        )

        # 3. Construir el historial conversacional.
        history_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in state.messages
            if msg.role in ("user", "assistant")
        ]

        messages_for_llm = [
            {"role": "system", "content": system_prompt},
            *history_messages,
        ]

        self.logger.info(
            f"Enviando {len(messages_for_llm)} mensajes al LLM"
        )

        # CAMBIO AQUÍ: Verificar llm_provider en lugar de ollama_client
        if not self.llm_provider:
            self.logger.warning("LLMProvider no disponible en modo conversacional")
            return Message(
                role="assistant",
                content="[SISTEMA] No se pudo conectar con el modelo de lenguaje."
            )

        print("[TRACE] CONVERSATIONAL -> LLM PROVIDER (streaming)")

        try:
            # CAMBIO AQUÍ: Usar self.llm_provider en lugar de self.ollama_client
            response_generator = await self.llm_provider.chat(
                messages=messages_for_llm,
                stream=True,
                temperature=0.7,
            )

            async def stream_and_save():
                full_response_content = ""

                try:
                    # El contrato de LLMProvider para stream=True es AsyncGenerator[tuple[str, bool]]
                    async for chunk_text, is_final in response_generator:
                        if chunk_text:
                            full_response_content += chunk_text

                        # Devolver cada fragmento inmediatamente al consumidor.
                        yield chunk_text, is_final

                    # Guardar la respuesta una sola vez al terminar el streaming.
                    if full_response_content.strip():
                        state.add_message("assistant", full_response_content)
                        self.logger.info(
                            "Respuesta del asistente guardada en el historial"
                        )

                except Exception:
                    self.logger.exception("Error durante el streaming del LLM")
                    raise

            return stream_and_save()

        except Exception as e:
            self.logger.error(f"Error al comunicarse con el LLM: {e}")
            return Message(
                role="assistant",
                content=(
                    "[SISTEMA] Error de comunicación con el modelo de lenguaje. "
                    "Intente nuevamente más tarde. "
                    f"Detalle: {str(e)[:100]}"
                )
            )
            
    async def _handle_autonomous_mode(
        self,
        user_input: str,
        mode: InteractionMode = InteractionMode.AUTONOMOUS
    ):
        """
        Ejecuta una tarea conservando el historial compartido
        y transmite la respuesta final progresivamente.
        """
        import asyncio

        if self.current_session is None:
            self.current_session = self.state_factory()
            self.logger.info("Nueva sesión compartida creada")

        shared_state = self.current_session

        shared_state.add_message(
            "user",
            f"[{mode.value}] {user_input}"
        )

        execution_state = self.state_factory()
        execution_state.messages = list(shared_state.messages)
        execution_state.session_id = shared_state.session_id

        tools_map = self._get_tools_map()

        print(
            f"[TRACE] {mode.value.upper()} usando historial compartido: "
            f"{len(execution_state.messages)} mensajes"
        )

        stream_queue = asyncio.Queue()
        stream_finished = object()

        def stream_callback(chunk, is_final):
            stream_queue.put_nowait((chunk, is_final))

        async def execute_orchestrator():
            try:
                return await self.orchestrator.run(
                    task=user_input,
                    state=execution_state,
                    tools_map=tools_map,
                    logger_callback=self.metrics_logger,
                    stream_callback=stream_callback,
                    mode=mode.value
                )
            finally:
                stream_queue.put_nowait(stream_finished)

        async def stream_response():
            task = asyncio.create_task(execute_orchestrator())
            streamed_content = ""

            try:
                while True:
                    item = await stream_queue.get()

                    if item is stream_finished:
                        break

                    chunk, is_final = item

                    if chunk:
                        streamed_content += chunk

                    yield chunk, is_final

                response = await task

                if response is None:
                    self.logger.error(
                        f"Orchestrator devolvió None para tarea: "
                        f"{user_input[:50]}..."
                    )
                    response = Message(
                        role="assistant",
                        content="[SISTEMA] No se pudo procesar la solicitud."
                    )

                # Si el orquestador no produjo fragmentos, conservar su respuesta.
                if not streamed_content and response.content:
                    yield response.content, True

                if response.content:
                    shared_state.add_message(
                        "assistant",
                        f"[Resultado {mode.value}] {response.content}"
                    )

            except Exception:
                self.logger.exception(
                    "Error durante la ejecución autónoma con streaming"
                )
                raise

            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        return stream_response()

    async def _handle_analysis_mode(self, user_input: str) -> Message:
        """
        Ejecuta el flujo explícito de análisis.
        """

        analysis_input = user_input.strip()

        if analysis_input.lower().startswith("/analiza"):
            analysis_input = analysis_input[len("/analiza"):].strip()

        print(f"[TRACE] ANALYSIS INPUT: {analysis_input}")

        return await self._handle_autonomous_mode(
            analysis_input,
            mode=InteractionMode.ANALYSIS
        )

    def _get_tools_map(self) -> Dict[str, Callable]:
        """
        Construye el mapa de herramientas para el orchestrator.
        
        Importa las funciones reales desde tools/ y las expone al orchestrator.
        
        Returns:
            dict[str, Callable] - Mapeo nombre_herramienta -> función ejecutable
        """
        # Mapeo de nombres lógicos (usados por el orchestrator/router) a funciones reales
        
        def filesystem_wrapper(args: dict, permission: str = "read", risk_level: str = "low") -> Any:
            operation = args.get('operation')
            if not operation:
                return "Error: 'operation' argument is required. Valid operations: create_file, modify_file, list_files, read_file, project_tree."
            
            dispatch_map = {
                'create_file': create_file_tool,
                'modify_file': modify_file_tool,
                'list_files': list_files_tool,
                'read_file': read_file_tool,
                'project_tree': project_tree_tool,
            }
            
            target_fn = dispatch_map.get(operation)
            if not target_fn:
                return f"Error: Unknown filesystem operation '{operation}'. Valid operations: {', '.join(dispatch_map.keys())}"
            
            clean_args = {k: v for k, v in args.items() if k != 'operation'}
            return target_fn(clean_args, permission, risk_level)
        
        
        def search_wrapper(args: dict, permission: str = "read", risk_level: str = "low") -> Any:
            """Wrapper para búsqueda que delega en search_text_tool."""
            if 'query' in args and 'pattern' not in args:
                args['pattern'] = args.pop('query')
            return search_text_tool(args, permission, risk_level)
        
        def shell_wrapper(args: dict, permission: str = "execute", risk_level: str = "high") -> Any:
            """Wrapper para shell que delega en execute_command_tool."""
            return execute_command_tool(args, permission, risk_level)
        
        def git_stub(args: dict, permission: str = "execute", risk_level: str = "medium") -> Any:
            """Stub para git hasta que se implementen las operaciones."""
            action = args.get('action', 'status')
            logger.info(f"[TOOLS] git (stub): {args}")
            return {"status": "success", "operation": "git_stub", "action": action}
        
        return {
            'filesystem': filesystem_wrapper,
            'search': search_wrapper,
            'shell': shell_wrapper,
            'git': git_stub
        }


    def get_metrics(self) -> List[Dict[str, Any]]:
        """Obtiene las métricas acumuladas de la ejecución actual."""
        return self.metrics_logger.metrics_history
    
    def reset_session(self):
        """Resetea el estado actual para nueva sesión."""
        self.current_session = None
        self.metrics_logger.metrics_history.clear()
        self.logger.info("Sesión reiniciada")
        
# API pública simplificada
def create_agent(
    config_path: str = "config/agent.yaml",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    sandbox_enabled: bool = True
) -> LocalAgent:
    """
    Fábrica para crear un agente con configuración desde config/.
    
    Args:
        config_path: Ruta al archivo agent.yaml
        model_name: Modelo override (None = usar models.yaml)
        base_url: Endpoint override (None = usar ollama_endpoint.yaml)
        sandbox_enabled: Activar sandbox basado en agent.yaml
    """
    return LocalAgent(
        config_path=config_path,
        model_name=model_name,
        base_url=base_url
    )


if __name__ == "__main__":
    # Ejemplo de uso standalone
    print("Inicializando agente local...")
    
    agent = create_agent(model_name="qwen3.6:latest")
    
    # Prueba con input simple
    # Nota: En un entorno real, esto debería ser async y usado con asyncio.run()
    import asyncio
    response = asyncio.run(agent.run("¿Qué herramientas tienes disponibles?"))
    
    print(f"\nRespuesta del agente: {response.content}")
    print(f"Métricas: {len(agent.get_metrics())} registros")