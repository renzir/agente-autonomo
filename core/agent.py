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
from tools.filesystem import read_file_tool, list_files_tool, create_file_tool, modify_file_tool
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
        
        # 2. Determinar model_name y base_url desde config si no se proporcionan
        resolved_model = model_name or self._resolve_model_name(models_config)
        resolved_base_url = base_url or self._resolve_endpoint(endpoint_config)
        
        self.logger.info(f"Modelo resuelto: {resolved_model}")
        self.logger.info(f"Endpoint resuelto: {resolved_base_url}")
        
        # 3. Inicializar OllamaClient con los valores configurados
        self.model_registry = ModelRegistry()
        self.ollama_client = OllamaClient(
            base_url=resolved_base_url,
            model_name=resolved_model,
            max_tokens=models_config.get('defaults', {}).get('max_tokens', 2048)
        )
        
        # 4. Configurar Sandbox desde config/agent.yaml
        safety_config = self.config.get('safety', {})
        if sandbox_enabled and safety_config.get('sandbox_enabled', True):
            allowed_dirs = safety_config.get('allowed_dirs', ['.'])
            command_timeout = safety_config.get('command_timeout', 30)
            
            from safety.sandbox import Sandbox
            self.sandbox = Sandbox(
                allowed_dirs=allowed_dirs,
                timeout=command_timeout
            )
            sandbox_config = self.sandbox.config
        else:
            self.sandbox = None
            sandbox_config = {"sandbox_enabled": False}
        
        # 5. Construir state (§12)
        self.state_factory = SessionState
        
        # 6. Unir orchestrator + router + planner (§12)
        self.planner = Planner(
            llm_client=self.ollama_client,
            config=self.config
        )
        
        # Inicializar catálogo central de herramientas
        self.tool_registry = ToolRegistry()

        # Inyectar el catálogo real en el Router
        self.router = Router(tool_registry=self.tool_registry)
        
        # Métricas (§14 wiring)
        from metrics.logger import MetricsLogger
        self.metrics_logger = MetricsLogger(path="metrics/run.jsonl")

        # Construir orchestrator con dependencias inyectadas (§12) + métricas (§14 wiring)
        self.orchestrator = Orchestrator(
            llm_client=self.ollama_client,
            planner=self.planner,
            router=self.router,
            config=self.config,
            metrics_logger=self.metrics_logger
        )
        
        self.current_session: Optional[SessionState] = None
        
        self.logger.info("Agente local inicializado correctamente con detección de modo estricta")


    async def run(self, user_input: str):
        print(f"[TRACE] RUN START: {user_input}")
        self.logger.info(f"Agente ejecutando: {user_input[:50]}...")

        mode = IntentClassifier.classify(user_input)

        print(f"[TRACE] MODE: {mode}")

        if mode == InteractionMode.CONVERSATIONAL:
            print("[TRACE] -> CONVERSATIONAL")
            return await self._handle_conversational_mode(user_input)

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
                'max_execution_time': 300
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


    async def _handle_conversational_mode(self, user_input: str):
        print("[TRACE] CONVERSATIONAL START")

        state = self.state_factory()
        state.add_message("user", user_input)

        system_prompt = (
            "Eres un asistente útil y conciso. Responde a las preguntas del usuario "
            "directamente sin usar herramientas externas ni realizar planes complejos."
        )

        if self.ollama_client:
            print("[TRACE] CONVERSATIONAL -> OLLAMA")

            return self.ollama_client.chat_stream(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.7,
                max_tokens=512,
            )

        return None
    
    async def _handle_autonomous_mode(self, user_input: str) -> Message:
        """
        Ejecuta el flujo completo actual: Planner -> Router -> Orchestrator.
        Se activa por defecto (sin prefijo) o si se especifica '/agente'.
        Preserva toda la lógica existente de herramientas y seguridad.
        """

        # Crear nuevo estado de sesión para esta ejecución
        state = self.state_factory()
        self.current_session = state

        # Construir herramientas disponibles
        tools_map = self._get_tools_map()

        response = await self.orchestrator.run(
            task=user_input,
            state=state,
            tools_map=tools_map,
            logger_callback=self.metrics_logger
        )

        # Validar que el Orchestrator haya devuelto una respuesta
        if response is None:
            self.logger.error(
                f"Orchestrator devolvió None para tarea: "
                f"{user_input[:50]}... Generando respuesta fallback."
            )

            return Message(
                role="assistant",
                content="[SISTEMA] No se pudo procesar la solicitud. Intente nuevamente más tarde."
            )

        # Devolver la respuesta generada por el Orchestrator
        return response

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
                return "Error: 'operation' argument is required. Valid operations: create_file, modify_file, list_files, read_file."
            
            dispatch_map = {
                'create_file': create_file_tool,
                'modify_file': modify_file_tool,
                'list_files': list_files_tool,
                'read_file': read_file_tool,
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