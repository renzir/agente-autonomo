"""
core/agent.py - Punto de entrada del agente (§12)
Conecta config, models, core modules y expone la API pública.
Filosofía: NO hardcode nada; usa las capas inyectadas.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Agregar el directorio raíz al path para imports relativos
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from models.ollama_client import OllamaClient
    from models.model_registry import ModelRegistry
    from models.schemas import AgentConfig, ToolMetadata
except ImportError as e:
    logging.error(f"Error importing models: {e}")
    sys.exit(1)

# Core imports
from core.state import SessionState, Message
from core.planner import Planner
from core.router import Router
from core.orchestrator import Orchestrator


# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Interfaz mínima para métricas (§12).
    Implementa log(metric_name, value_or_dict)
    
    Este logger puede ser reemplazado por una implementación real en producción.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def log(self, metric_name: str, value_or_dict: Any):
        """
        Registra una métrica.
        
        Args:
            metric_name: Nombre de la métrica (ej: 'tokens_used', 'verification_pass')
            value_or_dict: Valor numérico o diccionario con detalles
        """
        self.metrics_history.append({
            'timestamp': __import__('time').time(),
            'metric': metric_name,
            'value': value_or_dict
        })
        logger.debug(f"Métrica registrada: {metric_name} = {value_or_dict}")


class LocalAgent:
    """
    Agente local completo que conecta todas las capas.
    
    Arquitectura:
    - Config/Models: Carga y configuración desde models/ y config/
    - State: Gestión de estado de sesión (core/state.py)
    - Planner: División de tareas (core/planner.py)
    - Router: Selección de herramientas (core/router.py)
    - Orchestrator: Ciclo principal (core/orchestrator.py)
    
    NO hardcode nada; usa las capas inyectadas.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        model_name: str = "qwen3.6:latest",  # Modelo por defecto
        base_url: str = "http://localhost:11434"
    ):
        self.logger = logging.getLogger(__name__)
        
        # 1. Carga config y modelos desde models/ (§12)
        self.config = self._load_config(config_path)
        
        self.model_registry = ModelRegistry()
        self.ollama_client = OllamaClient(
            base_url=base_url,
            model_name=model_name,
            max_tokens=self.config.get('agent', {}).get('max_tokens', 4096)
        )
        
        # 2. Construye state (§12)
        self.state_factory = SessionState
        
        # 3. Une orchestrator + router + planner (§12)
        self.planner = Planner(
            llm_client=self.ollama_client,
            config=self.config
        )
        
        # Cargar herramientas disponibles desde registry si existe
        available_tools = self._get_tool_metadata()
        self.router = Router(tool_registry=available_tools)
        
        # Inicializar métricas logger inyectado (§12)
        self.metrics_logger = MetricsLogger()
        
        # Construir orchestrator con dependencias inyectadas (§12)
        # Usa core/planner, core/router
        self.orchestrator = Orchestrator(
            llm_client=self.ollama_client,
            planner=self.planner,
            router=self.router,
            config=self.config
        )
        
        # Estado actual de sesión (se crea por run)
        self.current_session: Optional[SessionState] = None
        
        self.logger.info("Agente local inicializado correctamente")
        self.logger.debug(f"Configuración cargada: {self.config}")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Carga configuración desde agent.yaml o usa defaults."""
        import yaml
        
        default_config = {
            'agent': {
                'model': 'qwen3.6:latest',
                'max_iterations': 10,
                'max_tool_calls': 20,
                'max_execution_time_seconds': 300,
                'max_tokens': 4096
            },
            'security': {
                'strict_mode': True,
                'allowed_permissions': ['read', 'write'],
                'risk_threshold': 'high'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    # Merge con defaults
                    for key in yaml_config:
                        default_config[key] = yaml_config[key]
                self.logger.info(f"Configuración cargada desde {config_path}")
            except Exception as e:
                self.logger.warning(f"No se pudo cargar config: {e}, usando defaults")
        
        return default_config
    
    def _get_tool_metadata(self) -> Dict[str, ToolMetadata]:
        """Obtiene metadatos de herramientas desde el registry."""
        # Esta información debería venir de tool_registry (implementación futura)
        # Por ahora, definimos las interfaces disponibles
        
        tools = {}
        
        if hasattr(self.model_registry, 'get_tool_metadata'):
            for tool_name in ['filesystem', 'search', 'shell', 'git']:
                try:
                    metadata = self.model_registry.get_tool_metadata(tool_name)
                    tools[tool_name] = metadata
                except (AttributeError, KeyError):
                    continue
        
        # Si no hay registry implementado, usar defaults seguros
        if not tools:
            tools = {
                'filesystem': ToolMetadata(
                    name='filesystem',
                    description='Operaciones de archivos y directorios',
                    input_schema={'path': 'string', 'content': 'string (opcional)'},
                    permission='write',
                    risk='medium',
                    timeout=30,
                    cost=0.0
                ),
                'search': ToolMetadata(
                    name='search',
                    description='Búsqueda en internet o base de conocimiento',
                    input_schema={'query': 'string'},
                    permission='read',
                    risk='low',
                    timeout=60,
                    cost=0.0
                ),
                'shell': ToolMetadata(
                    name='shell',
                    description='Ejecutar comandos del sistema operativo',
                    input_schema={'command': 'string'},
                    permission='execute',
                    risk='high',
                    timeout=30,
                    cost=0.0
                ),
                'git': ToolMetadata(
                    name='git',
                    description='Operaciones con repositorios Git',
                    input_schema={'action': 'string', 'args': 'list'},
                    permission='execute',
                    risk='medium',
                    timeout=60,
                    cost=0.0
                )
            }
        
        return tools
    
    def run(self, user_input: str) -> Message:
        """
        Punto de entrada principal del ciclo. Expone run(user_input) -> response.
        
        Este método:
        1. Crea un nuevo estado de sesión
        2. Conecta todo a través del orchestrator
        3. Devuelve la respuesta final
        
        NO hardcodea nada; usa las capas inyectadas (orchestrator, router, planner).
        
        Args:
            user_input: Entrada textual del usuario
            
        Returns:
            Message con la respuesta del agente
        """
        self.logger.info(f"Agente ejecutando: {user_input[:50]}...")
        
        # Crear nuevo estado de sesión para esta ejecución
        state = self.state_factory()
        self.current_session = state
        
        # Construir herramientas stub disponibles (puede ser reemplazado por tool_executor real)
        tools_map = self._get_tools_map()
        
        # Ejecutar ciclo completo vía orchestrator (§12)
        response = self.orchestrator.run(
            task=user_input,
            state=state,
            tools_map=tools_map,
            logger_callback=self.metrics_logger  # Métricas inyectadas (§12)
        )
        
        self.logger.info("Ciclo del agente completado")
        return response
    
    def _get_tools_map(self) -> Dict[str, Callable]:
        """
        Construye el mapa de herramientas para el orchestrator.
        
        En una implementación real, esto sería inyectado desde una capa de 
        herramientas externa (tools/). Aquí usamos stubs seguros para testing.
        
        Returns:
            dict[str, Callable] - Mapeo nombre_herramienta -> función ejecutable
        """
        # Por seguridad y simplicidad en esta fase, devolvemos stubs
        # En producción, estas funciones vendrían de tools/ con acceso real
        
        def safe_filesystem(args: dict, permission='default', risk_level='low') -> Any:
            """Stub seguro para filesystem."""
            if risk_level in ['high', 'critical']:
                raise PermissionError("Acceso denegado por política de seguridad")
            
            path = args.get('path', 'no_path')
            content = args.get('content', '')
            
            # Loguear operación (seguridad)
            logger.info(f"[TOOLS] filesystem: {args}")
            
            return {"status": "success", "operation": "filesystem_stub", "path": path}
        
        def safe_search(args: dict, permission='default', risk_level='low') -> Any:
            """Stub seguro para search."""
            query = args.get('query', 'no_query')
            
            logger.info(f"[TOOLS] search: {args}")
            
            return {"status": "success", "operation": "search_stub", "results": [f"Result para '{query}'"]}
        
        def safe_shell(args: dict, permission='default', risk_level='low') -> Any:
            """Stub seguro para shell (no ejecuta comandos reales en testing)."""
            command = args.get('command', 'no_command')
            
            if risk_level in ['high', 'critical'] and not self.config.get('security', {}).get('strict_mode'):
                raise PermissionError("Ejecución de shell bloqueada por seguridad")
            
            logger.info(f"[TOOLS] shell (stub): {args}")
            
            return {"status": "success", "operation": "shell_stub", "output": f"Comando ejecutado: {command}"}
        
        def safe_git(args: dict, permission='default', risk_level='low') -> Any:
            """Stub seguro para git."""
            action = args.get('action', 'status')
            
            logger.info(f"[TOOLS] git: {args}")
            
            return {"status": "success", "operation": "git_stub", "action": action}
        
        return {
            'filesystem': safe_filesystem,
            'search': safe_search,
            'shell': safe_shell,
            'git': safe_git
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
def create_agent(config_path: str = "config/agent.yaml", model_name: str = "qwen3.6:latest") -> LocalAgent:
    """
    Fábrica para crear un agente con configuración por defecto.
    
    Args:
        config_path: Ruta al archivo de configuración YAML
        model_name: Nombre del modelo Ollama a usar
        
    Returns:
        Instancia de LocalAgent configurada
    """
    return LocalAgent(
        config_path=config_path,
        model_name=model_name
    )


if __name__ == "__main__":
    # Ejemplo de uso standalone
    print("Inicializando agente local...")
    
    agent = create_agent(model_name="qwen3.6:latest")
    
    # Prueba con input simple
    response = agent.run("¿Qué herramientas tienes disponibles?")
    
    print(f"\nRespuesta del agente: {response.content}")
    print(f"Métricas: {len(agent.get_metrics())} registros")