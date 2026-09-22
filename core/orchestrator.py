"""
core/orchestrator.py - Ciclo principal del agente (Agent Loop §12)
INPUT → UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → VERIFY → REFLECT → FINAL RESPONSE
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

# Importar módulos propios
from core.state import SessionState, Message
from core.planner import Planner
from core.router import Router, RouterDecision

# Configurar logger
logger = logging.getLogger(__name__)


class OrchestrationMetrics(BaseModel):
    """Métricas de la orquestación."""
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    iterations_count: int = 0
    tool_calls_count: int = 0
    verification_passes: int = 0
    verification_fails: int = 0
    
    def to_dict(self) -> dict:
        return {
            'total_tokens': self.total_tokens,
            'total_time_seconds': round(self.total_time_seconds, 2),
            'iterations_count': self.iterations_count,
            'tool_calls_count': self.tool_calls_count,
            'verification_passes': self.verification_passes,
            'verification_fails': self.verification_fails
        }


class Orchestrator:
    """Orquestador principal del ciclo del agente."""
    
    def __init__(self, llm_client=None, planner: Planner = None, 
                 router: Router = None, config: dict = None):
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client
        self.planner = planner or Planner(llm_client=llm_client)
        self.router = router or Router()
        
        # Configuración por defecto (§13: Límites configurables)
        default_config = {
            'agent': {
                'max_iterations': 10,
                'max_tool_calls': 20,
                'max_execution_time_seconds': 300
            }
        }
        self.config = config or default_config
        self.max_iterations = self.config.get('agent', {}).get('max_iterations', 10)
        self.max_tool_calls = self.config.get('agent', {}).get('max_tool_calls', 20)
        self.max_execution_time = self.config.get('agent', {}).get('max_execution_time_seconds', 300)
        
        # Métricas
        self.metrics = OrchestrationMetrics()
        self.start_time = None
    
    def run(self, task: str, state: SessionState, 
            tools_map: Dict[str, Callable], 
            logger_callback: Optional[Callable] = None) -> Message:
        """
        Ejecuta el ciclo completo del agente.
        
        Args:
            task: Tarea a realizar
            state: Estado actual de la sesión
            tools_map: Diccionario {            
        Returns:
            Message con la respuesta final
        """
        self.logger.info(f"Orquestador iniciando tarea: {task[:50]}...")
        self.start_time = time.time()
        
        # Agregar tarea al estado
        state.add_message('user', task)
        state.current_task = task
        response = None
        
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteración {iteration}/{self.max_iterations}")
            
            # Verificar límites de tiempo (§13)
            elapsed = time.time() - self.start_time
            if elapsed > self.max_execution_time:
                self.logger.warning("Tiempo máximo excedido")
                break
            
            # 1. UNDERSTAND
            state.set_phase('understand')
            intention = self._understand(state)
            
            # 2. PLAN (§14)
            state.set_phase('plan')
            plan_result = self.planner.plan(intention, state.messages[-5].content if len(state.messages) > 0 else None)
            
            # Validar plan
            if not self.planner.validate_plan(plan_result):
                self.logger.error("Plan inválido")
                break
            
            # 3. SELECT TOOLS & EXECUTE (ciclo de herramientas)
            state.set_phase('execute')
            for step_idx, step in enumerate(plan_result.steps[:self.max_tool_calls]):
                # Verificar límites
                if time.time() - self.start_time > self.max_execution_time:
                    break
                    
                if iteration + step_idx >= self.max_iterations:
                    break
                
                # Seleccionar herramienta con router (§12)
                state.set_phase('route')
                decision = self.router.route(intention)
                
                if not decision.tool_name:
                    continue
                
                # Verificar que la herramienta exista en tools_map
                tool_func = tools_map.get(decision.tool_name)
                if not tool_func:
                    self.logger.warning(f"Herramienta {decision.tool_name} no disponible")
                    continue
                
                # 4. EXECUTE (§12 - paso de ejecución)
                state.set_phase('execute')
                try:
                    result = tool_func(
                        step.args,
                        permission=step.get('permission', 'default'),
                        risk_level=step.get('risk', 'low')
                    )
                    
                    # 5. OBSERVE RESULT (§12)
                    state.set_phase('observe')
                    observation = self._observe_result(result)
                    state.results_cache[step.id] = observation
                    
                    # 6. VERIFY (§13 - verificación de seguridad y resultados)
                    state.set_phase('verify')
                    verification_ok = self._verify_result(observation, step)
                    
                    if logger_callback:
                        logger_callback.log(f"iteration_{iteration}_verification", {
                            'step': step.id,
                            'passed': verification_ok
                        })
                    
                    if not verification_ok:
                        self.logger.warning(f"Verificación fallida en paso {step.id}")
                        # Intentar reparar o continuar
                        state.pending_actions.append({
                            'step': step,
                            'attempted': True,
                            'failed': True
                        })
                        continue
                    
                    self.metrics.verification_passes += 1
                    
                except Exception as e:
                    self.logger.error(f"Error ejecutando herramienta: {str(e)}")
                    state.pending_actions.append({
                        'step': step,
                        'attempted': True,
                        'error': str(e)
                    })
                    continue
            
            # 7. REFLECT (§12)
            state.set_phase('reflect')
            should_continue = self._reflect(state, iteration)
            
            self.metrics.iterations_count += 1
            self.metrics.tool_calls_count += len(plan_result.steps)
            
            if not should_continue:
                response = self._generate_response(state)
                break