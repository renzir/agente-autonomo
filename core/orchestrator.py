"""
core/orchestrator.py - Ciclo principal del agente (Agent Loop §12)
INPUT → UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → VERIFY → REFLECT → FINAL RESPONSE
Se inyecta metrics_logger para registrar métricas en puntos clave.
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
                 router: Router = None, config: dict = None, metrics_logger=None):
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client
        self.planner = planner or Planner(llm_client=llm_client)
        self.router = router or Router()
        # ✅ CAMBIO CRÍTICO: Guardar referencia al metrics_logger inyectado (§14)
        self.metrics_logger = metrics_logger

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
        
        # Métricas internas del orchestrator
        self.metrics = OrchestrationMetrics()
        self.start_time = None
    
    async def run(self, task: str, state: SessionState, 
            tools_map: Dict[str, Callable], 
            logger_callback=None,
            stream_callback: Optional[Callable[[str, bool], None]] = None) -> Message:  # AGREGA ESTE PARÁMETRO
        """
        Ejecuta el ciclo completo del agente.
        
        Args:
            task: Tarea a realizar
            state: Estado actual de la sesión
            tools_map: Diccionario {tool_name: function}
            logger_callback: Callback opcional para logs adicionales
            stream_callback: Callback opcional para streaming de respuestas
        Returns:
            Message con la respuesta final
        """
        self.logger.info(f"Orquestador iniciando tarea: {task[:50]}...")
        loop_start_time = time.time()
        
        # ✅ CAMBIO CRÍTICO §14: Registrar inicio de ejecución en métricas
        if self.metrics_logger:
            self.metrics_logger.log("loop_start", {"task": task})

        # Inicializar lista para latencias de herramientas si no existe
        if not hasattr(self, '_tool_latencies'):
            self._tool_latencies = []
        
        # Agregar tarea al estado
        state.add_message('user', task)
        state.current_task = task
        response = None
        
        total_tool_calls_made = 0  # Contador para límites (§13)
        
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteración {iteration}/{self.max_iterations}")
            
            # ✅ CAMBIO CRÍTICO §14: Registrar inicio de iteración
            if self.metrics_logger:
                self.metrics_logger.log("iteration_start", {"iteration": iteration})

            # Verificar límites de tiempo (§13)
            elapsed = time.time() - loop_start_time
            if elapsed > self.max_execution_time:
                self.logger.warning("Tiempo máximo excedido")
                break
            
            # 1. UNDERSTAND
            state.set_phase('understand')
            intention = await self._understand(state, stream_callback=stream_callback)
            

            # 2. PLAN (§14)
            state.set_phase('plan')
            
            # Obtener contexto de mensajes anteriores (últimos 5 si existen, o todos disponibles)
            context_messages = None
            if len(state.messages) > 0:
                # Tomar los últimos 5 mensajes como contexto (o menos si no hay tantos)
                start_idx = max(0, len(state.messages) - 5)
                context_messages = [msg.content for msg in state.messages[start_idx:]]
                context_str = "\n".join(context_messages) if context_messages else None
            
            plan_result = self.planner.plan(intention, context_str)

            # Validar plan
            if not self.planner.validate_plan(plan_result):
                self.logger.error("Plan inválido")
                break
            
            # 3. SELECT TOOLS & EXECUTE (ciclo de herramientas)
            state.set_phase('execute')
            for step_idx, step in enumerate(plan_result.steps[:self.max_tool_calls]):
          
                # CORREGIDO §12: Límites claros y separados
                # Límite de tiempo total
                if time.time() - loop_start_time > self.max_execution_time:
                    self.logger.warning("Tiempo máximo excedido en paso")
                    break
                
                # Límite de iteraciones externas (no mezclar con step_idx)
                if iteration > self.max_iterations:
                    self.logger.warning("Máximo de iteraciones alcanzado")
                    break
                
                # ✅ CAMBIO CRÍTICO §13: Limitar tool_calls totales
                if total_tool_calls_made >= self.max_tool_calls:
                    self.logger.warning(f"Límite de {self.max_tool_calls} tool calls alcanzado. Deteniendo ejecución.")
                    break
                
                # Límite de tool calls por paso del plan
                if step_idx >= len(plan_result.steps):
                    self.logger.warning("No hay más pasos en el plan")
                    break
                
                # Seleccionar herramienta con router (§12)
                state.set_phase('route')
                decision = self.router.route(intention, available_tools=tools_map)
                
                if not decision.tool_name:
                    continue
                
                # Verificar que la herramienta exista en tools_map
                tool_func = tools_map.get(decision.tool_name)
                if not tool_func:
                    self.logger.warning(f"Herramienta {decision.tool_name} no disponible")
                    continue

                # ✅ CAMBIO CRÍTICO §14: Registrar llamada a herramienta antes de ejecutarla
                if self.metrics_logger:
                    self.metrics_logger.log("tool_call", {"tool": decision.tool_name, "iteration": iteration})
                
                tool_exec_start = time.time()  # Medir latencia por tool call

                try:
                    result = tool_func(
                        step.args,
                        permission='default',
                        risk_level='low'
                    )
                    
                    tool_elapsed = time.time() - tool_exec_start
                    
                    # ✅ CAMBIO CRÍTICO §14: Registrar latencia de cada tool call
                    if self.metrics_logger:
                        self._tool_latencies.append(tool_elapsed)

                    # 5. OBSERVE RESULT (§12)
                    state.set_phase('observe')
                    observation = self._observe_result(result)
                    state.results_cache[step.id] = observation
                    
                    # ✅ CAMBIO CRÍTICO §13: Incrementar contador de tool_calls reales ejecutados
                    total_tool_calls_made += 1

                    # 6. VERIFY (§13 - verificación de seguridad y resultados)
                    state.set_phase('verify')
                    verification_ok = self._verify_result(observation, step)
                    
                    if logger_callback:
                        logger_callback.log(f"iteration_{iteration}_verification", {
                            'step': step.id,
                            'passed': verification_ok
                        })
                    
                    # ✅ CAMBIO CRÍTICO §14: Registrar resultado de verificación en métricas
                    if self.metrics_logger and not verification_ok:
                        self.metrics_logger.log("tool_error", {"tool": decision.tool_name, "step": step.id})

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
                    tool_elapsed = time.time() - tool_exec_start
                    
                    # ✅ CAMBIO CRÍTICO §14: Registrar error de herramienta en métricas
                    if self.metrics_logger:
                        self._tool_latencies.append(tool_elapsed)

                    if self.metrics_logger:
                        self.metrics_logger.log("tool_error", {
                            "tool": decision.tool_name, 
                            "error": str(e),
                            "iteration": iteration
                        })

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
                response = await self._generate_response(state)
                break
        
        # ✅ CAMBIO CRÍTICO §14: Registrar métricas finales del loop completo (latency, iterations)
        total_time = time.time() - loop_start_time
        avg_tool_latency = 0.0
        if hasattr(self, '_tool_latencies') and self._tool_latencies:
            avg_tool_latency = sum(self._tool_latencies) / len(self._tool_latencies)

        final_metrics_bundle = {
            "latency": round(total_time, 3),
            "iterations_used": iteration,
            "total_tool_calls_executed": total_tool_calls_made,
            "avg_tool_latency_sec": round(avg_tool_latency, 4),
            "verification_passes": self.metrics.verification_passes,
            "verification_fails": self.metrics.verification_fails
        }

        if self.metrics_logger:
            # Registrar métricas finales de orquestación como 'orchestration_metrics'
            self.metrics_logger.log("orchestration_summary", final_metrics_bundle)

        # CORRECCIÓN GRUPO B: Si el bucle terminó por límite de iteraciones y response es None,
        # generamos una respuesta de fallback para evitar NoneType errors en tests o consumers.
        if response is None:
            self.logger.warning("Límite de iteraciones alcanzado sin finalizar la tarea exitosamente.")
            response = await self._generate_response(state, stream_callback=stream_callback)

        return response 
    
    # ====================================================================
    # Métodos auxiliares del agente loop (§12, §13)
    # ====================================================================
    
    async def _understand(self, state: SessionState, stream_callback: Optional[Callable[[str, bool], None]] = None) -> str:
        """UNDERSTAND: Interpreta la intención del usuario. Soporta streaming."""
        if not self.llm_client:
            return "Generic intent for now"

        system_prompt = (
            "Eres un analizador de intenciones para un agente autónomo. "
            "Tu tarea es extraer una descripción concisa y clara de la intención del usuario "
            "a partir de su mensaje. Devuelve SOLO la descripción de la intención, sin explicaciones adicionales."
        )

        last_user_msg = None
        for msg in reversed(state.messages):
            if msg.role == 'user':
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return "No user input found"

        try:
            if stream_callback:
                full_intent = ""
                async for chunk, is_final in self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Mensaje del usuario: {last_user_msg}"}
                    ],
                    temperature=0.1,
                    max_tokens=256,
                    stream=True
                ):
                    full_intent += chunk
                    stream_callback(chunk, is_final)

                return full_intent.strip() if full_intent else "Generic intent for now"

            response = await self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Mensaje del usuario: {last_user_msg}"}
                ],
                temperature=0.1,
                max_tokens=256
            )

            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "") or response.get("content", "")
            else:
                content = str(response)

            return content.strip() if content else "Generic intent for now"

        except Exception as e:
            self.logger.error(f"Error al interpretar intención del usuario: {str(e)}")
            return "Generic intent for now"
    
    def _observe_result(self, result: Any) -> Dict[str, Any]:
        """OBSERVE: Observa y normaliza el resultado de una herramienta (§12)."""
        if isinstance(result, dict):
            return result
        return {"status": "success", "result": str(result)}
    
    def _verify_result(self, observation: Dict[str, Any], step) -> bool:
        """VERIFY: Verifica seguridad y corrección del resultado (§13)."""
        # Verificar que no haya errores críticos
        if observation.get("error"):
            return False
        # Verificar estructura básica
        if not observation.get("status"):
            return False
        return True
    
    def _reflect(self, state: SessionState, iteration: int) -> bool:
        """REFLECT: Decide si continúa o termina el ciclo (§12)."""
        # Si ya se completaron todas las tareas del plan
        if len(state.pending_actions) == 0 and hasattr(state, 'completed_tasks') and state.completed_tasks:
            return False
        
        # Limitar por número de iteraciones externas
        if iteration >= self.max_iterations:
            self.logger.info("Iteraciones máximas alcanzadas, finalizando")
            return False
        
        return True
    
    async def _generate_response(self, state: SessionState, stream_callback: Optional[Callable[[str, bool], None]] = None) -> Message:
        """Genera la respuesta final al usuario usando el LLM cuando está disponible.
        
        Si se proporciona stream_callback, utiliza streaming para enviar los chunks de texto.
        De lo contrario, usa el comportamiento estándar (no streaming).
        """
        if not (self.llm_client and state.current_task):
            return Message(
                role="assistant",
                content=f"Tarea completada (stub). Iteraciones: {self.metrics.iterations_count}"
            )

        # Construir contexto con los resultados de las herramientas ejecutadas
        tool_results = []
        results_cache = getattr(state, 'results_cache', {})
        for step_id, result in (results_cache or {}).items():
            tool_results.append(f"Paso {step_id}: {result}")

        results_context = "\n".join(tool_results) if tool_results else "Sin resultados previos."

        system_prompt = (
            "Eres un asistente que genera respuestas finales claras y útiles para el usuario. "
            "Basándote en los resultados de las herramientas ejecutadas, proporciona una respuesta "
            "concisa y amigable confirmando la tarea completada o explicando lo sucedido."
        )

        user_prompt = f"La tarea era: {state.current_task}\nResultados de herramientas:\n{results_context}"

        try:
            # Camino con streaming si hay callback disponible
            if stream_callback:
                full_content = ""
                async for chunk, is_final in self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    stream=True
                ):
                    full_content += chunk
                    if stream_callback:
                        stream_callback(chunk, is_final)

                return Message(
                    role="assistant", 
                    content=full_content.strip() if full_content else "Tarea completada."
                )

            # Camino estándar (no streaming) para compatibilidad hacia atrás
            response = await self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )

            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "") or response.get("content", "")
            else:
                content = str(response)

            return Message(role="assistant", content=content.strip() if content else "Tarea completada.")

        except Exception as e:
            self.logger.error(f"Error al generar respuesta final con LLM: {str(e)}")
            # Fallback en caso de error incluso con streaming
            return Message(
                role="assistant",
                content=f"Tarea completada (error). Iteraciones: {self.metrics.iterations_count}"
            )