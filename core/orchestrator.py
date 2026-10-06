"""
core/orchestrator.py - Ciclo principal del agente (Agent Loop §12)
INPUT → UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → FINAL RESPONSE

Simplificado para reducir redundancia entre UNDERSTAND → PLAN → ROUTE.
El Router no adivina herramientas mediante keywords.
Se prepara el flujo para la siguiente etapa de native tool calling.
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
        self.metrics_logger = metrics_logger

        # Configuración por defecto (§13: Límites configurables)
        default_config = {
            'agent': {
                'max_iterations': 10,
                'max_tool_calls': 20,
                'max_execution_time_seconds': 60
            }
        }
        self.config = config or default_config
        self.max_iterations = self.config.get('agent', {}).get('max_iterations', 10)
        self.max_tool_calls = self.config.get('agent', {}).get('max_tool_calls', 20)
        self.max_execution_time = self.config.get('agent', {}).get('max_execution_time_seconds', 60)
        
        # Métricas internas del orchestrator
        self.metrics = OrchestrationMetrics()
        self.start_time = None
    
    async def run(self, task: str, state: SessionState, 
            tools_map: Dict[str, Callable], 
            logger_callback=None,
            stream_callback: Optional[Callable[[str, bool], None]] = None) -> Message:
        """
        Ejecuta el ciclo simplificado del agente.

        Flujo:
        1. UNDERSTAND: Extrae intención (para contexto/métricas).
        2. PLAN: Si no hay respuesta previa, pide pasos al Planner.
        3. EXECUTE/ROUTE: Si hay pasos y herramientas disponibles, ejecuta.
        4. OBSERVE: Guarda resultados en state/results_cache e historial.
        5. RESPOND: Genera respuesta final con los resultados.
        
        No depende de keywords post-understand para decidir ejecutar herramientas.
        """
        print("[TRACE] ORCHESTRATOR START")
        self.logger.info(f"Orquestador iniciando tarea: {task[:50]}...")
        loop_start_time = time.time()
        
        # Registrar inicio de ejecución en métricas
        if self.metrics_logger:
            self.metrics_logger.log("loop_start", {"task": task})

        # Inicializar lista para latencias de herramientas si no existe
        if not hasattr(self, '_tool_latencies'):
            self._tool_latencies = []
        
        # Agregar tarea al estado
        state.add_message('user', task)
        state.current_task = task
        response = None
        
        total_tool_calls_made = 0
        plan_result = None
        iteration = 0
        intention = None
        
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Iteración {iteration}/{self.max_iterations}")
            
            # Verificar límites de tiempo (§13)
            if time.time() - loop_start_time > self.max_execution_time:
                self.logger.warning("Tiempo máximo excedido")
                break
            
            # Protección temporal contra loops (no como mecanismo normal)
            if iteration > self.max_iterations:
                self.logger.warning("Máximo de iteraciones alcanzado")
                break

            # ====================================================================
            # PASO 1: UNDERSTAND - Extraer intención del usuario
            # ====================================================================

            if intention is None:
                print("[TRACE] ORCHESTRATOR -> UNDERSTAND")
                state.set_phase('understand')
                intention = await self._understand(
                    state,
                    stream_callback=stream_callback
                )
                print("[TRACE] ORCHESTRATOR <- UNDERSTAND")
            # ====================================================================
            # PASO 2: PLANIFICACIÓN
            # ====================================================================
            state.set_phase('plan')
            
            # Obtener contexto para el planner
            context_parts = []

            if len(state.messages) > 0:
                start_idx = max(0, len(state.messages) - 5)
                context_messages = [
                    msg.content for msg in state.messages[start_idx:]
                ]

                if context_messages:
                    context_parts.append(
                        "Mensajes recientes:\n" +
                        "\n".join(context_messages)
                    )

            # Incluir resultados reales de herramientas ejecutadas previamente.
            # Esto permite que el Planner conozca archivos, rutas y datos
            # descubiertos durante una iteración anterior.
            results_cache = getattr(state, 'results_cache', {})

            if results_cache:
                tool_results = []

                for step_id, result in results_cache.items():
                    tool_results.append(
                        f"Paso {step_id}: {result}"
                    )

                context_parts.append(
                    "Resultados de herramientas ejecutadas:\n" +
                    "\n".join(tool_results)
                )

            context_str = (
                "\n\n".join(context_parts)
                if context_parts
                else None
            )

            # Llamamos al planner para obtener pasos.
            # Si plan_result es None, se genera una nueva planificación
            # utilizando también los resultados obtenidos previamente.
            if not plan_result:
                print("[TRACE] ORCHESTRATOR -> PLANNER")

                try:
                    plan_result = await self.planner.plan(
                        intention,
                        context_str
                    )

                except Exception as e:
                    self.logger.error(f"Error en Planner: {e}")
                    state.set_phase('error')
                    raise

                print("[TRACE] ORCHESTRATOR <- PLANNER")

            # Validar que el plan no esté vacío
            if not plan_result or not hasattr(plan_result, 'steps') or len(plan_result.steps) == 0:
                state.set_phase('respond')
                self.logger.info("No steps in plan. Generating simple response.")
                response = await self._generate_simple_response(
                    state,
                    intention,
                    stream_callback=stream_callback
                )
                break
            
            # Validar plan estructuralmente
            if not self.planner.validate_plan(plan_result):
                self.logger.error("Plan inválido según validación estructural")
                break

                        # ====================================================================
            # PASO 3: EJECUCIÓN DE HERRAMIENTAS (Simplificado)
            # Ejecutamos los pasos del plan devueltos por el Planner.
            # No usamos keywords para decidir, confiamos en lo que el Planner dijo.
            # ====================================================================
            state.set_phase('execute')

            tools_executed_in_iteration = False
            requires_replan = False

            for step_idx, step in enumerate(plan_result.steps[:self.max_tool_calls]):

                # Límite de tool calls totales (§13)
                if total_tool_calls_made >= self.max_tool_calls:
                    self.logger.warning(
                        f"Límite de {self.max_tool_calls} tool calls alcanzado."
                    )
                    break

                # ====================================================================
                # ROUTER - Resolver el Step contra el catálogo de herramientas
                # ====================================================================
                state.set_phase('route')

                decision = self.router.route(step)

                print(f"[TRACE] ROUTER DECISION: {decision}")

                if not decision.tool_name:
                    self.logger.warning(
                        f"Router no pudo resolver el step {step.id}: {step.description}"
                    )
                    continue

                decision_tool_name = decision.tool_name
                decision_args = decision.args or {}

                # Obtener la herramienta real desde el ToolRegistry
                try:
                    tool_entry = self.router.tool_registry.get_tool(
                        decision_tool_name
                    )
                    tool_func = tool_entry["fn"]

                except KeyError:
                    self.logger.warning(
                        f"Herramienta {decision_tool_name} no existe en ToolRegistry"
                    )
                    continue

                # Registrar llamada a herramienta antes de ejecutarla (§14)
                if self.metrics_logger:
                    self.metrics_logger.log(
                        "tool_call",
                        {
                            "tool": decision_tool_name,
                            "iteration": iteration
                        }
                    )

                tool_exec_start = time.time()

                try:
                    # Ejecutar la herramienta real
                    result = tool_func(
                        decision_args,
                        permission='default',
                        risk_level='low'
                    )

                    print(f"[TRACE] TOOL RESULT: {result}")

                    tool_elapsed = time.time() - tool_exec_start

                    # Registrar latencia de cada tool call (§14)
                    if self.metrics_logger:
                        self._tool_latencies.append(tool_elapsed)

                    # OBSERVE RESULT (incorporar al estado/historial)
                    state.set_phase('observe')
                    observation = self._observe_result(result)

                    # Incorporar resultado al historial para la respuesta siguiente
                    state.results_cache[step.id] = observation

                    # Incrementar contador de tool_calls reales ejecutados (§13)
                    total_tool_calls_made += 1
                    tools_executed_in_iteration = True

                    # Si acabamos de descubrir el contenido de una carpeta,
                    # necesitamos volver al Planner con ese resultado real.
                    if decision_tool_name == "list_files":
                        requires_replan = True

                    if logger_callback:
                        logger_callback.log(
                            f"iteration_{iteration}_verification",
                            {
                                'step': step.id,
                                'passed': True
                            }
                        )

                    self.metrics.verification_passes += 1

                except Exception as e:
                    tool_elapsed = time.time() - tool_exec_start

                    if self.metrics_logger:
                        self._tool_latencies.append(tool_elapsed)

                    if self.metrics_logger:
                        self.metrics_logger.log(
                            "tool_error",
                            {
                                "tool": decision_tool_name,
                                "error": str(e),
                                "iteration": iteration
                            }
                        )

                    self.logger.error(
                        f"Error ejecutando herramienta: {str(e)}"
                    )

                    # Registrar error en el historial para la respuesta final
                    state.results_cache[step.id] = {
                        "status": "error",
                        "error": str(e)
                    }

                    continue
            # ====================================================================
            # PASO 4: FINALIZAR O CONTINUAR
            # Solo finalizamos cuando no quedan pasos pendientes y no
            # necesitamos una nueva planificación.
            # ====================================================================

            # Si acabamos de descubrir archivos/directorios, debemos volver
            # al Planner utilizando los resultados reales obtenidos.
            if requires_replan:
                print("[TRACE] ORCHESTRATOR -> REPLAN")
                plan_result = None
                continue

            is_last_step = step_idx >= len(plan_result.steps) - 1

            if is_last_step:
                # Ya no quedan steps pendientes.
                # Si el resultado de la última herramienta es directamente
                # presentable, no hacemos otra llamada al LLM.
                if tools_executed_in_iteration:
                    response = self._generate_direct_response(state)
                else:
                    self.logger.warning(
                        "El último step no pudo ejecutarse. Generando respuesta final."
                    )
                    response = await self._generate_response(
                        state,
                        stream_callback=stream_callback
                    )

                break

            # Todavía quedan steps pendientes.
            # Si no se ejecutó ninguna herramienta en esta iteración,
            # no podemos avanzar de forma segura.
            if not tools_executed_in_iteration:
                self.logger.warning(
                    "Ninguna herramienta se ejecutó en el step actual. "
                    "No se puede continuar el plan."
                )
                response = await self._generate_response(
                    state,
                    stream_callback=stream_callback
                )
                break

            # Si llegamos al límite de iteraciones sin finalizar,
            # generamos una respuesta de fallback.
            if response is None:
                self.logger.warning(
                    "Límite de iteraciones alcanzado sin finalizar "
                    "la tarea exitosamente."
                )
                response = await self._generate_response(
                    state,
                    stream_callback=stream_callback
                )

            # ====================================================================
            # MÉTRICAS
            # ====================================================================

            self.metrics.tool_calls_count = total_tool_calls_made

        return response        

    async def _understand(
        self,
        state: SessionState,
        stream_callback: Optional[Callable[[str, bool], None]] = None
    ) -> str:
        """UNDERSTAND: Extrae una intención mínima para el Planner."""

        if not self.llm_client:
            return "Generic intent for now"

        system_prompt = (
            "Extrae la intención del usuario para un agente de programación. "
            "Responde SOLO con una frase muy breve, máximo 15 palabras. "
            "No expliques ni describas razonamiento. "
            "Conserva nombres de archivos, rutas, comandos y datos importantes."
        )

        last_user_msg = None

        for msg in reversed(state.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break

        if not last_user_msg:
            return "No user input found"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_user_msg}
        ]

        try:
            if stream_callback:
                full_intent = ""

                chat_generator = await self.llm_client.chat(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=32,
                    stream=True, 
                    
                )

                async for chunk, is_final in chat_generator:
                    if chunk:
                        full_intent += chunk
                        stream_callback(chunk, is_final)

                return (
                    full_intent.strip()
                    if full_intent.strip()
                    else "Generic intent for now"
                )

            response = await self.llm_client.chat(
                messages=messages,
                temperature=0.0,
                max_tokens=32,
                stream=False, 
                think=False
            )

            if isinstance(response, dict):
                content = (
                    response.get("message", {}).get("content", "")
                    or response.get("content", "")
                )
            else:
                content = str(response)

            return content.strip() if content else "Generic intent for now"

        except Exception as e:
            self.logger.error(
                f"Error al interpretar intención del usuario: {str(e)}"
            )
            return "Generic intent for now"


    def _observe_result(self, result: Any) -> Dict[str, Any]:
        """OBSERVE: Observa y normaliza el resultado de una herramienta (§12)."""
        if isinstance(result, dict):
            return result
        return {"status": "success", "result": str(result)}
    
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

    async def _generate_simple_response(self, state: SessionState, intention: str, 
            stream_callback: Optional[Callable[[str, bool], None]] = None) -> Message:
        """Responde directamente para intenciones simples sin pasar por herramientas."""
        import asyncio
        if not self.llm_client:
            return Message(role="assistant", content=f"Recibí: {intention}")

        system_prompt = "Eres un asistente útil. Responde de forma breve y amable."
        user_content = f"Usuario dice: {intention}"

        try:
            if stream_callback:
                full_content = ""
                chat_generator = await self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    max_tokens=256,
                    stream=True
                )
                
                if hasattr(chat_generator, '__aiter__'):
                    async for chunk, is_final in chat_generator:
                        if chunk:
                            full_content += chunk
                            stream_callback(chunk, is_final)
                else:
                    content = chat_generator.get("message", {}).get("content", "") if isinstance(chat_generator, dict) else str(chat_generator)
                    return Message(role="assistant", content=content.strip())

                return Message(role="assistant", content=full_content.strip() if full_content else "Tarea completada.")

            # Camino no-streaming (para compatibilidad con mocks y tests)
            raw_response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=256
            )
            
            if asyncio.iscoroutine(raw_response):
                response_dict = await raw_response
            else:
                response_dict = raw_response
            
            content = response_dict.get("message", {}).get("content", "") if isinstance(response_dict, dict) else str(response_dict)

            return Message(role="assistant", content=content.strip())

        except Exception as e:
            self.logger.error(f"Error al generar respuesta simple con LLM: {str(e)}")
            return Message(role="assistant", content=f"Tarea no completada (error). {str(e)}")

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
                chat_generator = await self.llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=512,
                    stream=True
                )
                
                # Iterar correctamente sobre el generador
                if hasattr(chat_generator, '__aiter__'):
                    async for chunk, is_final in chat_generator:
                        if chunk:
                            full_content += chunk
                            if stream_callback:
                                stream_callback(chunk, is_final)
                else:
                     content = chat_generator.get("message", {}).get("content", "") if isinstance(chat_generator, dict) else str(chat_generator)
                     return Message(role="assistant", content=content.strip())

                return Message(
                    role="assistant", 
                    content=full_content.strip() if full_content else "Tarea completada."
                )

            # Camino estándar (no streaming) para compatibilidad hacia atrás
            # Camino estándar (no streaming)
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

            return Message(
                role="assistant",
                content=content.strip() if content else "Tarea completada."
            )
        except Exception as e:
            self.logger.error(f"Error al generar respuesta final con LLM: {str(e)}")
            # Fallback en caso de error incluso con streaming
            return Message(
                role="assistant",
                content=f"Tarea completada (error). Iteraciones: {self.metrics.iterations_count}"
            )

    def _generate_direct_response(self, state: SessionState) -> Message:
        """Devuelve directamente el resultado de una herramienta sin llamar al LLM."""

        results_cache = getattr(state, 'results_cache', {})

        if not results_cache:
            return Message(
                role="assistant",
                content="La herramienta se ejecutó correctamente, pero no devolvió resultados."
            )

        # Tomamos el último resultado registrado
        _, result = next(reversed(results_cache.items()))

        # Resultado normalizado por _observe_result()
        if isinstance(result, dict):
            if result.get("status") == "success":
                content = result.get("result", "")
            else:
                content = result.get("error", str(result))
        else:
            content = str(result)

        return Message(
            role="assistant",
            content=content.strip() if content else "La herramienta se ejecutó correctamente."
        )