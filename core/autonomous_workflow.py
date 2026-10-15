import time
from typing import Callable, Dict, Optional
from core.state import SessionState, Message
    

async def _run_autonomous_workflow(
        self,
        task: str,
        state: SessionState,
        tools_map: Dict[str, Callable],
        logger_callback=None,
        stream_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Message:
        """
        Ejecuta el ciclo actual del agente autónomo.

        Flujo:
        1. UNDERSTAND
        2. PLAN
        3. EXECUTE/ROUTE
        4. OBSERVE
        5. RESPOND / REPLAN
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

                    if (
                        decision_tool_name == "list_files"
                        and step_idx < len(plan_result.steps) - 1
                    ):
                        requires_replan = True
                        break
                        
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
