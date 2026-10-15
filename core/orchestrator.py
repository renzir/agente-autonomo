"""
core/orchestrator.py - Ciclo principal del agente (Agent Loop §12)
INPUT → UNDERSTAND → PLAN → SELECT TOOLS → EXECUTE → OBSERVE → FINAL RESPONSE

Simplificado para reducir redundancia entre UNDERSTAND → PLAN → ROUTE.
El Router no adivina herramientas mediante keywords.
Se prepara el flujo para la siguiente etapa de native tool calling.
"""

import logging
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel, Field
from .autonomous_workflow import _run_autonomous_workflow

# Importar módulos propios
from core.state import SessionState, Message
from core.planner import Planner
from core.router import Router

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

        # Configuración por defecto
        default_config = {
            "agent": {
                "max_iterations": 10,
                "max_tool_calls": 20,
                "max_execution_time": 900,
            }
        }

        self.config = config if config is not None else default_config

        agent_config = self.config.get("agent", {})

        self.max_iterations = agent_config.get(
            "max_iterations", 10
        )

        self.max_tool_calls = agent_config.get(
            "max_tool_calls", 20
        )

        # Preferir la clave del YAML y admitir el nombre antiguo
        self.max_execution_time = agent_config.get(
            "max_execution_time",
            agent_config.get("max_execution_time_seconds", 900),
        )

        print(
        f"[TRACE] MAX EXECUTION TIME: "
        f"{self.max_execution_time}s"
        )
        self.logger.info(
            "Límites del Orchestrator: iterations=%s, "
            "tool_calls=%s, execution_time=%ss",
            self.max_iterations,
            self.max_tool_calls,
            self.max_execution_time,
        )

        # Métricas internas del orchestrator
        self.metrics = OrchestrationMetrics()
        self.start_time = None

    async def run(
        self,
        task: str,
        state: SessionState,
        tools_map: Dict[str, Callable],
        logger_callback=None,
        stream_callback: Optional[Callable[[str, bool], None]] = None,
        mode: str = "conversational"
    ) -> Message:

        print(f"[TRACE] ORCHESTRATOR MODE: {mode}")

        if mode == "autonomous":
            return await _run_autonomous_workflow(
                self,
                task=task,
                state=state,
                tools_map=tools_map,
                logger_callback=logger_callback,
                stream_callback=stream_callback
            )
        if mode == "analysis":
            return await self._run_analysis_mode(
                task=task,
                state=state,
                tools_map=tools_map,
                logger_callback=logger_callback,
                stream_callback=stream_callback
            )

        return Message(
            role="assistant",
            content="[SISTEMA] Modo no implementado."
        )

    async def _run_analysis_mode(
        self,
        task: str,
        state: SessionState,
        tools_map: Dict[str, Callable],
        logger_callback=None,
        stream_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Message:
        """Ejecuta el flujo de análisis estático."""

        import time
        import re
        from pathlib import Path
        from core.batcher import create_batches, split_batch_for_retry
        import asyncio


        print("[TRACE] ANALYSIS MODE START")

        state.set_phase("plan")
        loop_start_time = time.time()
        total_tool_calls = 0

        ALLOWED_TOOLS = {"list_files", "project_tree", "search_text", "read_file"}
        PROHIBITED_TOOLS = {"create_file", "modify_file", "execute_command"}

        # ========================================================================
        # WORKSPACE
        # ========================================================================

        workspace_root = Path(__file__).resolve().parent.parent.resolve()

        EXCLUDED_DIRS = {
            ".git",
            ".venv",
            "node_modules",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            "build",
            "dist",
        }

        # ========================================================================
        # CONTEXTO @CARPETA CON PYTHON
        # ========================================================================

        import os

        initial_paths = set()

        # ====================================================================
        # VALIDAR RUTAS DE ARCHIVOS EXPLÍCITAS ANTES DEL PLANNER
        # ====================================================================


        explicit_path_pattern = re.compile(
            r'(?<![\w./\\-])((?:[\w.-]+[\\/])*[\w.-]+\.[A-Za-z0-9_]+)'
        )

        explicit_paths = explicit_path_pattern.findall(task)
        validated_explicit_files = set()

        for raw_path in explicit_paths:
            normalized_path = raw_path.replace("\\", "/")
            candidate = (workspace_root / normalized_path).resolve()

            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                state.set_phase("error")
                return Message(
                    role="assistant",
                    content=f"[ANALYSIS] Ruta fuera del workspace: {raw_path}",
                )

            if not candidate.is_file():
                # No confundir una mención a un archivo con una ruta válida.
                continue

            relative_path = candidate.relative_to(workspace_root).as_posix()
            validated_explicit_files.add(relative_path)

            print(f"[TRACE] EXPLICIT FILE VALIDATED: {relative_path}")

        print(
            f"[TRACE] EXPLICIT FILES FOUND: "
            f"{len(validated_explicit_files)}"
        )
   

        MAX_CONTEXT_FILES = 2000
        MAX_CONTEXT_DEPTH = 8

        folder_refs = re.findall(
            r'@([A-Za-z0-9_.-]+)',
            task
        )

        for folder_ref in folder_refs:
            potential_dir = (workspace_root / folder_ref).resolve()

            try:
                potential_dir.relative_to(workspace_root)
            except ValueError:
                print(f"[TRACE] CONTEXT REJECTED: @{folder_ref}")
                continue

            if not potential_dir.is_dir():
                continue

            base_depth = len(potential_dir.relative_to(workspace_root).parts)
            resolved_count = 0
            truncated = False

            for current_dir, dirnames, filenames in os.walk(
                potential_dir,
                topdown=True,
                followlinks=False,
            ):
                current_path = Path(current_dir)
                current_depth = (
                    len(current_path.relative_to(workspace_root).parts)
                    - base_depth
                )

                # Podar directorios excluidos antes de recorrerlos.
                dirnames[:] = sorted(
                    name for name in dirnames
                    if name not in EXCLUDED_DIRS
                    and not (current_path / name).is_symlink()
                )

                if current_depth >= MAX_CONTEXT_DEPTH:
                    dirnames.clear()

                for filename in sorted(filenames):
                    file_path = current_path / filename

                    if file_path.is_symlink() or not file_path.is_file():
                        continue

                    relative_path = file_path.relative_to(workspace_root)
                    initial_paths.add(relative_path.as_posix())
                    resolved_count += 1

                    if len(initial_paths) >= MAX_CONTEXT_FILES:
                        truncated = True
                        break

                if truncated:
                    break

            print(
                f"[TRACE] WORKSPACE CONTEXT: "
                f"@{folder_ref} -> {resolved_count} archivos"
                + (" [LÍMITE ALCANZADO]" if truncated else "")
            )

            if truncated:
                break

        print(f"[TRACE] WORKSPACE CONTEXT FILES: {len(initial_paths)}")

        # ========================================================================
        # DESCUBRIMIENTO DINÁMICO
        # ========================================================================


        files_to_analyze = set()
        files_from_planner_directly = set()
        plan_result = None

        # Si se indicó un archivo explícito válido, analizar solo ese archivo.
        if validated_explicit_files:
            files_to_analyze.update(validated_explicit_files)
            files_from_planner_directly.update(validated_explicit_files)

            print(
                "[TRACE] EXPLICIT FILES RESOLVED BY PYTHON: "
                f"{len(files_to_analyze)} archivos"
            )

        # Si @carpeta ya fue resuelta por Python, NO usamos Planner.
        elif initial_paths:
            files_to_analyze.update(initial_paths)

            print(
                "[TRACE] ANALYSIS CONTEXT RESOLVED BY PYTHON: "
                f"{len(initial_paths)} archivos"
            )

        else:

            # ====================================================================
            # FASE 1 + 2: PLANNER / DESCUBRIMIENTO / EJECUCIÓN
            # ====================================================================

            for iteration in range(1, self.max_iterations + 1):

                # ---------------------------------------------------------------
                # Límite de tiempo
                # ---------------------------------------------------------------

                if time.time() - loop_start_time > self.max_execution_time:
                    state.set_phase("error")

                    return Message(
                        role="assistant",
                        content=(
                            "[ERROR] Límite de tiempo excedido "
                            "durante el descubrimiento."
                        )
                    )

                # ---------------------------------------------------------------
                # PLAN
                # ---------------------------------------------------------------

                if (
                    not plan_result
                    or not hasattr(plan_result, "steps")
                    or len(plan_result.steps) == 0
                ):
                    print(
                        f"[TRACE] ANALYSIS -> PLANNER "
                        f"(Iter {iteration})"
                    )

                    state.set_phase("plan")

                    context_parts = []

                    if state.results_cache:
                        for key, val in list(
                            state.results_cache.items()
                        )[-20:]:
                            if isinstance(val, dict):
                                context_parts.append(
                                    f"{key}: {val.get('result', str(val))}"
                                )
                            else:
                                context_parts.append(
                                    f"{key}: {str(val)}"
                                )

                    context_str = (
                        "\n".join(context_parts)
                        if context_parts
                        else None
                    )

                    try:
                        plan_result = await self.planner.plan(
                            task,
                            context_str
                        )

                        print(
                            "[TRACE] ANALYSIS <- PLANNER "
                            f"(Steps: {len(plan_result.steps)})"
                        )

                    except Exception as e:
                        self.logger.error(
                            "Error en Planner durante descubrimiento: %s",
                            e
                        )

                        return Message(
                            role="assistant",
                            content=(
                                "[ERROR] El planificador falló "
                                "al generar el plan de análisis."
                            )
                        )

                # ---------------------------------------------------------------
                # VALIDAR PLAN
                # ---------------------------------------------------------------

                if not self.planner.validate_plan(plan_result):
                    self.logger.warning(
                        "Plan inválido durante análisis. Replanificando..."
                    )

                    plan_result = None
                    continue

                # ================================================================
                # FASE 2: EJECUCIÓN
                # ================================================================

                state.set_phase("execute")

                tools_executed_in_iteration = False
                had_critical_error = False

                # ---------------------------------------------------------------
                # EJECUTAR STEPS
                # ---------------------------------------------------------------

                remaining_calls = (
                    self.max_tool_calls - total_tool_calls
                )

                for step in plan_result.steps[:remaining_calls]:

                    if total_tool_calls >= self.max_tool_calls:
                        self.logger.warning(
                            "Límite de tool calls alcanzado."
                        )
                        break

                    # -----------------------------------------------------------
                    # ROUTER
                    # -----------------------------------------------------------

                    try:
                        decision = self.router.route(step)

                        if not decision.tool_name:
                            state.results_cache[
                                f"step_{step.id}"
                            ] = {
                                "status": "error",
                                "error": (
                                    f"Herramienta '{step.tool}' "
                                    "no válida en modo análisis."
                                )
                            }
                            continue

                        tool_entry = (
                            self.router.tool_registry.get_tool(
                                decision.tool_name
                            )
                        )

                        tool_func = tool_entry["fn"]

                    except KeyError:
                        state.results_cache[
                            f"step_{step.id}"
                        ] = {
                            "status": "error",
                            "error": (
                                f"Herramienta "
                                f"'{getattr(decision, 'tool_name', step.tool)}' "
                                "no encontrada."
                            )
                        }
                        continue

                    # -----------------------------------------------------------
                    # SEGURIDAD
                    # -----------------------------------------------------------

                    if decision.tool_name in PROHIBITED_TOOLS:
                        state.results_cache[
                            f"step_{step.id}"
                        ] = {
                            "status": "error",
                            "error": (
                                "[CRÍTICO] Herramienta prohibida "
                                f"en modo análisis: '{decision.tool_name}'."
                            )
                        }

                        had_critical_error = True
                        continue

                    if decision.tool_name not in ALLOWED_TOOLS:
                        state.results_cache[
                            f"step_{step.id}"
                        ] = {
                            "status": "error",
                            "error": (
                                f"Herramienta '{decision.tool_name}' "
                                "no permitida en modo análisis."
                            )
                        }

                        had_critical_error = True
                        continue

                    # -----------------------------------------------------------
                    # EJECUTAR TOOL
                    # -----------------------------------------------------------

                    try:
                        result = tool_func(
                            decision.args or {},
                            permission="read",
                            risk_level="low"
                        )

                        self.metrics.tool_calls_count += 1
                        total_tool_calls += 1
                        tools_executed_in_iteration = True

                        observation = self._observe_result(result)

                        state.results_cache[
                            f"step_{step.id}"
                        ] = observation

                        if logger_callback:
                            logger_callback.log(
                                "tool_call",
                                {
                                    "tool": decision.tool_name,
                                    "iteration": iteration
                                }
                            )

                        # ========================================================
                        # RESULTADOS DE read_file
                        # ========================================================

                        if decision.tool_name == "read_file":

                            path = (decision.args or {}).get("path")

                            result_text = (
                                result
                                if isinstance(result, str)
                                else ""
                            )

                            result_failed = (
                                result_text.strip().lower().startswith(
                                    ("error:", "[error]", "security error:")
                                )
                                or "file not found" in result_text.lower()
                                or "no such file" in result_text.lower()
                            )

                            if isinstance(result, dict):
                                result_failed = (
                                    result.get("status") in {"error", "failed"}
                                    or bool(result.get("error"))
                                )
                                result_text = str(
                                    result.get("error")
                                    or result.get("result")
                                    or result
                                )

                            if path:
                                try:
                                    candidate = (
                                        workspace_root / str(path)
                                    ).resolve()

                                    candidate.relative_to(workspace_root)

                                except ValueError:
                                    state.set_phase("error")

                                    return Message(
                                        role="assistant",
                                        content=(
                                            "[ANALYSIS] Ruta fuera del "
                                            f"workspace: {path}"
                                        ),
                                    )

                                normalized_task = (
                                    task.replace("\\", "/").lower()
                                )
                                normalized_path = (
                                    str(path).replace("\\", "/").lower()
                                )

                                explicitly_requested = (
                                    normalized_path in normalized_task
                                )

                                if result_failed or not candidate.is_file():

                                    self.logger.warning(
                                        "No se pudo leer el archivo: %s",
                                        path,
                                    )

                                    # Si el usuario pidió expresamente
                                    # ese archivo, detener el análisis.
                                    if explicitly_requested:
                                        state.set_phase("error")

                                        detail = (
                                            result_text
                                            or "El archivo no existe."
                                        )

                                        return Message(
                                            role="assistant",
                                            content=(
                                                "[ANALYSIS] No se pudo leer "
                                                f"el archivo solicitado "
                                                f"'{path}'. Detalle: {detail}"
                                            ),
                                        )

                                else:
                                    files_from_planner_directly.add(
                                        candidate.relative_to(
                                            workspace_root
                                        ).as_posix()
                                    )


                        # ========================================================
                        # RESULTADOS DE list_files
                        # ========================================================

                        elif decision.tool_name == "list_files":

                            if isinstance(result, str):

                                base_path = (
                                    (decision.args or {})
                                    .get("path", ".")
                                )

                                base_path = str(
                                    Path(base_path)
                                ).replace("\\", "/")

                                for line in result.splitlines():

                                    stripped_line = line.strip()

                                    if not stripped_line:
                                        continue

                                    is_dir = (
                                        stripped_line.startswith("[D]")
                                    )

                                    is_file = (
                                        stripped_line.startswith("[F]")
                                    )

                                    if not (is_dir or is_file):
                                        continue

                                    name = (
                                        stripped_line[3:]
                                        .strip()
                                    )

                                    if not name:
                                        continue

                                    # Mantener ruta relativa completa
                                    if base_path in ("", "."):
                                        relative_path = name
                                    else:
                                        relative_path = str(
                                            Path(base_path) / name
                                        )

                                    relative_path = (
                                        relative_path
                                        .replace("\\", "/")
                                    )

                                    if is_file:
                                        files_to_analyze.add(
                                            relative_path
                                        )

                                    elif is_dir:
                                        files_to_analyze.add(
                                            f"[DIR]{relative_path}"
                                        )


                        # ========================================================
                        # RESULTADOS DE search_text
                        # ========================================================

                        elif decision.tool_name == "search_text":

                            if isinstance(result, str):

                                print("[TRACE] SEARCH_TEXT RAW RESULT:")
                                print(result[:4000])

                                found_paths = set()

                                for line in result.splitlines():
                                    stripped_line = line.strip()

                                    if not stripped_line:
                                        continue

                                    # Procesar también líneas con este prefijo.
                                    if stripped_line.startswith("Match found:"):
                                        stripped_line = stripped_line[
                                            len("Match found:"):
                                        ].strip()

                                    # Formato: ruta:numero:contenido
                                    # Acepta rutas relativas y absolutas de Windows.
                                    match = re.match(
                                        r"^(.*?)\s*:\s*(\d+)(?:\s*:|$)",
                                        stripped_line,
                                    )

                                    if not match:
                                        continue

                                    raw_path = (
                                        match.group(1)
                                        .strip()
                                        .strip("'\"")
                                    )

                                    if not raw_path:
                                        continue

                                    raw_path = raw_path.replace("\\", "/")
                                    path_obj = Path(raw_path)

                                    if not path_obj.is_absolute():
                                        path_obj = workspace_root / path_obj

                                    try:
                                        candidate = path_obj.resolve()
                                        candidate.relative_to(workspace_root)
                                    except (ValueError, OSError):
                                        continue

                                    if candidate.is_file():
                                        relative_path = (
                                            candidate
                                            .relative_to(workspace_root)
                                            .as_posix()
                                        )
                                        found_paths.add(relative_path)


                                # Si la tarea pide localizar un archivo por nombre,
                                # conservar solo los archivos cuyo nombre coincide.
                                requested_file_match = re.search(
                                    r"\b(?:archivo\s+)?([A-Za-z0-9_.-]+\.py)\b",
                                    task,
                                    re.IGNORECASE,
                                )

                                if requested_file_match:
                                    requested_filename = requested_file_match.group(1).lower()

                                    matching_files = {
                                        path
                                        for path in found_paths
                                        if Path(path).name.lower() == requested_filename
                                    }

                                    if matching_files:
                                        found_paths = matching_files

                                files_to_analyze.update(found_paths)

                                print(
                                    "[TRACE] SEARCH_TEXT FILES FOUND: "
                                    f"{len(found_paths)}"
                                )

                                for found_path in sorted(found_paths):
                                    print(
                                        f"[TRACE] SEARCH_TEXT FILE: "
                                        f"{found_path}"
                                    )

                    except Exception as e:
                        self.logger.error(
                            "Error ejecutando herramienta %s: %s",
                            decision.tool_name,
                            e,
                            exc_info=True,
                        )

                        state.results_cache[f"step_{step.id}"] = {
                            "status": "error",
                            "error": str(e),
                        }
                # ---------------------------------------------------------------
                # SIN EJECUCIÓN
                # ---------------------------------------------------------------

                if not tools_executed_in_iteration:

                    if had_critical_error:
                        self.logger.warning(
                            "Se encontraron herramientas prohibidas. "
                            "Deteniendo análisis."
                        )
                        break

                    self.logger.warning(
                        "Ninguna herramienta se ejecutó. "
                        "Replanificando..."
                    )

                    plan_result = None
                    continue

                # ---------------------------------------------------------------
                # PRIORIZACIÓN DE RESULTADOS
                # ---------------------------------------------------------------

                if files_from_planner_directly:
                    all_files_to_process = (
                        files_from_planner_directly
                    )

                else:
                    all_files_to_process = files_to_analyze

                if all_files_to_process:

                    print(
                        "[TRACE] ARCHIVOS ENCONTRADOS "
                        f"({len(all_files_to_process)}). "
                        "SALIENDO DE FASE PLANNER."
                    )

                    break

                # Si hubo herramientas pero no encontramos archivos,
                # volver a planificar.
                plan_result = None

             # ========================================================================
        # FASE 3: BATCHING + ANÁLISIS
        # ========================================================================

        state.set_phase("analyze")
        print("[TRACE] ANALYSIS -> PREPARE BATCHES")

        if initial_paths:
            final_file_list = set(initial_paths)
        elif files_from_planner_directly:
            final_file_list = set(files_from_planner_directly)
        else:
            final_file_list = set(files_to_analyze)

        # Expandir directorios descubiertos dinámicamente.
        expanded_files = set()

        for path in final_file_list:
            if not path.startswith("[DIR]"):
                expanded_files.add(path)
                continue

            directory = (workspace_root / path[5:]).resolve()

            try:
                directory.relative_to(workspace_root)
            except ValueError:
                self.logger.warning("Directorio fuera del workspace rechazado: %s", path)
                continue

            if not directory.is_dir():
                continue

            for file_path in directory.rglob("*"):
                if not file_path.is_file():
                    continue

                relative_path = file_path.relative_to(workspace_root)

                if any(part in EXCLUDED_DIRS for part in relative_path.parts):
                    continue

                expanded_files.add(relative_path.as_posix())

        # Validar rutas y conservar nombres relativos completos.
        valid_files = []

        for relative_path in sorted(expanded_files):
            candidate = (workspace_root / relative_path).resolve()

            try:
                candidate.relative_to(workspace_root)
            except ValueError:
                continue

            if candidate.is_file():
                valid_files.append(relative_path.replace("\\", "/"))

        print(f"[TRACE] ANALYSIS FILES: {len(valid_files)} found.")

        if not valid_files:
            return Message(
                role="assistant",
                content="[ANALYSIS] No se encontraron archivos válidos para analizar."
            )

        # Un archivo único puede enviarse completo.
        # Para múltiples archivos, el batcher agrupa los pequeños y
        # mantiene los grandes separados.
        batches = create_batches(
            workspace_root=workspace_root,
            relative_paths=valid_files,
            max_lines_per_batch=300,
            max_chars_per_batch=40000,
        )

        print(f"[TRACE] ANALYSIS BATCHES CREATED: {len(batches)}")

        analysis_results = {}
        analysis_failures = []
        analysis_incomplete = False

        # La cola permite insertar subdivisiones cuando un batch falla.
        queue = [(batch, 0) for batch in batches]
        completed_batches = 0
        max_retries = 2
        max_split_depth = 5

        async def analyze_batch_streaming(prompt: str) -> str:
            """Consume Ollama por streaming y devuelve el texto acumulado."""
            print("[TRACE] ANALYZE REQUEST: stream=True, think=False")
            response_stream = await self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un analizador experto de código. "
                            "Proporciona análisis técnicos precisos."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                think=False,
                temperature=0.1,
            )

            # Protección por si el cliente devuelve una respuesta normal
            # en vez de un generador de chunks.
            if isinstance(response_stream, dict):
                message = response_stream.get("message", {})
                content = message.get("content", "")
                if content.strip():
                    return content
                raise RuntimeError("Ollama devolvió una respuesta vacía.")

            if not hasattr(response_stream, "__aiter__"):
                raise TypeError("El cliente Ollama no devolvió un stream válido.")

            chunks = []

            async for chunk_text, is_final in response_stream:
                if chunk_text:
                    chunks.append(chunk_text)

            content = "".join(chunks).strip()

            if not content:
                raise RuntimeError("Ollama terminó el stream sin contenido.")

            return content

        while queue:
            batch, split_depth = queue.pop(0)

            # No iniciar otro batch cuando se alcanzó el límite global.
            if time.time() - loop_start_time > self.max_execution_time:
                analysis_incomplete = True
                pending = [(batch, split_depth)] + queue

                for pending_batch, _ in pending:
                    paths = [block.path for block in pending_batch.files]
                    analysis_failures.append(
                        f"No procesado por límite de tiempo: {', '.join(paths)}"
                    )

                print(
                    "[TRACE] TIME LIMIT REACHED: "
                    f"{len(pending)} batches pendientes."
                )
                break

            print(
                f"[TRACE] ANALYSIS BATCH "
                f"{completed_batches + 1}: "
                f"{len(batch.files)} blocks / "
                f"{batch.total_lines} lines / "
                f"{batch.total_chars} chars"
            )

            context_parts = []

            for block in batch.files:
                header = (
                    f"--- Archivo: {block.path} "
                    f"(Líneas {block.start_line}-{block.end_line}) ---\n\n"
                )
                context_parts.append(header + block.lines_content)

            analysis_prompt = f"""
            Analiza el siguiente código en el contexto de la tarea:
            "{task}"

            Archivos de este lote:
            {"".join(context_parts)}

            Instrucciones:
            - Basa tus conclusiones exclusivamente en el código proporcionado.
            - Resume solo los hallazgos relevantes para la tarea.
            - Si hay varios archivos, distingue claramente sus funciones.
            - No afirmes que has revisado archivos o líneas que no están incluidos.
            - Sé preciso y conciso.
            """

            analysis_response = None
            last_error = None

            # Primer intento + hasta dos reintentos.
            for attempt in range(max_retries + 1):
                try:
                    if attempt:
                        print(
                            f"[TRACE] ANALYSIS RETRY "
                            f"{attempt}/{max_retries}"
                        )
                        await asyncio.sleep(attempt)

                    analysis_response = await analyze_batch_streaming(
                        analysis_prompt
                    )
                    last_error = None
                    break

                except Exception as exc:
                    last_error = exc
                    self.logger.warning(
                        "Fallo ANALYZE batch %s, intento %s/%s: %s",
                        completed_batches + 1,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                    )

            # Si tras los reintentos sigue fallando, intentar subdividir.
            if last_error is not None:
                can_split = (
                    len(batch.files) > 1
                    or batch.total_lines > 300
                    or batch.total_chars > 40000
                )

                child_batches = []

                if can_split and split_depth < max_split_depth:
                    child_batches = split_batch_for_retry(
                        batch,
                        max_lines=300,
                    )

                if child_batches:
                    print(
                        "[TRACE] ANALYSIS RECOVERY: "
                        f"subdividiendo batch tras {max_retries + 1} fallos."
                    )
                    queue = [
                        (child, split_depth + 1)
                        for child in child_batches
                    ] + queue
                    continue

                paths = [block.path for block in batch.files]
                failure = (
                    f"No se pudo analizar {', '.join(paths)}: {last_error}"
                )
                analysis_failures.append(failure)
                analysis_incomplete = True

                analysis_results[
                    f"batch_error_{completed_batches + 1}"
                ] = {
                    "files": paths,
                    "content": f"[ANÁLISIS FALLIDO] {failure}",
                    "status": "error",
                }
                continue

            completed_batches += 1

            analysis_results[
                f"batch_{completed_batches}"
            ] = {
                "files": [block.path for block in batch.files],
                "content": analysis_response,
                "status": "success",
            }

        print(
            f"[TRACE] ANALYSIS BATCHES COMPLETED: {completed_batches}"
        )

        if analysis_failures:
            print(
                f"[TRACE] ANALYSIS INCOMPLETE: "
                f"{len(analysis_failures)} batches/blocks fallidos o pendientes."
            )

        # Variables para que FASE 4 pueda informar si el análisis fue parcial:
        # analysis_results, analysis_failures, analysis_incomplete
                # ========================================================================
        # FASE 4: SÍNTESIS FINAL
        # ========================================================================

        state.set_phase("synthesis")
        print("[TRACE] ANALYSIS -> SYNTHESIS")

        synthesis_context_parts = [
            f"Tarea original: {task}"
        ]

        for batch_id, result_data in analysis_results.items():
            files_str = ", ".join(result_data.get("files", []))
            status = result_data.get("status", "success")

            synthesis_context_parts.append(
                f"--- Análisis del lote ({files_str}) "
                f"[Estado: {status}] ---\n"
                f"{result_data.get('content', '')}"
            )

        if analysis_failures:
            synthesis_context_parts.append(
                "--- BLOQUES NO PROCESADOS O FALLIDOS ---\n"
                + "\n".join(f"- {failure}" for failure in analysis_failures)
            )

        if analysis_incomplete:
            synthesis_context_parts.append(
                "--- AVISO DE INTEGRIDAD ---\n"
                "El análisis quedó incompleto. No afirmes que se "
                "ha revisado todo el código. Identifica claramente "
                "las partes no procesadas y limita las conclusiones "
                "a la evidencia disponible."
            )

        if not analysis_results:
            synthesis_context_parts.append(
                "No se obtuvo ningún análisis técnico satisfactorio."
            )

        synthesis_context = "\n\n".join(synthesis_context_parts)

        synthesis_prompt = f"""
Eres un analizador de código.

Responde a la tarea original:
"{task}"

Utiliza únicamente los resultados técnicos proporcionados.

Contexto técnico:
{synthesis_context}

Reglas:
1. No inventes información.
2. Distingue los resultados satisfactorios de los fallidos.
3. Si el análisis está incompleto, adviértelo claramente.
4. No presentes conclusiones parciales como si se hubiera revisado todo.
5. Sé preciso y conciso.
"""

        final_synthesis = await self._call_llm_with_think(
            synthesis_prompt,
            think=False,
            stream=True,
            stream_callback=stream_callback,
        )

        print("[TRACE] ANALYSIS MODE END")

        return Message(
            role="assistant",
            content=final_synthesis
        )
    
    # =========================================================================
    # Helpers para el modo Análisis
    # =========================================================================

    async def _execute_read_tool(self, path: str, state: SessionState) -> Any:
        """Ejecuta read_file de forma segura y guarda en results_cache."""
        try:
            tool_entry = self.router.tool_registry.get_tool('read_file')
            tool_func = tool_entry["fn"]
            
            result = tool_func({"path": path}, permission='read', risk_level='low')
            
            # Normalizar resultado
            obs = self._observe_result(result)
            
            key = f"analysis_read_{path}"
            state.results_cache[key] = obs
            
            return result
            
        except KeyError:
            return {"status": "error", "error": "read_file no disponible"}
# LOCALIZACIÓN: core/orchestrator.py -> Método _call_llm_with_think

    async def _call_llm_with_think(
        self,
        prompt: str,
        think: bool,
        stream: bool = False,
        stream_callback: Optional[Callable[[str, bool], None]] = None,
    ) -> str:
        """Llama al LLM y devuelve siempre el texto completo."""

        if not self.llm_client:
            return "[ERROR] LLM no disponible para análisis."

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un analizador experto de código. "
                    "Proporciona análisis técnicos precisos."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            # CAMBIO AQUÍ: Añadir 'await' si self.llm_client es ahora un LLMProvider o OllamaProvider
            # El contrato define async def chat(...)
            response = await self.llm_client.chat(
                messages=messages,
                stream=stream,
                think=think,
                temperature=0.1,
            )

            # Respuesta completa: compatibilidad con llamadas existentes (no streaming)
            if not hasattr(response, "__aiter__"):
                if isinstance(response, dict):
                    content = (
                        response.get("message", {}).get("content", "")
                        or response.get("content", "")
                    )

                    if stream_callback and content:
                        try:
                            stream_callback(content, True)
                        except Exception:
                            self.logger.exception(
                                "Error enviando respuesta al callback."
                            )

                    return content.strip() or (
                        "[ERROR] El modelo devolvió una respuesta vacía."
                    )
                else:
                     return str(response).strip()

            # Streaming: acumular todos los fragmentos.
            chunks = []

            async for chunk_text, is_final in response:
                if chunk_text:
                    chunks.append(chunk_text)

                if stream_callback and (chunk_text or is_final):
                    try:
                        stream_callback(chunk_text, is_final)
                    except Exception:
                        self.logger.exception(
                            "Error enviando fragmento al callback."
                        )

            content = "".join(chunks).strip()

            return content or (
                "[ERROR] El modelo terminó el stream sin contenido."
            )

        except Exception as e:
            self.logger.exception(
                "Error en llamada LLM con think=%s, stream=%s",
                think,
                stream,
            )
            return f"[ERROR] Fallo al generar análisis: {e}"
                
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