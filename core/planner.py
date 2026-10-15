"""
core/planner.py - Tarea PARTE 2/5: Núcleo del agente
Divide una tarea compleja en subpasos accionables mediante LLM.
"""
import json
import logging
from typing import List, Optional
from pydantic import BaseModel, Field

# Importar modelos existentes para mantener compatibilidad
try:
    from models.schemas import PlannerRequest, PlannerResponse, Step
except ImportError:
    # Fallback por si los módulos no están en el path
    class PlannerRequest(BaseModel):
        task: str
        context: Optional[str] = None

    class Step(BaseModel):
        id: int = Field(default=0)
        description: str = ""  # ✅ Agregado default vacío
        depends_on: List[int] = Field(default_factory=list)
        tool: str = ""
        args: dict = Field(default_factory=dict)
        expected_output: str = ""

    class PlannerResponse(BaseModel):
        steps: List[Step] = Field(default_factory=list)
        estimated_tokens: int = 0
        max_iterations: int = 0


class Planner:
    """Divide una tarea compleja en subpasos accionables para el agente."""
    print("[TRACE] PLANNER START")
    def __init__(self, llm_client=None, config: dict = None):
        self.logger = logging.getLogger(__name__)
        self.llm_client = llm_client
        # Valores por defecto de configuración
        default_config = {
            'agent': {
                'max_iterations': 10
            }
        }
        self.config = config or default_config
        self.max_iterations = self.config.get('agent', {}).get('max_iterations', 10)

    async def plan(self, task: str, context: Optional[str] = None) -> PlannerResponse:
        """
        Planifica los pasos necesarios para completar una tarea.
        
        Args:
            task: Tarea a planificar
            context: Contexto adicional disponible
            
        Returns:
            PlannerResponse con la lista de pasos
        """
        self.logger.info(f"Planificando tarea: {task[:50]}...")
        
        # Si no hay cliente LLM, usar planificación por defecto
        if not self.llm_client:
            return self._default_planning(task)
            
        try:
            # Usar el cliente LLM para generar un plan
            response = await self._llm_plan(task, context)
            return response
        except Exception as e:
            self.logger.exception("Error al planificar con LLM; usando planificación por defecto")
            return self._default_planning(task)

    async def _llm_plan(self, task: str, context: Optional[str] = None) -> PlannerResponse:
        """Genera un plan usando el cliente LLM. CORREGIDO PARA DEVOLVER JSON."""
        
        # 1. Construir prompt EXPLICITO para forzar JSON
        system_prompt = (
            "Eres un planner de agentes. Tu ÚNICA tarea es devolver un objeto JSON válido. "
            "NO incluyas explicaciones, markdown ni texto fuera del JSON.\n\n"

            "HERRAMIENTAS VÁLIDAS — usa exactamente estos nombres:\n"
            "- create_file\n"
            "- modify_file\n"
            "- read_file\n"
            "- list_files\n"
            "- search_text\n"
            "- execute_command\n\n"
            "- project_tree\n"

            "No inventes herramientas ni uses nombres alternativos como "
            "search, shell, write_file, file_manager o file_writer.\n\n"

            "FORMATO OBLIGATORIO:\n"
            "{\n"
            '  "steps": [\n'
            '    {\n'
            '      "id": 1,\n'
            '      "description": "...",\n'
            '      "depends_on": [],\n'
            '      "tool": "nombre_exacto",\n'
            '      "args": {},\n'
            '      "expected_output": "..."\n'
            '    }\n'
            '  ],\n'
            '  "estimated_tokens": 0,\n'
            '  "max_iterations": 1\n'
            "}\n\n"

            "REGLAS:\n"
            "1. Cada step debe usar una herramienta válida.\n"
            "2. Los argumentos de 'args' deben corresponder exactamente a la herramienta elegida.\n"
            "3. Usa depends_on solo cuando un step dependa de otro.\n"
            "4. Devuelve SOLO JSON válido.\n"
            "5. Nunca inventes nombres de archivos, carpetas, rutas ni recursos que no aparezcan en la tarea o en resultados previos de herramientas.\n"
            "6. Si la tarea requiere descubrir primero archivos, rutas o recursos mediante list_files o search_text, el primer step debe ser únicamente la herramienta de descubrimiento.\n"
            "7. No generes en el mismo plan un read_file u otro step que dependa del resultado de list_files o search_text. Ese resultado todavía no se conoce al crear el plan.\n"
            "8. Después de ejecutar una herramienta de descubrimiento, el sistema volverá a planificar usando los resultados reales. Solo entonces puedes generar los steps que utilicen las rutas encontradas.\n"
            "9. Nunca uses placeholders ni nombres inventados para representar resultados futuros, como 'archivo_identificado_en_paso_1', 'resultado_anterior', '<archivo_encontrado>' o similares.\n"            "   search_text requiere exactamente los argumentos 'pattern' y 'path'.\n"
            "   No uses 'query' ni 'file'.\n"
            "10. Cuando la tarea pida buscar dónde se define, aparece, usa o referencia una clase, función, variable, símbolo o texto concreto, utiliza search_text para localizarlo.\n"
            "11. Si todavía no conoces los archivos necesarios para continuar, no los inventes.\n"
            "12. Para buscar un patrón dentro de una carpeta, usa search_text con 'path' apuntando a esa carpeta.\n"
            "    Para buscar dentro de un archivo concreto, usa también 'path' con la ruta del archivo.\n"
            "    Nunca uses un argumento llamado 'file'.\n"
        )
        
        user_prompt = f"Tarea a realizar: {task}\nContexto disponible: {context or 'Ninguno'}"

        # 2. Llamar al LLM ASINCRONAMENTE usando chat() (ya que generate_completion no existe)
        try:
            print("[TRACE] PLANNER -> OLLAMA")

            response_dict = await self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024, 
                think=False,
                stream=False
            )

            print("[TRACE] PLANNER <- OLLAMA")
            print(f"[TRACE] PLANNER RESPONSE TYPE: {type(response_dict)}")
            print(f"[TRACE] PLANNER RESPONSE: {response_dict}")

            if isinstance(response_dict, dict):
                content = (
                    response_dict.get("message", {}).get("content", "")
                    or response_dict.get("content", "")
                )
            else:
                content = str(response_dict)

            print(f"[TRACE] PLANNER RAW CONTENT: {content}")

            if content.startswith("```"):
                lines = content.split("\n")
                if len(lines) > 1:
                    lines.pop(0)
                    if lines[-1].strip() == "}":
                        lines.pop(-1)
                    content = "\n".join(lines)

            plan_data = json.loads(content)

            if 'steps' not in plan_data:
                raise ValueError("El JSON devuelto no tiene la clave 'steps'")

            return PlannerResponse(**plan_data)   
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Error al parsear respuesta del planner (JSON inválido): {e} | Respuesta cruda: {content[:100]}")
            # Fallback seguro en lugar de romper
            return self._default_planning(task)
        except Exception as e:
            self.logger.exception(
                "Error crítico en _llm_plan: %s: %s",
                type(e).__name__,
                str(e),
            )
            raise  

    def _default_planning(self, task: str) -> PlannerResponse:
        """Planificación por defecto para tareas simples."""
        # Para tareas muy simples, un solo paso es suficiente
        return PlannerResponse(
            steps=[Step(
                id=1,
                description=f"Ejecutar tarea: {task}",
                depends_on=[],
                tool="",  # Se decidirá en el router
                args={},
                expected_output="Resultado de la tarea"
            )],
            estimated_tokens=len(task) * 2,  # Estimación muy básica
            max_iterations=1
        )


    def validate_plan(self, plan: 'PlannerResponse') -> bool:
        """Valida que el plan sea ejecutable."""
        if not plan.steps:
            return False

        # Verificar que no haya IDs duplicados entre pasos
        seen_ids = set()

        for step in plan.steps:
            if step.id in seen_ids:
                return False

            seen_ids.add(step.id)

        return True