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
            self.logger.error(f"Error al planificar con LLM: {str(e)}")
            return self._default_planning(task)

    async def _llm_plan(self, task: str, context: Optional[str] = None) -> PlannerResponse:
        """Genera un plan usando el cliente LLM. CORREGIDO PARA DEVOLVER JSON."""
        
        # 1. Construir prompt EXPLICITO para forzar JSON
        system_prompt = (
            "Eres un planner de agentes. Tu ÚNICA tarea es devolver un objeto JSON válido. "
            "NO incluyas explicaciones, texto de preámbulo ni markdown. Solo el JSON.\n\n"

            "REGLAS OBLIGATORIAS:\n"
            "1. El campo 'tool' SOLO puede contener uno de estos valores exactos:\n"
            "   - \"create_file\"\n"
            "   - \"modify_file\"\n"
            "   - \"read_file\"\n"
            "   - \"list_files\"\n"
            "   - \"search\"\n"
            "   - \"shell\"\n"
            "   - \"git\"\n"
            "2. NO inventes nombres de herramientas. "
            "NO uses valores como \"file_manager\", \"file_writer\", \"write_file\", "
            "\"file_creator\" ni ningún otro nombre.\n"
            "3. El campo 'args' debe contener ÚNICAMENTE los argumentos necesarios "
            "para la herramienta elegida.\n"
            "4. Para \"create_file\", los argumentos EXACTOS son:\n"
            "   {\"path\": \"ruta/al/archivo\", \"content\": \"contenido\"}\n"
            "5. Para \"modify_file\", los argumentos deben usar 'path' y los campos "
            "necesarios para modificar el archivo.\n"
            "6. Para \"read_file\", usa:\n"
            "   {\"path\": \"ruta/al/archivo\"}\n"
            "7. Para \"list_files\", usa los argumentos definidos para listar archivos.\n"
            "8. Para \"search\", usa 'pattern' y los demás argumentos necesarios.\n"
            "9. Para \"shell\", usa únicamente los argumentos necesarios para ejecutar "
            "el comando.\n"
            "10. Si una tarea requiere una herramienta pero no conoces con seguridad "
            "sus argumentos, NO inventes nombres de argumentos. Usa únicamente los "
            "argumentos definidos por este contrato.\n\n"

            "La estructura debe ser EXACTAMENTE:\n"
            "{\n"
            '  "steps": [\n'
            '    {\n'
            '      "id": 1,\n'
            '      "description": "...",\n'
            '      "depends_on": [],\n'
            '      "tool": "create_file",\n'
            '      "args": {"path": "...", "content": "..."},\n'
            '      "expected_output": "..."\n'
            '    }\n'
            '  ],\n'
            '  "estimated_tokens": 0,\n'
            '  "max_iterations": 1\n'
            "}\n\n"

            "Ejemplo válido para crear un archivo:\n"
            "{\n"
            '  "steps": [\n'
            '    {\n'
            '      "id": 1,\n'
            '      "description": "Crear el archivo solicitado",\n'
            '      "depends_on": [],\n'
            '      "tool": "create_file",\n'
            '      "args": {"path": "hola.txt", "content": "estoy probando"},\n'
            '      "expected_output": "Archivo creado correctamente"\n'
            '    }\n'
            '  ],\n'
            '  "estimated_tokens": 0,\n'
            '  "max_iterations": 1\n'
            "}\n\n"

            "Si no puedes planificar la tarea, devuelve exactamente:\n"
            '{"steps": [], "estimated_tokens": 0, "max_iterations": 0}'
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
                max_tokens=1024
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
            self.logger.error(f"Error crítico en _llm_plan: {str(e)}")
            return self._default_planning(task)
        
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
            
        # Si max_iterations no está configurado (0), validar solo estructura básica
        if plan.max_iterations > 0 and len(plan.steps) > plan.max_iterations:
            return False
        
        # Verificar que no haya IDs duplicados entre pasos
        seen_ids = set()
        for step in plan.steps:
            if step.id in seen_ids:
                return False
            seen_ids.add(step.id)
            
        return True


