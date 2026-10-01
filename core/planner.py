"""
core/planner.py - Tarea PARTE 2/5: Núcleo del agente
Divide una tarea compleja en subpasos accionables mediante LLM.
"""

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
            "NO incluyas explicaciones, texto preámbulo ni markdown. Solo el JSON.\n"
            "La estructura debe ser exactamente:\n"
            "{\n"
            '  "steps": [\n'
            '    {"id": 1, "description": "...", "depends_on": [], "tool": "", "args": {}, "expected_output": "..."}\n'
            "  ],\n"
            '  "estimated_tokens": 0,\n'
            '  "max_iterations": 1\n'
            "}\n"
            "Si no puedes planificar, devuelve: {\"steps\": [], \"estimated_tokens\": 0, \"max_iterations\": 0}"
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
                temperature=0.1, # Temperatura muy baja para ser preciso con el JSON
                max_tokens=1024
            )
            
            # 3. Extraer el contenido de texto del response
            if isinstance(response_dict, dict):
                content = response_dict.get("message", {}).get("content", "") or response_dict.get("content", "")
            else:
                content = str(response_dict)

            # 4. Limpiar posibles marcas de markdown (```json ... ```) que el LLM a veces añade
            if content.startswith("```"):
                lines = content.split("\n")
                # Eliminar la primera línea (```json o ```) y la última (```)
                if len(lines) > 1:
                    lines.pop(0)
                    if lines[-1].strip() == "}":
                        lines.pop(-1)
                    content = "\n".join(lines)
            
            # 5. Parsear JSON
            import json
            plan_data = json.loads(content)
            
            # Validar que tenga la estructura básica
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


