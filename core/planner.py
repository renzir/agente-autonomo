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

    def plan(self, task: str, context: Optional[str] = None) -> PlannerResponse:
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
            response = self._llm_plan(task, context)
            return response
        except Exception as e:
            self.logger.error(f"Error al planificar con LLM: {str(e)}")
            return self._default_planning(task)

    def _llm_plan(self, task: str, context: Optional[str] = None) -> PlannerResponse:
        """Genera un plan usando el cliente LLM."""
        # Construir prompt para planificación
        system_prompt = """Eres un planner de agentes. Tu trabajo es dividir una tarea en pasos pequeños y accionables.
Debes considerar qué herramientas pueden ser necesarias y en qué orden ejecutarlas.
Devuelve solo un JSON válido con la estructura: { "steps": [{ "id": número, "description": "string", "depends_on": [lista_de_ids], "tool": "nombre_herramienta", "args": {}, "expected_output": "string" }], "estimated_tokens": número, "max_iterations": número }
"""
        
        user_prompt = f"Tarea a realizar: {task}\nContexto disponible: {context or 'Ninguno'}\nPor favor divide esta tarea en pasos lógicos y accionables."
        
        # Llamar al LLM
        from models.schemas import PlannerRequest
        request = PlannerRequest(
            task=task,
            context=context,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,  # Baja temperatura para planificación determinista
            max_tokens=2048
        )
        
        response = self.llm_client.generate_completion(request)
        
        # Parsear la respuesta del LLM
        try:
            import json
            plan_data = json.loads(response)
            return PlannerResponse(**plan_data)
        except Exception as e:
            self.logger.error(f"Error al parsear respuesta del planner: {str(e)}")
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


