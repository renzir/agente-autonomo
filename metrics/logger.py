import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class MetricsLogger:
    """
    Logger persistente que escribe métricas en formato JSONL (JSON Lines).
    
    Cada llamada a .log() añade una línea al archivo especificado.
    Soporta agregación de métricas por tarea mediante .record_task().
    """

    def __init__(self, path: Optional[str] = "metrics/run.jsonl"):
        # Permite que MetricsLogger también se use en modo sin escritura a disco
        self.metrics_history: List[Dict[str, Any]] = []
        
        if path is None:
            self.path = None
        else:
            # ✅ CORRECCIÓN: Convertir a ruta absoluta para evitar problemas con directorios temporales relativos
            self.path = os.path.abspath(path)
            
            # Crear directorio si no existe (ej: metrics/)
            dir_name = os.path.dirname(self.path)
            if dir_name and not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except OSError as e:
                    print(f"Warning: Could not create directory {dir_name}: {e}")

    def log(self, metric_name: str, value_or_dict: Any) -> None:
        """
        Registra una métrica en el historial y/o escribe al archivo JSONL.
        
        Args:
            metric_name: Nombre de la métrica.
            value_or_dict: Valor o diccionario de la métrica.
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metric': metric_name,
            'value': value_or_dict
        }
        self.metrics_history.append(entry)
        
        if self.path is not None:
            try:
                # ✅ CORRECCIÓN: Asegurar que el directorio padre exista antes de abrir
                dir_name = os.path.dirname(self.path)
                if dir_name and not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)

                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, default=str) + '\n')
            except IOError as e:
                print(f"Error writing to metrics file {self.path}: {e}")

    def record_task(
        self, 
        task_success: bool = False,
        tests_passed: int = 0,
        tool_errors: int = 0,
        retrieval_success: bool = True,
        context_tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_seconds: float = 0.0,
        model_load_time: float = 0.0,
        tool_calls: int = 0,
        iterations: int = 0,
        user_corrections: int = 0
    ) -> None:
        """
        Registra un resumen completo de una ejecución de tarea.
        """
        metrics_bundle = {
            "task_success": task_success,
            "tests_passed": tests_passed,
            "tool_errors": tool_errors,
            "retrieval_success": retrieval_success,
            "context_tokens": context_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency_seconds,
            "model_load_time": model_load_time,
            "tool_calls_count": tool_calls,
            "iterations_completed": iterations,
            "user_corrections": user_corrections
        }
        
        self.log("task_summary", metrics_bundle)

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Retorna las métricas almacenadas en memoria."""
        return list(self.metrics_history)

class SimpleMetricsLogger(MetricsLogger):
    """
    Logger compatible con interfaz simple que opera EN MEMORIA cuando path=None.
    
    Cuando path es None, almacena métricas en una lista interna (sin escribir a disco).
    Cuando path se proporciona, escribe JSONL normal y también mantiene historial en memoria.
    """
    
    def __init__(self, path: Optional[str] = None):
        self._in_memory_mode = (path is None)
        self.metrics_history: List[Dict[str, Any]] = []
        
        if self._in_memory_mode:
            # Modo solo memoria: no inicializa la superclase (evita escritura a disco)
            self.path = None
        else:
            # Modo normal: llama al constructor padre para crear directorios y establecer path
            super().__init__(path=path)
    
    def log(self, metric_name: str, value_or_dict: Any) -> None:
        """
        Registra una métrica.
        
        En modo memoria: añade a self.metrics_history.
        En modo disco: llama al padre (escribe JSONL).
        En ambos modos, también mantiene historial en memoria para consistencia.
        """
        entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metric': metric_name,
            'value': value_or_dict
        }
        self.metrics_history.append(entry)
        
        if not self._in_memory_mode:
            if self.path is None:
                return
            
            try:
                dir_name = os.path.dirname(self.path)
                if dir_name and not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)

                with open(self.path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, default=str) + '\n')
            except IOError as e:
                print(f"Error writing to metrics file {self.path}: {e}")

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Retorna las métricas almacenadas en memoria."""
        return list(self.metrics_history)

    def record_task(
        self, 
        task_success: bool = False,
        tests_passed: int = 0,
        tool_errors: int = 0,
        retrieval_success: bool = True,
        context_tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_seconds: float = 0.0,
        model_load_time: float = 0.0,
        tool_calls: int = 0,
        iterations: int = 0,
        user_corrections: int = 0
    ) -> None:
        """
        Registra un resumen completo de una ejecución de tarea (task_success).
        """
        metrics_bundle = {
            "task_success": task_success,
            "tests_passed": tests_passed,
            "tool_errors": tool_errors,
            "retrieval_success": retrieval_success,
            "context_tokens": context_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_seconds": latency_seconds,
            "model_load_time": model_load_time,
            "tool_calls_count": tool_calls,
            "iterations_completed": iterations,
            "user_corrections": user_corrections
        }
        
        self.log("task_summary", metrics_bundle)

    def reset(self):
        """Opcional: Reinicia el logger o limpia archivo si se desea empezar de cero."""
        pass


# Eliminé la segunda definición conflictiva de SimpleMetricsLogger que sobrescribía a la anterior