import json
from typing import Dict, Any, List


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Carga todas las métricas desde un archivo JSONL."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Archivo de métricas no encontrado en {path}")
        return []


def compute_success_rate(jsonl_path: str) -> Dict[str, float]:
    """Calcula la tasa de éxito basada en task_summary."""
    metrics = load_metrics(jsonl_path)
    
    # Filtrar solo entradas que son 'task_summary' o tienen clave 'value.task_success'
    summaries = []
    for m in metrics:
        if m.get('metric') == 'task_summary':
            val = m['value']
            if isinstance(val, dict):
                summaries.append(val)
    
    if not summaries:
        return {"success_rate": 0.0, "total_tasks": 0}

    successes = sum(1 for s in summaries if s.get('task_success') is True or s.get('task_success'))
    total = len(summaries)
    
    return {
        "success_rate": (successes / total * 100.0), 
        "total_tasks_evaluated": total,
        "successful_runs": successes,
        "failed_runs": total - successes
    }


def avg_latency(jsonl_path: str) -> Dict[str, float]:
    """Calcula la latencia promedio y por tarea."""
    metrics = load_metrics(jsonl_path)
    
    latency_values = []
    for m in metrics:
        if m.get('metric') == 'task_summary':
            val = m['value']
            if isinstance(val, dict):
                lat = val.get('latency_seconds', 0.0)
                if lat > 0: # Solo contar si hay registro real de tiempo positivo
                    latency_values.append(lat)

    total_latency = sum(latency_values)
    
    return {
        "average_latency_sec": (total_latency / len(latency_values)) if latency_values else 0.0,
        "max_latency_seen": max(latency_values) if latency_values else 0.0,
        "min_latency_seen": min(latency_values) if latency_values else 0.0,
        "samples_counted": len(latency_values)
    }


def tool_error_rate(jsonl_path: str) -> Dict[str, float]:
    """Calcula la tasa de errores en herramientas."""
    metrics = load_metrics(jsonl_path)
    
    total_errors = 0
    task_summaries_with_tool_info = []

    for m in metrics:
        if m.get('metric') == 'task_summary':
            val = m['value']
            if isinstance(val, dict):
                errors_in_task = val.get('tool_errors', 0)
                total_errors += int(errors_in_task)
                
                # Necesitamos saber cuántas llamadas totales hubo para calcular la tasa correctamente si quisiéramos call-level error rate.
                # Pero aquí calcularemos: (Total Errors en tareas / Total Tareas evaluadas que reportaron tool info).
                task_summaries_with_tool_info.append(val)

    total_tasks = len(task_summaries_with_tool_info)
    
    return {
        "total_tool_errors_recorded": total_errors,
        "tasks_reporting_on_tools": total_tasks,
        # Una tasa simple de errores por tarea (no por llamada individual ya que requeriría logueo más fino en cada tool call).
        "avg_error_rate_per_task": (total_errors / total_tasks) if total_tasks > 0 else 0.0 
    }


def get_all_metrics_summary(jsonl_path: str) -> Dict[str, Any]:
    """Devuelve un diccionario consolidado de todas las métricas."""
    return {
        "success_rate": compute_success_rate(jsonl_path),
        "latency_stats": avg_latency(jsonl_path),
        "error_rates": tool_error_rate(jsonl_path)
    }
