"""Carga y gestiona los modelos desde config/models.yaml.

Proporciona acceso a la lista de modelos, sus identificadores y las opciones por defecto 
(temperature, max_tokens, top_p, etc.) que heredarán todas las llamadas al cliente Ollama.
Corresponde a la sección models/model_registry.py del proyecto.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

_DEFAULT_FILE = Path(__file__).parent.parent / "config" / "models.yaml"


class ModelRegistry:
    """Carga y gestiona los modelos desde config/models.yaml."""

    def __init__(self, filepath: str | Path | None = None) -> None:
        self._filepath = Path(filepath) if filepath else _DEFAULT_FILE
        self._data: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ internal
    def _load(self) -> None:
        if not self._filepath.exists():
            raise FileNotFoundError(f"Archivo de config de modelos no encontrado: {self._filepath}")
        with open(self._filepath, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        # Reemplaza completamente con lo cargado del YAML
        self._data.update(loaded)
        # asegura estructura mínima esperada
        self._data.setdefault("models", {})
        self._data.setdefault("defaults", {})

    # ------------------------------------------------------------------ public API
    def get_model_config(self, name: str) -> str | None:
        """Retorna el identificador completo de un modelo por nombre (ej. 'primary', 'coding')."""
        return self._data.get("models", {}).get(name)

    def list_models(self) -> dict[str, str]:
        """Lista todos los modelos disponibles como {alias: identificador}."""
        return dict(self._data.get("models", {}))

    def select_model(self, model_type: str | None = None) -> str:
        """Selecciona un modelo por tipo o usa el configurado en defaults.model.

        Raises:
            ValueError: Si no hay ningún modelo disponible ni default.
        """
        models = self._data.get("models", {})
        defaults = self._data.get("defaults", {})

        if model_type and model_type in models:
            return models[model_type]
        default_alias = defaults.get("model")
        if default_alias and default_alias in models:
            return models[default_alias]
        # fallback al primero registrado
        first_model = next(iter(models), None)
        if first_model:
            return models[first_model]
        raise ValueError("No hay modelos registrados en config/models.yaml")

    def get_defaults(self) -> dict[str, Any]:
        """Retorna los valores por defecto (temperature, max_tokens, top_p, keep_alive_seconds, ...)."""
        return dict(self._data.get("defaults", {}))

    @property
    def raw(self) -> dict[str, Any]:
        """Acceso crudo al diccionario cargado desde YAML (útil para depuración)."""
        return self._data


# ------------------------------------------------------------------ instancia global por conveniencia
DEFAULT_REGISTRY = ModelRegistry()