"""
Catálogo Central de Herramientas (§9).
Expone metadatos y funciones para el Orchestrator.
"""

from typing import Dict, Any, Callable
import json
from pydantic import ValidationError

from models.schemas import ToolMetadata
from safety.policy import Policy
from tools.filesystem import read_file_tool, list_files_tool, create_file_tool, modify_file_tool
from tools.search import search_text_tool
from tools.shell import execute_command_tool


class ToolRegistry:
    """Catálogo central de herramientas con validación de inputs y metadata."""

    def __init__(self):
        self._tools: Dict[str, ToolMetadata] = {}
        self._functions: Dict[str, Callable] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Registra las herramientas estándar con sus metadatos."""
        
        definitions = [
            {
                "name": "read_file",
                "description": "Read the content of a file at the specified path.",
                "permission": "read",
                "risk": "low",
                "timeout": 10,
                "cost": 0.0,
                "requires_confirmation": False,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The absolute or relative path to the file."}
                    },
                    "required": ["path"]
                },
                "fn": read_file_tool
            },
            {
                "name": "list_files",
                "description": "List files and directories in the specified directory.",
                "permission": "read",
                "risk": "low",
                "timeout": 10,
                "cost": 0.0,
                "requires_confirmation": False,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The directory path to list."}
                    },
                    "required": ["path"]
                },
                "fn": list_files_tool
            },
            {
                "name": "create_file",
                "description": "Create a new file with the specified content.",
                "permission": "write",
                "risk": "medium",
                "timeout": 10,
                "cost": 0.5,
                "requires_confirmation": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path where to create the file."},
                        "content": {"type": "string", "description": "The content to write into the file."}
                    },
                    "required": ["path", "content"]
                },
                "fn": create_file_tool
            },
            {
                "name": "modify_file",
                "description": "Modify an existing file by appending or overwriting content.",
                "permission": "write",
                "risk": "medium",
                "timeout": 10,
                "cost": 0.5,
                "requires_confirmation": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "The path to the file."},
                        "content": {"type": "string", "description": "The content to write (append or replace)."}
                    },
                    "required": ["path", "content"]
                },
                "fn": modify_file_tool
            },
            {
                "name": "search_text",
                "description": "Search for a text pattern in files within a directory.",
                "permission": "read",
                "risk": "low",
                "timeout": 30,
                "cost": 1.0,
                "requires_confirmation": False,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "The text pattern to search for."},
                        "path": {"type": "string", "description": "The directory path to search in (optional)."}
                    },
                    "required": ["pattern"]
                },
                "fn": search_text_tool
            },
            {
                "name": "execute_command",
                "description": "Execute a shell command in the sandbox environment.",
                "permission": "execute",
                "risk": "high",
                "timeout": 30,
                "cost": 1.5,
                "requires_confirmation": True,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to execute."}
                    },
                    "required": ["command"]
                },
                "fn": execute_command_tool
            }
        ]
        for d in definitions:
            meta = ToolMetadata(
                name=d["name"],
                description=d["description"],
                permission=d["permission"],
                risk=d["risk"],
                timeout=d["timeout"],
                cost=d["cost"],
                requires_confirmation=d["requires_confirmation"],
                input_schema=d["input_schema"]
            )
            self._tools[d["name"]] = meta
            self._functions[d["name"]] = d["fn"]

    def list_tools(self) -> Dict[str, ToolMetadata]:
        """Devuelve todas las herramientas disponibles con sus metadatos."""
        return self._tools.copy()

    def get_tool(self, name: str) -> dict:
        """Devuelve metadata y función ejecutable para una herramienta. §9"""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry.")
        
        return {
            "metadata": self._tools[name],
            "fn": self._functions[name]
        }

    def validate_input(self, tool_name: str, args: dict) -> bool:
        """Valida los argumentos de entrada contra el schema de la herramienta. §9"""
        if tool_name not in self._tools:
            raise KeyError(f"Tool '{tool_name}' not found.")
        
        schema = self._tools[tool_name].input_schema
        if not schema:
            return True # Sin schema definido, se asume válido

        try:
            # Usamos Pydantic para validar contra el schema JSON
            # Creamos un modelo temporal basado en el schema (simplificado para MVP)
            from pydantic import create_model, BaseModel
            
            # Extracción básica de tipos del schema json simple
            fields = {}
            for prop_name, prop_def in schema.get("properties", {}).items():
                p_type = prop_def.get("type", "string")
                # Mapeo simple de tipos JSON a Python
                if p_type == "string":
                    fields[prop_name] = (str, ...)
                elif p_type == "number":
                    fields[prop_name] = (float, ...)
                elif p_type == "integer":
                    fields[prop_name] = (int, ...)
                elif p_type == "boolean":
                    fields[prop_name] = (bool, ...)
                else:
                    fields[prop_name] = (str, ...)

            required = schema.get("required", [])
            
            # Crear clase dinámica
            TempModel = create_model(f'Temp{tool_name.capitalize()}Model', **fields)
            
            # Instanciar con los args recibidos
            instance = TempModel(**args)
            
            # Verificar que existan todos los requeridos en el objeto instanciado (si no tienen default)
            for req in required:
                if req not in args:
                    raise ValueError(f"Missing required argument: {req}")

            return True

        except ValidationError as e:
            raise ValueError(f"Input validation failed for '{tool_name}': {e}")
        except Exception as e:
            # Fallback: si el schema es complejo, solo verificamos keys básicas para MVP
            if "required" in schema:
                for req in schema["required"]:
                    if req not in args:
                        raise ValueError(f"Missing required argument: {req}")
            return True