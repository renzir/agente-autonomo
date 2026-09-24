"""
Herramientas de Sistema de Archivos con Sandbox (§10).
Valida paths antes de cualquier operación.
"""

import os
from pathlib import Path
from safety.sandbox import Sandbox


# Instancia global compartida (se inyectará desde el orchestrator en producción)
_sandbox: Sandbox = None

def set_sandbox(sandbox: Sandbox):
    """Inyección de dependencia para el sandbox."""
    global _sandbox
    _sandbox = sandbox

def _get_sandbox() -> Sandbox:
    if not _sandbox:
        # Fallback por defecto si no se inyectó
        return Sandbox()
    return _sandbox


def read_file_tool(args: dict, permission: str = "read", risk_level: str = "low") -> str:
    """Lee un archivo. Valida sandbox antes de leer. §10"""
    path_str = args.get("path")
    if not path_str:
        return "Error: 'path' argument is required."

    # Validación Sandbox
    sbx = _get_sandbox()
    if not sbx.validate_path(path_str):
        return f"Security Error: Path '{path_str}' is outside allowed directories."

    full_path = Path(path_str).resolve()
    
    try:
        if not full_path.exists():
            return f"Error: File not found at {full_path}"
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Truncar si es muy grande para el contexto del MVP
        max_len = 4000
        if len(content) > max_len:
            return content[:max_len] + "\n\n... [Content truncated for context limits]"
        
        return content

    except PermissionError:
        return f"Error: Permission denied to read {full_path}"
    except UnicodeDecodeError:
        return f"Error: File {full_path} is not a valid text file."
    except Exception as e:
        return f"Error reading file: {str(e)}"


def list_files_tool(args: dict, permission: str = "read", risk_level: str = "low") -> str:
    """Lista archivos. Valida sandbox antes de listar. §10"""
    path_str = args.get("path", ".")

    # Validación Sandbox
    sbx = _get_sandbox()
    if not sbx.validate_path(path_str):
        return f"Security Error: Path '{path_str}' is outside allowed directories."

    full_path = Path(path_str).resolve()

    try:
        if not full_path.exists():
            return f"Error: Directory not found at {full_path}"
        
        if not full_path.is_dir():
            return f"Error: Path {full_path} is not a directory."

        items = []
        for item in sorted(full_path.iterdir()):
            # Indicador de tipo (f=DIRECTORY, f=FILE)
            type_mark = "D" if item.is_dir() else "F"
            items.append(f"[{type_mark}] {item.name}")
        
        return "\n".join(items) if items else "(Empty directory)"

    except PermissionError:
        return f"Error: Permission denied to list {full_path}"
    except Exception as e:
        return f"Error listing files: {str(e)}"


def create_file_tool(args: dict, permission: str = "write", risk_level: str = "medium") -> str:
    """Crea un nuevo archivo. Valida sandbox antes de escribir. §10"""
    path_str = args.get("path")
    content = args.get("content", "")

    if not path_str:
        return "Error: 'path' argument is required."

    # Validación Sandbox
    sbx = _get_sandbox()
    if not sbx.validate_path(path_str):
        return f"Security Error: Path '{path_str}' is outside allowed directories."

    full_path = Path(path_str).resolve()

    try:
        # Aseguramos que el directorio padre exista
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"File created successfully at {full_path}"

    except PermissionError:
        return f"Error: Permission denied to create {full_path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"


def modify_file_tool(args: dict, permission: str = "write", risk_level: str = "medium") -> str:
    """Modifica/Reemplaza el contenido de un archivo. Valida sandbox antes de escribir. §10"""
    path_str = args.get("path")
    content = args.get("content", "")

    if not path_str:
        return "Error: 'path' argument is required."

    # Validación Sandbox
    sbx = _get_sandbox()
    if not sbx.validate_path(path_str):
        return f"Security Error: Path '{path_str}' is outside allowed directories."

    full_path = Path(path_str).resolve()

    try:
        if not full_path.exists():
            # Si no existe, lo creamos (o devolvemos error según preferencia, aquí optamos por crear para evitar loops del agente)
             return f"File {full_path} does not exist. Use 'create_file' or ensure path is correct."

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return f"File modified successfully at {full_path}"

    except PermissionError:
        return f"Error: Permission denied to modify {full_path}"
    except Exception as e:
        return f"Error modifying file: {str(e)}"