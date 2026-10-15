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
    print(f"[TRACE] READ_FILE ARGS: {args}")
    path_str = args.get("path")
    if not path_str:
        return "Error: 'path' argument is required."

    # Validación Sandbox
    sbx = _get_sandbox()
    if not sbx.validate_path(path_str):
        return f"Security Error: Path '{path_str}' is outside allowed directories."

    print(f"[TRACE] CWD: {Path.cwd()}")
    print(f"[TRACE] PATH RECEIVED: {path_str}")
    full_path = Path(path_str).resolve()
    
    try:
        if not full_path.exists():
            return f"Error: File not found at {full_path}"
        
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
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



def project_tree_tool(
    args: dict,
    permission: str = "read",
    risk_level: str = "low"
) -> str:
    """Devuelve el árbol de un proyecto sin leer el contenido de sus archivos."""
    path_str = args.get("path", ".")
    try:
        max_depth = max(1, min(int(args.get("max_depth", 5)), 10))
        max_entries = max(10, min(int(args.get("max_entries", 500)), 2000))
    except (ValueError, TypeError):
        return "Error: max_depth y max_entries deben ser números enteros."

    sbx = _get_sandbox()

    if not sbx.validate_path(path_str):
        return f"Security Error: Path '{path_str}' is outside allowed directories."

    root = Path(path_str).resolve()

    try:
        if not root.exists():
            return f"Error: Directory not found at {root}"
        if not root.is_dir():
            return f"Error: Path {root} is not a directory."

        excluded_dirs = {
            "node_modules", "__pycache__", ".pytest_cache",
            ".mypy_cache", ".ruff_cache", ".tox", "venv",
            ".venv", "dist", "build", ".next", ".astro"
        }

        def ignored(item: Path, is_directory: bool) -> bool:
            # No seguir enlaces ni rutas que salgan del sandbox.
            try:
                resolved = item.resolve()
                if not sbx.validate_path(str(resolved)):
                    return True
            except (OSError, RuntimeError):
                return True

            name = item.name.lower()

            if is_directory:
                # Ignora todas las carpetas ocultas, además de
                # las carpetas generadas indicadas arriba.
                return name.startswith(".") or name in excluded_dirs

            # Evita mostrar archivos que puedan contener secretos.
            return name == ".env" or name.startswith(".env.")

        lines = [f"{root.name or str(root)}"]
        count = 0
        truncated = False

        def walk(directory: Path, prefix: str, depth: int) -> None:
            nonlocal count, truncated

            if depth > max_depth or truncated:
                return

            try:
                items = sorted(
                    directory.iterdir(),
                    key=lambda item: (
                        not item.is_dir(),
                        item.name.lower()
                    )
                )
            except (PermissionError, OSError):
                lines.append(f"{prefix}└── [No se pudo acceder]")
                return

            visible = []
            for item in items:
                try:
                    is_directory = item.is_dir()
                    if ignored(item, is_directory):
                        continue
                    visible.append((item, is_directory))
                except OSError:
                    continue

            for index, (item, is_directory) in enumerate(visible):
                if count >= max_entries:
                    truncated = True
                    lines.append(f"{prefix}└── ... [Límite de entradas alcanzado]")
                    return

                count += 1
                last = index == len(visible) - 1
                connector = "└── " if last else "├── "
                suffix = "/" if is_directory else ""

                lines.append(f"{prefix}{connector}{item.name}{suffix}")

                if is_directory:
                    next_prefix = prefix + ("    " if last else "│   ")
                    if depth < max_depth:
                        walk(item, next_prefix, depth + 1)

                if truncated:
                    return

        walk(root, "", 1)
        return "\n".join(lines)

    except PermissionError:
        return f"Error: Permission denied to list {root}"
    except OSError as e:
        return f"Error listing project tree: {e}"
