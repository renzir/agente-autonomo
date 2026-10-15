
"""
Herramienta de búsqueda textual.
Usa ripgrep (rg) cuando está disponible y expresiones regulares
con Python como alternativa.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from safety.sandbox import Sandbox


_sandbox: Optional[Sandbox] = None

MAX_FILES = 100
MAX_MATCHES_PER_FILE = 3
MAX_OUTPUT_LINES = 300

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "build",
    "dist",
}

SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".yaml", ".yml", ".toml",
    ".md", ".txt", ".html", ".css",
    ".scss", ".java", ".cs", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".ini",
}


def set_sandbox(sandbox: Sandbox) -> None:
    global _sandbox
    _sandbox = sandbox


def _get_sandbox() -> Sandbox:
    if _sandbox is None:
        return Sandbox()
    return _sandbox


def _get_workspace_root() -> Path:
    """Localiza la raíz del proyecto a partir de este módulo."""
    module_path = Path(__file__).resolve()

    for candidate in module_path.parents:
        if (candidate / "core").is_dir():
            return candidate

    return Path.cwd().resolve()


def _relative_path(path: Path, workspace_root: Path) -> Optional[str]:
    """Devuelve una ruta relativa segura dentro del workspace."""
    try:
        resolved = path.resolve()
        return resolved.relative_to(workspace_root).as_posix()
    except (ValueError, OSError):
        return None


def _add_match(
    matches_by_file: dict[str, list[str]],
    path: str,
    line_number: int,
    content: str,
) -> bool:
    """Añade una coincidencia respetando los límites por archivo."""
    if path not in matches_by_file:
        if len(matches_by_file) >= MAX_FILES:
            return False
        matches_by_file[path] = []

    file_matches = matches_by_file[path]

    if len(file_matches) < MAX_MATCHES_PER_FILE:
        content = content.rstrip("\r\n")
        file_matches.append(f"{path}:{line_number}: {content}")

    return True


def _format_results(
    matches_by_file: dict[str, list[str]],
    truncated: bool = False,
) -> str:
    """Prioriza una coincidencia de cada archivo antes de añadir extras."""
    if not matches_by_file:
        return "No matches found."

    output = ["Match found:"]

    # Primero: al menos una coincidencia por archivo.
    paths = sorted(matches_by_file)

    for path in paths:
        file_matches = matches_by_file[path]
        if file_matches:
            output.append(file_matches[0])

    # Después: coincidencias adicionales, sin ocultar otros archivos.
    for path in paths:
        for match in matches_by_file[path][1:]:
            if len(output) >= MAX_OUTPUT_LINES:
                truncated = True
                break
            output.append(match)

        if len(output) >= MAX_OUTPUT_LINES:
            truncated = True
            break

    if truncated:
        output.append(
            "[TRUNCATED] Se alcanzó el límite de resultados. "
            "Refina la búsqueda o limita la carpeta."
        )

    return "\n".join(output)


def _search_with_rg(
    pattern: str,
    base_path: Path,
    workspace_root: Path,
    sandbox: Sandbox,
) -> Optional[str]:
    """
    Busca con ripgrep en formato JSON para obtener rutas y líneas
    sin tener que interpretar manualmente los dos puntos de las rutas.
    Devuelve None si rg no está disponible o no puede completar la búsqueda.
    """
    if base_path.is_file():
        search_cwd = base_path.parent
        target = base_path.name
    else:
        search_cwd = base_path
        target = "."

    excluded_globs = [
        "!**/.git/**",
        "!**/.venv/**",
        "!**/venv/**",
        "!**/node_modules/**",
        "!**/__pycache__/**",
        "!**/.pytest_cache/**",
        "!**/.mypy_cache/**",
        "!**/build/**",
        "!**/dist/**",
    ]

    command = [
        "rg",
        "--json",
        "--line-number",
        "--color", "never",
        "--ignore-case",
        "--max-count", str(MAX_MATCHES_PER_FILE),
    ]

    for glob_pattern in excluded_globs:
        command.extend(["--glob", glob_pattern])

    command.extend([
        "--regexp", pattern,
        "--",
        target,
    ])

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=sandbox.env.timeout,
            cwd=str(search_cwd),
        )

    except FileNotFoundError:
        return None

    except subprocess.TimeoutExpired:
        return (
            f"Error: Search timed out after "
            f"{sandbox.env.timeout}s."
        )

    except Exception:
        return None

    if result.returncode == 1:
        return "No matches found."

    if result.returncode not in (0,):
        # Se intentará el fallback de Python.
        return None

    matches_by_file: dict[str, list[str]] = {}
    truncated = False

    for raw_line in result.stdout.splitlines():
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if event.get("type") != "match":
            continue

        data = event.get("data", {})
        path_data = data.get("path", {})
        raw_path = path_data.get("text")

        if not raw_path:
            continue

        path_obj = Path(raw_path)

        if not path_obj.is_absolute():
            path_obj = search_cwd / path_obj

        try:
            resolved_path = path_obj.resolve()
        except OSError:
            continue

        if not sandbox.validate_path(str(resolved_path)):
            continue

        relative_path = _relative_path(
            resolved_path,
            workspace_root,
        )

        # El Orchestrator necesita rutas relativas al workspace.
        if relative_path is None:
            continue

        line_number = data.get("line_number", 0)
        line_content = data.get("lines", {}).get("text", "")

        was_new_file = relative_path not in matches_by_file

        if was_new_file and len(matches_by_file) >= MAX_FILES:
            truncated = True
            continue

        _add_match(
            matches_by_file,
            relative_path,
            line_number,
            line_content,
        )

    return _format_results(matches_by_file, truncated)


def _search_with_python(
    regex: re.Pattern,
    base_path: Path,
    workspace_root: Path,
    sandbox: Sandbox,
) -> str:
    """Fallback con las mismas reglas de expresión regular."""
    matches_by_file: dict[str, list[str]] = {}
    truncated = False

    if base_path.is_file():
        candidate_files = [base_path]
    else:
        candidate_files = []

        for current_dir, dirnames, filenames in os.walk(
            base_path,
            topdown=True,
            followlinks=False,
        ):
            current_path = Path(current_dir)

            dirnames[:] = sorted(
                name for name in dirnames
                if name not in EXCLUDED_DIRS
                and not (current_path / name).is_symlink()
            )

            for filename in sorted(filenames):
                file_path = current_path / filename

                if file_path.is_symlink():
                    continue

                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                candidate_files.append(file_path)

                if len(candidate_files) >= 10000:
                    truncated = True
                    break

            if len(candidate_files) >= 10000:
                break

    for file_path in candidate_files:
        try:
            resolved_path = file_path.resolve()

            if not sandbox.validate_path(str(resolved_path)):
                continue

            relative_path = _relative_path(
                resolved_path,
                workspace_root,
            )

            if relative_path is None:
                continue

            for line_number, line in enumerate(
                resolved_path.open(
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ),
                start=1,
            ):
                if regex.search(line):
                    if (
                        relative_path not in matches_by_file
                        and len(matches_by_file) >= MAX_FILES
                    ):
                        truncated = True
                        break

                    _add_match(
                        matches_by_file,
                        relative_path,
                        line_number,
                        line,
                    )

                    if (
                        len(matches_by_file[relative_path])
                        >= MAX_MATCHES_PER_FILE
                    ):
                        break

        except (PermissionError, OSError):
            continue

    return _format_results(matches_by_file, truncated)


def search_text_tool(
    args: dict,
    permission: str = "read",
    risk_level: str = "low",
) -> str:
    """Busca un patrón regex en archivos del workspace."""
    pattern = args.get("pattern")
    path_str = args.get("path", ".")

    if not isinstance(pattern, str) or not pattern.strip():
        return "Error: 'pattern' argument is required."

    sandbox = _get_sandbox()
    workspace_root = _get_workspace_root()

    try:
        # Admitir expresiones regulares también en el fallback.
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Error: Invalid search pattern: {exc}"

    try:
        requested_path = Path(path_str)

        if requested_path.is_absolute():
            base_path = requested_path.resolve()
        else:
            base_path = (workspace_root / requested_path).resolve()

        # No permitir salir del workspace.
        base_path.relative_to(workspace_root)

    except ValueError:
        return (
            f"Security Error: Search path "
            f"'{path_str}' is outside the workspace."
        )

    except (OSError, TypeError) as exc:
        return f"Error: Invalid search path: {exc}"

    if not base_path.exists():
        return f"Error: Search path '{path_str}' does not exist."

    if not sandbox.validate_path(str(base_path)):
        return (
            f"Security Error: Search path "
            f"'{path_str}' is outside allowed directories."
        )

    # Intentar primero ripgrep.
    rg_result = _search_with_rg(
        pattern,
        base_path,
        workspace_root,
        sandbox,
    )

    if rg_result is not None:
        return rg_result

    # Si rg no está disponible o no puede procesar la búsqueda,
    # usar Python con expresiones regulares.
    try:
        return _search_with_python(
            regex,
            base_path,
            workspace_root,
            sandbox,
        )
    except Exception as exc:
        return f"Error in fallback search: {exc}"