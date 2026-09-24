"""
Herramienta de Búsqueda Textual (§10).
Usa ripgrep (rg) si está disponible, sino fallback con glob.
"""

import os
import glob
from pathlib import Path
from safety.sandbox import Sandbox


_sandbox: Sandbox = None

def set_sandbox(sandbox: Sandbox):
    global _sandbox
    _sandbox = sandbox

def _get_sandbox() -> Sandbox:
    if not _sandbox:
        return Sandbox()
    return _sandbox


def search_text_tool(args: dict, permission: str = "read", risk_level: str = "low") -> str:
    """Busca un patrón de texto en archivos. §10"""
    pattern = args.get("pattern")
    path_str = args.get("path", ".")

    if not pattern:
        return "Error: 'pattern' argument is required."

    # Validación Sandbox para el directorio base
    sbx = _get_sandbox()
    base_path = Path(path_str).resolve()
    
    if not sbx.validate_path(str(base_path)):
        return f"Security Error: Search path '{path_str}' is outside allowed directories."

    try:
        # Intentar usar ripgrep (rg) por rendimiento y seguridad
        import subprocess
        rg_cmd = ["rg", "--no-heading", "--line-number", pattern, str(base_path)]
        
        # Ejecutar con timeout del sandbox
        result = subprocess.run(
            rg_cmd,
            capture_output=True,
            text=True,
            timeout=sbx.env.timeout,
            cwd=str(base_path) # Asegurar contexto seguro
        )

        if result.returncode == 0:
            return result.stdout if result.stdout else "No matches found."
        elif result.returncode == 1:
            return "No matches found."
        else:
            # Error de rg (ej: binario, permisos) -> Fallback a Python
            pass

    except FileNotFoundError:
        pass # rg no disponible, usar fallback
    except subprocess.TimeoutExpired:
        return f"Error: Search timed out after {sbx.env.timeout}s."
    except Exception as e:
        return f"Error executing search: {str(e)}"

    # Fallback con glob/os.read (más lento pero universal)
    matches = []
    try:
        # Buscar archivos de código común
        extensions = ['*.py', '*.js', '*.ts', '*.txt', '*.md', '*.json', '*.yaml', '*.yml']
        
        for ext in extensions:
            glob_path = base_path / '**' / ext
            for file_path in glob.glob(str(glob_path), recursive=True):
                p = Path(file_path)
                # Validar cada archivo encontrado también (seguridad extra)
                if not sbx.validate_path(str(p)):
                    continue
                
                try:
                    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        if pattern.lower() in line.lower():
                            rel_path = p.relative_to(base_path) if base_path in p.parents else p.name
                            matches.append(f"{rel_path}:{i}: {line.strip()}")
                            
                            # Limitar resultados para no saturar contexto MVP
                            if len(matches) >= 50:
                                break
                except (PermissionError, IOError):
                    continue
            if matches:
                break
                
    except Exception as e:
        return f"Error in fallback search: {str(e)}"

    if matches:
        return "Match found:\n" + "\n".join(matches[:10]) # Mostrar top 10
    return "No matches found."