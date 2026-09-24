"""
Capa de Aislamiento (Sandbox) §10.
Valida paths y configura entornos restringidos.
"""

import os
from pathlib import Path
from typing import List, Optional


class SandboxEnv:
    """Contenedor de metadatos del entorno sandbox para pasar a herramientas."""
    def __init__(self, allowed_dirs: List[str], timeout: float):
        self.allowed_dirs = [Path(d).resolve() for d in allowed_dirs]
        self.timeout = timeout

    def is_safe_path(self, path_str: str) -> bool:
        """Valida que el path resuelto esté dentro de los directorios permitidos. §10"""
        try:
            # Resolvemos el path absoluto para evitar '..' tricks
            target_path = Path(path_str).resolve()
            
            # Verificamos si está dentro de alguno de los allowed_dirs
            for allowed_dir in self.allowed_dirs:
                # Aseguramos que ambos terminen en separador para comparar prefijos correctamente
                allowed_str = str(allowed_dir) + os.sep
                target_str = str(target_path)
                
                # Verificar que el path sea un prefijo válido (estando dentro del directorio)
                if target_str == str(allowed_dir) or target_str.startswith(allowed_str):
                    return True
            
            return False
        except (ValueError, OSError):
            return False
class Sandbox:
    """Gestiona el entorno de seguridad."""

    def __init__(self, allowed_dirs: List[str] = None, timeout: float = 30.0):
        if allowed_dirs is None:
            allowed_dirs = ["."]
        
        # Normalizamos paths relativos al cwd actual para mayor seguridad
        resolved_dirs = []
        for d in allowed_dirs:
            p = Path(d)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            else:
                p = p.resolve()
            resolved_dirs.append(p)

        self.env = SandboxEnv(resolved_dirs, timeout)

    def validate_path(self, path_str: str) -> bool:
        """Delega la validación de directorio. §10"""
        return self.env.is_safe_path(path_str)

    @property
    def config(self):
        """Exponer metadatos para que tools los usen."""
        return {
            "allowed_dirs": [str(d) for d in self.env.allowed_dirs],
            "timeout": self.env.timeout,
            "sandbox_enabled": True
        }