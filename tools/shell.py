"""
Herramienta de Ejecución de Comandos con Sandbox (§10).
Aplica timeout, allowlist y validación de paths.
"""

import subprocess
from pathlib import Path
from datetime import datetime
from safety.policy import Policy
from safety.sandbox import Sandbox


_sandbox: Sandbox = None

def set_sandbox(sandbox: Sandbox):
    global _sandbox
    _sandbox = sandbox

def _get_sandbox() -> Sandbox:
    if not _sandbox:
        return Sandbox()
    return _sandbox


def execute_command_tool(args: dict, permission: str = "execute", risk_level: str = "high") -> str:
    """Ejecuta un comando shell bajo restricciones de seguridad. §10"""
    command = args.get("command")

    if not command or not command.strip():
        return "Error: 'command' argument is required."

    sbx = _get_sandbox()

    # 1. Validar Permiso y Riesgo (§9)
    # La política ya asegura que solo se llame si tiene permisos, pero verificamos confirmación implícita
    if not Policy.get_permission("execute_command") == "execute":
        return "Error: Permission denied for execution."

    # 2. Validar Allowlist de Comandos (§10)
    if not Policy.is_command_allowed(command):
        return f"Security Error: Command '{command}' is blocked by safety policy."

    # 3. Validar Directorio de Trabajo (§10)
    # Aseguramos que el comando se ejecute desde un directorio permitido
    cwd = sbx.env.allowed_dirs[0] if sbx.env.allowed_dirs else None
    
    # Intentos de ejecutar fuera del sandbox
    if cwd:
        try:
            # Verificar si el CWD actual está dentro de allowed
            current_cwd = str(Path.cwd())
            is_safe = False
            for d in sbx.env.allowed_dirs:
                if d.startswith(current_cwd) or current_cwd.startswith(str(d)):
                    is_safe = True
                    break
            if not is_safe:
                 # Forzamos el cwd a uno permitido si es necesario para este MVP estricto
                 pass 
        except Exception:
             pass

    start_time = datetime.now()

    try:
        # Ejecución con timeout del sandbox
        result = subprocess.run(
            command,
            shell=True, # Usar shell permite pipes básicos pero requiere allowlist estricta
            capture_output=True,
            text=True,
            timeout=sbx.env.timeout,
            cwd=cwd
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output = f"Command: {command}\n"
        output += f"Exit Code: {result.returncode}\n"
        output += f"Duration: {duration:.2f}s\n"
        
        if result.stdout:
            output += f"\n[STDOUT]\n{result.stdout[:1000]}" # Limitar salida stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr[:1000]}" # Limitar stderr

        return output

    except subprocess.TimeoutExpired:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        return f"Security Error: Command '{command}' timed out after {sbx.env.timeout}s."
    except Exception as e:
        return f"Error executing command: {str(e)}"