"""
Política de Seguridad (§9)
Define reglas estáticas de permisos y allowlists.
"""

from typing import Dict, List

# Comandos permitidos por defecto (Allowlist) §10
ALLOWED_COMMANDS: List[str] = [
    "ls", "cat", "echo", "pwd", "git", "python", 
    "mkdir", "cp", "mv", "head", "tail", "grep",
    "wc", "sort", "uniq", "find", "date", "whoami", "touch", "ping"
]

# Comandos bloqueados por defecto (Danger List) §10
BLOCKED_COMMANDS_PATTERNS: List[str] = [
    "rm -rf", "rm -r", "sudo", "netcat", "nc ", 
    "curl ", "wget ", "chmod ", "chown ", "dd ",
    "mkfs", "fdisk", "shutdown", "reboot", "poweroff",
    ">/dev/tcp", "<<EOF" # Bloquear redirecciones peligrosas
]

# Mapeo de herramientas a permisos requeridos §9
TOOL_PERMISSIONS: Dict[str, str] = {
    "read_file": "read",
    "list_files": "read", 
    "search_text": "read",
    "execute_command": "execute",
    "create_file": "write",
    "modify_file": "write",
}

# Mapeo de herramientas a riesgo estimado §9
TOOL_RISK_LEVELS: Dict[str, str] = {
    "read_file": "low",
    "list_files": "low", 
    "search_text": "low",
    "execute_command": "high", # Por defecto alto por naturaleza
    "create_file": "medium",
    "modify_file": "medium",
}


class Policy:
    """Encapsula las políticas de seguridad estáticas."""

    @staticmethod
    def is_command_allowed(cmd: str) -> bool:
        """Verifica si un comando está permitido según la allowlist y la denylist. §10"""
        if not cmd or not cmd.strip():
            return False
        
        # 1. Checar bloqueos explícitos primero (seguridad first)
        for pattern in BLOCKED_COMMANDS_PATTERNS:
            if pattern in cmd:
                return False
        
        # 2. Checar allowlist
        base_cmd = cmd.strip().split()[0]
        return base_cmd in ALLOWED_COMMANDS

    @staticmethod
    def get_permission(tool_name: str) -> str:
        """Obtiene el nivel de permiso requerido para una herramienta. §9"""
        return TOOL_PERMISSIONS.get(tool_name, "read")

    @staticmethod
    def get_risk_level(tool_name: str) -> str:
        """Obtiene el nivel de riesgo asociado a una herramienta. §9"""
        return TOOL_RISK_LEVELS.get(tool_name, "low")