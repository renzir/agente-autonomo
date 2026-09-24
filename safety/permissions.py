"""
Gestor de Permisos y Confirmaciones (§9).
Determina qué herramientas requieren aprobación humana vs automática.
"""

from .policy import Policy


class PermissionManager:
    """Lógica para decidir si una operación procede."""

    @staticmethod
    def requires_confirmation(tool_name: str) -> bool:
        """Determina si se requiere confirmación explícita del usuario. §9"""
        permission = Policy.get_permission(tool_name)
        risk = Policy.get_risk_level(tool_name)
        
        # Regla: 'execute' o riesgo 'high' requieren confirmación siempre
        if permission == "execute":
            return True
        
        # Regla: 'write' con riesgo 'medium' o alto requiere confirmación
        if permission == "write" and risk in ["medium", "high"]:
            return True
            
        return False

    @staticmethod
    def check_permission(tool_name: str, user_granted: bool = True) -> bool:
        """Verifica si el permiso es válido. Si requiere confirmación, verifica user_granted."""
        if PermissionManager.requires_confirmation(tool_name):
            return user_granted
        return True # Operaciones auto-permitidas (ej: read low risk)