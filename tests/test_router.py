"""
tests/test_router.py - Tests para core/router.py
Verifica la decisión de enrutamiento de herramientas.
"""

import pytest
from core.router import Router, RouterDecision
from models.schemas import ToolMetadata


class TestRouter:
    """Tests básicos del router."""
    
    def test_create_router(self):
        """Crear instancia del router."""
        router = Router()
        assert router is not None
        assert len(router.priority_order) > 0
    
    def test_route_without_tools(self):
        """Enrutar sin herramientas disponibles (debe devolver decisión vacía)."""
        router = Router()
        decision = router.route("Ejecutar comando")
        
        assert isinstance(decision, RouterDecision)
        assert decision.tool_name == ""
        assert decision.confidence == 0.0


class TestRouterWithTools:
    """Tests del router con herramientas disponibles."""
    
    def _create_test_tools(self):
        """Crea diccionario de herramientas de prueba."""
        return {
            'filesystem': ToolMetadata(
                name='filesystem',
                description='Operaciones de archivo y directorio',
                permission='write',
                risk='medium'
            ),
            'search': ToolMetadata(
                name='search',
                description='Búsqueda en internet',
                permission='read',
                risk='low'
            ),
            'shell': ToolMetadata(
                name='shell',
                description='Ejecutar comandos del sistema',
                permission='execute',
                risk='high'
            ),
            'git': ToolMetadata(
                name='git',
                description='Operaciones con git',
                permission='execute',
                risk='medium'
            )
        }
    
    def test_route_filesystem_intention(self):
        """Enrutar intención de archivos."""
        router = Router(tool_registry=self._create_test_tools())
        
        decision = router.route("Leer archivo de configuración")
        
        assert decision.tool_name == 'filesystem'
        assert decision.confidence > 0
        assert len(decision.reason) > 0
    
    def test_route_search_intention(self):
        """Enrutar intención de búsqueda."""
        router = Router(tool_registry=self._create_test_tools())
        
        decision = router.route("Buscar documentación en internet")
        
        assert decision.tool_name == 'search'
        assert decision.confidence > 0
    
    def test_route_shell_intention(self):
        """Enrutar intención de ejecución."""
        router = Router(tool_registry=self._create_test_tools())
        
        decision = router.route("Ejecutar script bash")
        
        # Puede ser shell o git dependiendo de keywords
        assert decision.tool_name in ['shell', 'git']
        assert decision.confidence > 0
    
    def test_route_git_intention(self):
        """Enrutar intención de git."""
        router = Router(tool_registry=self._create_test_tools())
        
        decision = router.route("Hacer commit y push")
        
        assert decision.tool_name == 'git'
        assert decision.confidence > 0
    
    def test_route_no_match_priority(self):
        """Enrutar sin match por prioridad."""
        tools = {
            'filesystem': ToolMetadata(
                name='filesystem',
                description='Archivo',
                permission='read',
                risk='low'
            )
        }
        router = Router(tool_registry=tools)
        
        # Intentión que no coincide con keywords conocidas
        decision = router.route("Hacer algo especial")
        
        # Debe elegir por prioridad
        assert decision.tool_name == 'filesystem'
        assert decision.confidence < 0.5  # Confianza baja por no match


class TestRouterDecision:
    """Tests para la decisión del router."""
    
    def test_create_decision(self):
        """Crear una decisión válida."""
        decision = RouterDecision(
            tool_name='test_tool',
            args={'key': 'value'},
            confidence=0.9,
            reason="Test reason"
        )
        
        assert decision.tool_name == 'test_tool'
        assert decision.confidence == 0.9
    
    def test_decision_to_dict(self):
        """Convertir decisión a diccionario."""
        decision = RouterDecision(
            tool_name='shell',
            args={'cmd': 'ls'},
            confidence=0.85,
            reason="Keyword match"
        )
        
        d = decision.to_dict()
        
        assert d['tool_name'] == 'shell'
        assert d['args'] == {'cmd': 'ls'}
        assert d['confidence'] == 0.85


class TestRouterUpdateRegistry:
    """Tests para actualización del registry."""
    
    def test_update_registry(self):
        """Actualizar registry de herramientas."""
        router = Router()
        
        new_tools = {
            'browser': ToolMetadata(
                name='browser',
                description='Navegador web',
                permission='read',
                risk='low'
            )
        }
        
        router.update_registry(new_tools)
        
        assert 'browser' in router.tool_registry


class TestEdgeCases:
    """Tests para casos límite."""
    
    def test_empty_intention(self):
        """Enrutar intención vacía."""