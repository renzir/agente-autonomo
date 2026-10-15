import pytest
from core.router import Router, RouterDecision
from models.schemas import ToolMetadata


class MockToolRegistry:
    """Mock del registry para pruebas."""
    
    def __init__(self, tools_dict):
        self.tools = tools_dict

    def get_tool(self, name):
        if name not in self.tools:
            raise KeyError(f"Tool {name} not found")
        return self.tools[name]
    
    def validate_input(self, tool_name, args):
        # Mock vacío para validar entrada sin fallar
        pass

class TestRouter:
    """Tests básicos del router."""
    
    def test_create_router(self):
        """Crear instancia del router."""
        router = Router()
        assert router is not None
        # El Router actual solo inicializa tool_registry, no priority_order
        assert hasattr(router, 'tool_registry')
    
    def test_route_without_tools(self):
        """Enrutar sin herramientas disponibles (debe devolver decisión vacía)."""
        router = Router()
        
        # Mock de un paso con atributo tool vacío o nulo
        class MockStep:
            id = "step1"
            tool = ""
            args = {}
        
        decision = router.route(MockStep())
        
        assert isinstance(decision, RouterDecision)
        assert decision.tool_name == ""
        assert decision.confidence == 0.0


class TestRouterWithTools:
    """Tests del router con herramientas disponibles."""
    
    def _create_test_tools_metadata(self):
        """Crea diccionario de herramientas de prueba (ToolMetadata)."""
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
        # El router llama a self.tool_registry.get_tool(), así que tool_registry debe ser un objeto
        mock_registry = MockToolRegistry(self._create_test_tools_metadata())
        router = Router(tool_registry=mock_registry)
        
        # Simular un paso que ya sabe qué herramienta necesita (ej: desde el Planner)
        class MockStep:
            id = "step2"
            tool = 'filesystem'
            args = {}
            
        decision = router.route(MockStep())
        
        # Dado que filesystem existe en el registry creado, debería devolver confianza alta
        assert decision.tool_name == 'filesystem'
        assert decision.confidence == 1.0
        assert len(decision.reason) > 0
    
  
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
        router = Router()
        
        # Crear un mock para el step con tool vacío
        class MockStep:
            id = "step3"
            tool = ""
            args = {}
            
        decision = router.route(MockStep())
        
        assert decision.tool_name == ""
        assert decision.confidence == 0.0