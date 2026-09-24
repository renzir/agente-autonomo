"""
Tests para ToolRegistry (§9).
Verifica metadatos y validación de inputs.
"""

import unittest
from tools.registry import ToolRegistry
from safety.permissions import PermissionManager

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()

    def test_list_tools_exposes_all(self):
        """list_tools() debe exponer todas las herramientas registradas."""
        tools = self.registry.list_tools()
        self.assertIn("read_file", tools)
        self.assertIn("execute_command", tools)
        self.assertIn("search_text", tools)
        
    def test_get_tool_returns_metadata_and_fn(self):
        """get_tool(name) debe devolver dict con metadata y fn."""
        tool_info = self.registry.get_tool("read_file")
        self.assertIn("metadata", tool_info)
        self.assertIn("fn", tool_info)
        self.assertTrue(callable(tool_info["fn"]))

    def test_validate_input_valid(self):
        """validate_input debe retornar True para inputs correctos."""
        valid_args = {"path": "test.py"}
        is_valid = self.registry.validate_input("read_file", valid_args)
        self.assertTrue(is_valid)

    def test_validate_input_missing_required(self):
        """validate_input debe lanzar ValueError si faltan requeridos."""
        invalid_args = {} # Falta 'path'
        with self.assertRaises(ValueError):
            self.registry.validate_input("read_file", invalid_args)

    def test_get_tool_unknown_raises_keyerror(self):
        """get_tool con nombre inválido debe lanzar KeyError."""
        with self.assertRaises(KeyError):
            self.registry.get_tool("non_existent_tool")

if __name__ == '__main__':
    unittest.main()