"""
Registry y herramientas del Agente.
Corresponde a la sección tools/.
"""

from .registry import ToolRegistry
from .filesystem import read_file_tool, list_files_tool, create_file_tool, modify_file_tool
from .search import search_text_tool
from .shell import execute_command_tool

__all__ = [
    "ToolRegistry",
    "read_file_tool", "list_files_tool", "create_file_tool", "modify_file_tool",
    "search_text_tool", 
    "execute_command_tool"
]