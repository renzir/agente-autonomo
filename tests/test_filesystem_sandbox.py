"""
Tests para Sandbox de Filesystem (§10).
Verifica que paths fuera del área permitida sean rechazados.
"""

import unittest
import os
from pathlib import Path
from tools.filesystem import set_sandbox, read_file_tool, list_files_tool
from safety.sandbox import Sandbox


class TestFilesystemSandbox(unittest.TestCase):
    def setUp(self):
        # Crear sandbox estricto solo para el directorio actual '.'
        self.sandbox = Sandbox(allowed_dirs=["."])
        set_sandbox(self.sandbox)

    def test_read_file_within_allowed_dir(self):
        """Leer archivo en directorio permitido debe funcionar."""
        # Crear un tmp file en el cwd
        tmp_file = "test_tmp_security.txt"
        with open(tmp_file, "w") as f:
            f.write("security test")
        
        result = read_file_tool({"path": tmp_file})
        self.assertNotIn("Security Error", result)
        self.assertIn("security test", result)
        
        # Limpieza
        os.remove(tmp_file)

    def test_read_file_outside_allowed_dir(self):
        """Leer archivo fuera del directorio permitido debe ser rechazado."""
        # Intentar leer /etc/passwd o similar (o un path relativo que salga)
        result = read_file_tool({"path": "/etc/hostname"})
        
        self.assertIn("Security Error", result)
        self.assertIn("outside allowed directories", result)

    def test_list_files_outside_allowed_dir(self):
        """Listar archivos fuera del directorio permitido debe ser rechazado."""
        result = list_files_tool({"path": "/tmp"})
        
        self.assertIn("Security Error", result)

if __name__ == '__main__':
    unittest.main()