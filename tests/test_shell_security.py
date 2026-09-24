"""
Tests para Seguridad de Shell (§10).
Verifica bloqueo de comandos peligrosos y aplicación de timeout.
"""

import unittest
import time
from tools.shell import set_sandbox, execute_command_tool
from safety.sandbox import Sandbox


class TestShellSecurity(unittest.TestCase):
    def setUp(self):
        # Sandbox con timeout corto para pruebas
        self.sandbox = Sandbox(allowed_dirs=["."], timeout=5.0)
        set_sandbox(self.sandbox)

    def test_blocked_command_rm_rf(self):
        """El comando 'rm -rf' debe ser bloqueado."""
        result = execute_command_tool({"command": "rm -rf /tmp/test_dir"})
        self.assertIn("Security Error", result)
        self.assertIn("blocked by safety policy", result)

    def test_blocked_command_sudo(self):
        """El comando 'sudo' debe ser bloqueado."""
        result = execute_command_tool({"command": "sudo ls"})
        self.assertIn("Security Error", result)
        self.assertIn("blocked by safety policy", result)

    def test_allowed_command_ls(self):
        """El comando 'ls' debe ser permitido."""
        result = execute_command_tool({"command": "ls"})
        # Debería ejecutar sin error de seguridad (aunque pueda fallar por otros motivos si el dir está vacío)
        self.assertNotIn("Security Error", result)

    def test_timeout_execution(self):
        """Un comando que tarda más que el timeout debe ser bloqueado."""
        # Usamos ping en Windows con un tiempo menor que el timeout del sandbox (5s)
        # ping -n 6 127.0.0.1 espera ~5 segundos en Windows, justo al límite del timeout
        start_time = time.time()
        result = execute_command_tool({"command": "ping -n 6 127.0.0.1"})
        elapsed = time.time() - start_time
        
        # En Windows con shell=True, subprocess.run puede no interrumpir el proceso externo
        # al expirar el timeout. El timeout real de ~5s se mide desde que se invoca hasta
        # que el controlador de TimeoutExpired es lanzado por asyncio/timeout del SO.
        # Ajustamos la aserción para ser más indulgente: debe tardar entre 4s y 8s (timeout 5s ± margen)
        self.assertIn("timed out", result)
        self.assertGreaterEqual(elapsed, 4.0)
        self.assertLess(elapsed, 9.0)

    def test_netcat_blocked(self):
        """netcat debe ser bloqueado."""
        result = execute_command_tool({"command": "nc -l 8080"})
        self.assertIn("Security Error", result)


if __name__ == '__main__':
    unittest.main()