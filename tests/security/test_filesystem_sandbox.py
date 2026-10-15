
"""
Tests para Sandbox de Filesystem (§10).
Comprueba lectura, traversal, symlinks y protección de escritura.
"""

import tempfile
import unittest
from pathlib import Path

from tools.filesystem import (
    set_sandbox,
    read_file_tool,
    list_files_tool,
    create_file_tool,
    modify_file_tool,
)
from safety.sandbox import Sandbox


class TestFilesystemSandbox(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.allowed = self.base / "workspace"
        self.outside = self.base / "outside"

        self.allowed.mkdir()
        self.outside.mkdir()

        self.sandbox = Sandbox(allowed_dirs=[str(self.allowed)])
        set_sandbox(self.sandbox)

    def tearDown(self):
        set_sandbox(None)
        self.temp.cleanup()

    def test_read_file_within_allowed_dir(self):
        target = self.allowed / "test.txt"
        target.write_text("security test", encoding="utf-8")

        result = read_file_tool({"path": str(target)})

        self.assertNotIn("Security Error", result)
        self.assertIn("security test", result)

    def test_read_file_outside_allowed_dir(self):
        target = self.outside / "secret.txt"
        target.write_text("private", encoding="utf-8")

        result = read_file_tool({"path": str(target)})

        self.assertIn("Security Error", result)

    def test_list_files_outside_allowed_dir(self):
        result = list_files_tool({"path": str(self.outside)})

        self.assertIn("Security Error", result)

    def test_parent_traversal_is_rejected(self):
        target = self.allowed / ".." / "outside" / "secret.txt"
        target.write_text("private", encoding="utf-8")

        result = read_file_tool({"path": str(target)})

        self.assertIn("Security Error", result)

    def test_create_file_outside_sandbox_is_rejected(self):
        target = self.outside / "created.txt"

        result = create_file_tool({
            "path": str(target),
            "content": "must not be written",
        })

        self.assertIn("Security Error", result)
        self.assertFalse(target.exists())

    def test_modify_file_outside_sandbox_is_rejected(self):
        target = self.outside / "existing.txt"
        target.write_text("original", encoding="utf-8")

        result = modify_file_tool({
            "path": str(target),
            "content": "modified",
        })

        self.assertIn("Security Error", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_create_file_inside_sandbox(self):
        target = self.allowed / "created.txt"

        result = create_file_tool({
            "path": str(target),
            "content": "hello",
        })

        self.assertNotIn("Security Error", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "hello")

    def test_symlink_to_outside_is_rejected(self):
        target = self.outside / "secret.txt"
        target.write_text("private", encoding="utf-8")
        link = self.allowed / "outside_link.txt"

        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"No se pudo crear el enlace simbólico: {exc}")

        result = read_file_tool({"path": str(link)})

        self.assertIn("Security Error", result)


if __name__ == "__main__":
    unittest.main()
