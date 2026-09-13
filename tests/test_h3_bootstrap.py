"""Bootstrap must install requirements.txt, not a hardcoded package subset."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import h3_bootstrap as boot  # noqa: E402


class EnsureRequirementsTests(unittest.TestCase):
    def setUp(self) -> None:
        boot._installed_this_process = False

    def test_install_argv_uses_requirements_file(self) -> None:
        req = boot.requirements_file()
        self.assertTrue(req.is_file())
        argv = boot._install_argv(req)
        self.assertIn("-r", argv)
        self.assertEqual(argv[argv.index("-r") + 1], str(req))
        self.assertNotIn("fastapi", argv)
        self.assertNotIn("websockets", argv)

    def test_skips_pip_when_imports_and_stamp_ok(self) -> None:
        req = boot.requirements_file()
        with (
            patch.object(boot, "_imports_ok", return_value=True),
            patch.object(boot, "_stamp_matches", return_value=True),
            patch.object(boot.subprocess, "check_call") as pip,
        ):
            boot.ensure_python_requirements()
        pip.assert_not_called()
        self.assertTrue(boot._installed_this_process)

    def test_installs_requirements_when_import_missing(self) -> None:
        req = boot.requirements_file()
        with (
            patch.object(boot, "_imports_ok", return_value=False),
            patch.object(boot, "_can_import", return_value=True),
            patch.object(boot, "_write_stamp"),
            patch.object(boot.subprocess, "check_call") as pip,
        ):
            boot.ensure_python_requirements()
        pip.assert_called_once()
        argv = pip.call_args[0][0]
        self.assertIn("-r", argv)
        self.assertEqual(argv[argv.index("-r") + 1], str(req))

    def test_reinstalls_when_requirements_txt_changes(self) -> None:
        req = boot.requirements_file()
        with (
            patch.object(boot, "_imports_ok", return_value=True),
            patch.object(boot, "_stamp_matches", return_value=False),
            patch.object(boot, "_can_import", return_value=True),
            patch.object(boot, "_write_stamp"),
            patch.object(boot.subprocess, "check_call") as pip,
        ):
            boot.ensure_python_requirements()
        pip.assert_called_once()
        argv = pip.call_args[0][0]
        self.assertEqual(argv[argv.index("-r") + 1], str(req))


if __name__ == "__main__":
    unittest.main()
