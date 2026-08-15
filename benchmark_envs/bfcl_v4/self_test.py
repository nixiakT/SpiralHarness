#!/usr/bin/env python3
"""Lightweight negative tests for the standalone BFCL verifier."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bfcl_environment_verifier", ROOT / "verify_snapshot.py"
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery invariant
    raise RuntimeError("could not load BFCL environment verifier")
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class VerifierNegativeTests(unittest.TestCase):
    def test_requires_isolated_no_site_no_bytecode_interpreter(self) -> None:
        VERIFIER._require_safe_interpreter_flags()

    def test_bytecode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            cache = root / "pkg/__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.cpython-312.pyc").write_bytes(b"not executable bytecode")
            with self.assertRaisesRegex(VERIFIER.VerificationError, "pre-existing Python bytecode"):
                VERIFIER._reject_bytecode(root, scope="self-test")

    def test_symlink_component_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            real = root / "real"
            real.mkdir()
            link = root / "link"
            try:
                link.symlink_to(real, target_is_directory=True)
            except OSError as exc:  # pragma: no cover - platforms without symlink permission
                self.skipTest(f"symlinks unavailable: {exc}")
            with self.assertRaisesRegex(VERIFIER.VerificationError, "symlink component"):
                VERIFIER._resolve_contained(link, root, label="self-test path")

    def test_parent_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root).resolve()
            with self.assertRaisesRegex(VERIFIER.VerificationError, "outside"):
                VERIFIER._resolve_contained(root.parent, root, label="self-test escape")

    def test_control_files_and_lock_are_sealed(self) -> None:
        manifest = VERIFIER._load_manifest()
        fingerprint = VERIFIER._verify_control_files(manifest)
        self.assertRegex(fingerprint, r"^[0-9a-f]{64}$")
        self.assertEqual(VERIFIER._sha256(VERIFIER.LOCK_PATH), VERIFIER.EXPECTED_LOCK_SHA256)

    def test_every_uv_sync_pins_the_project_environment(self) -> None:
        launcher = 'UV_PROJECT_ENVIRONMENT="$VIRTUALENV_ROOT" "$UV_EXECUTABLE" sync'
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        verify = (ROOT / "verify.sh").read_text(encoding="utf-8")
        self.assertEqual(setup.count(launcher), 2)
        self.assertEqual(setup.count('"$UV_EXECUTABLE" sync'), 2)
        self.assertEqual(verify.count(launcher), 1)
        self.assertEqual(verify.count('"$UV_EXECUTABLE" sync'), 1)

    def test_setup_treats_dangling_symlinks_as_existing(self) -> None:
        setup = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertIn("[[ -e $candidate || -L $candidate ]]", setup)
        guarded_paths = (
            'path_exists_or_symlink "$SOURCE_ROOT"',
            'path_exists_or_symlink "$ENVIRONMENT_ROOT/.source"',
            'path_exists_or_symlink "$VIRTUALENV_ROOT"',
            'path_exists_or_symlink "$BUILD_MARKER"',
        )
        for guard in guarded_paths:
            with self.subTest(guard=guard):
                self.assertIn(guard, setup)
        with tempfile.TemporaryDirectory() as raw_root:
            dangling = Path(raw_root) / "dangling"
            try:
                dangling.symlink_to(Path(raw_root) / "absent")
            except OSError as exc:  # pragma: no cover - platforms without symlink permission
                self.skipTest(f"symlinks unavailable: {exc}")
            self.assertTrue(dangling.is_symlink())
            self.assertFalse(dangling.exists())
            completed = subprocess.run(
                [
                    "/usr/bin/bash",
                    "-c",
                    "candidate=$1; [[ -e $candidate || -L $candidate ]]",
                    "bfcl-lexistence-self-test",
                    str(dangling),
                ],
                check=False,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                timeout=5,
            )
            self.assertEqual(completed.returncode, 0)


def main() -> int:
    if not sys.flags.isolated or not sys.flags.no_site or not sys.flags.dont_write_bytecode:
        print("self-test requires Python flags -I -S -B", file=sys.stderr)
        return 2
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VerifierNegativeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "kind": "bfcl-v4-verifier-self-test",
                "negative_tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "dependency_environment_attested": False,
                "network_isolation_attested": False,
                "reportable_result": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
