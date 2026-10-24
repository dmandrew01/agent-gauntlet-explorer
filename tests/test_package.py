"""Tests for submission packaging and local admission pre-validation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_clients import package


def _pilot_admission():
    for parent in Path(__file__).resolve().parents:
        path = parent / ".scratch" / "senthil-sandbox" / "runner" / "admission.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("pilot_admission", path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    return None


class ManifestValidationTests(unittest.TestCase):
    def _manifest(self, **overrides: object) -> dict:
        value = {
            "schemaVersion": 1,
            "kind": "GauntletAgentSubmission",
            "metadata": {"agentId": "sample-agent", "agentName": "Sample Agent"},
            "runtime": {
                "language": "python",
                "pythonVersion": "3.12",
                "workingDirectory": ".",
                "entrypoint": ["python", "agent.py"],
            },
            "dependencies": {"lockfile": "requirements.lock"},
            "artifact": {"files": ["agent.py", "requirements.lock"]},
        }
        value.update(overrides)
        return value

    def test_valid_manifest_is_accepted(self) -> None:
        package.validate_manifest(self._manifest())

    def test_uppercase_agent_id_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["metadata"] = {"agentId": "Sample-Agent", "agentName": "Sample"}
        with self.assertRaisesRegex(package.PackagingError, "agentId"):
            package.validate_manifest(manifest)

    def test_shell_entrypoint_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["runtime"] = {
            "language": "python",
            "pythonVersion": "3.12",
            "workingDirectory": ".",
            "entrypoint": ["bash", "agent.py"],
        }
        with self.assertRaisesRegex(package.PackagingError, "python"):
            package.validate_manifest(manifest)

    def test_undeclared_entrypoint_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["runtime"]["entrypoint"] = ["python", "missing.py"]
        with self.assertRaisesRegex(package.PackagingError, "entrypoint script"):
            package.validate_manifest(manifest)

    def test_python_version_must_be_3_12(self) -> None:
        manifest = self._manifest()
        manifest["runtime"]["pythonVersion"] = "3.11"
        with self.assertRaisesRegex(package.PackagingError, "3.12"):
            package.validate_manifest(manifest)


class DirectoryAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self._temporary.name) / "gauntlet-submission"
        self.directory.mkdir()
        (self.directory / "agent.py").write_text("print('ok')\n", encoding="utf-8")
        (self.directory / "requirements.lock").write_text("demo==1.0\n", encoding="utf-8")
        (self.directory / "submission.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "kind": "GauntletAgentSubmission",
                    "metadata": {"agentId": "sample-agent", "agentName": "Sample"},
                    "runtime": {
                        "language": "python",
                        "pythonVersion": "3.12",
                        "workingDirectory": ".",
                        "entrypoint": ["python", "agent.py"],
                    },
                    "dependencies": {"lockfile": "requirements.lock"},
                    "artifact": {"files": ["agent.py", "requirements.lock"]},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_valid_directory_passes(self) -> None:
        manifest = package.validate_submission_directory(self.directory)
        self.assertEqual(manifest["metadata"]["agentId"], "sample-agent")

    def test_undeclared_file_is_rejected(self) -> None:
        (self.directory / "extra.txt").write_text("nope", encoding="utf-8")
        with self.assertRaisesRegex(package.PackagingError, "undeclared files"):
            package.validate_submission_directory(self.directory)

    def test_missing_declared_file_is_rejected(self) -> None:
        (self.directory / "agent.py").unlink()
        with self.assertRaisesRegex(package.PackagingError, "declared files are missing"):
            package.validate_submission_directory(self.directory)

    def test_env_file_is_rejected_even_when_declared(self) -> None:
        manifest = json.loads((self.directory / "submission.json").read_text())
        manifest["artifact"]["files"].append(".env")
        (self.directory / "submission.json").write_text(json.dumps(manifest))
        (self.directory / ".env").write_text("ARENA_API_KEY=not-a-real-key\n")
        with self.assertRaisesRegex(package.PackagingError, "credential-shaped filename"):
            package.validate_submission_directory(self.directory)

    def test_symlink_is_rejected(self) -> None:
        manifest = json.loads((self.directory / "submission.json").read_text())
        manifest["artifact"]["files"].append("linked.py")
        (self.directory / "submission.json").write_text(json.dumps(manifest))
        (self.directory / "linked.py").symlink_to("agent.py")
        with self.assertRaisesRegex(package.PackagingError, "symlinks"):
            package.validate_submission_directory(self.directory)

    def test_raw_api_key_assignment_is_rejected(self) -> None:
        (self.directory / "agent.py").write_text(
            "ARENA_API_KEY=not-a-real-key\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(package.PackagingError, "raw API key assignment"):
            package.validate_submission_directory(self.directory)

    def test_api_key_name_in_prose_is_allowed(self) -> None:
        (self.directory / "agent.py").write_text(
            '"""Uses the ARENA_API_KEY environment variable.\n\nvalue = elsewhere\n"""\n',
            encoding="utf-8",
        )
        manifest = package.validate_submission_directory(self.directory)
        self.assertEqual(manifest["metadata"]["agentId"], "sample-agent")

    def test_oversized_file_is_rejected(self) -> None:
        (self.directory / "agent.py").write_bytes(b"x" * (package.MAX_FILE_BYTES + 1))
        with self.assertRaisesRegex(package.PackagingError, "file limit"):
            package.validate_submission_directory(self.directory)


class PackagingCliTests(unittest.TestCase):
    def test_packages_python_simple_and_writes_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package.package_submission(
                ROOT,
                output,
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            archive = result["archive"]
            checksum = result["checksum"]
            assert isinstance(directory, Path)
            assert isinstance(archive, Path)
            assert isinstance(checksum, Path)
            manifest = json.loads((directory / "submission.json").read_text())
            self.assertEqual(manifest["kind"], "GauntletAgentSubmission")
            self.assertEqual(manifest["runtime"]["pythonVersion"], "3.12")
            self.assertEqual(
                manifest["runtime"]["entrypoint"],
                ["python", "examples/python_simple/agent.py"],
            )
            self.assertIn("requirements.lock", manifest["artifact"]["files"])
            self.assertIn("arena_clients/package.py", manifest["artifact"]["files"])
            self.assertIn("my_strategy.py", manifest["artifact"]["files"])
            self.assertTrue(archive.is_file())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertTrue(checksum.read_text(encoding="utf-8").startswith(digest))
            with tarfile.open(archive, "r:gz") as handle:
                names = handle.getnames()
            self.assertTrue(all(name == "gauntlet-submission" or name.startswith("gauntlet-submission/") for name in names))
            self.assertIn("gauntlet-submission/submission.json", names)

    def test_cli_requires_identity(self) -> None:
        code = package.main([])
        self.assertEqual(code, 2)

    def test_cli_check_accepts_valid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package.package_submission(
                ROOT,
                output,
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            assert isinstance(directory, Path)
            self.assertEqual(package.main(["--check", str(directory)]), 0)

    def test_pilot_admission_accepts_generated_package_when_reachable(self) -> None:
        admission = _pilot_admission()
        if admission is None:
            self.skipTest("pilot runner/admission.py is not on this machine")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = package.package_submission(
                ROOT,
                output,
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            assert isinstance(directory, Path)
            record = admission.admit(directory)
            self.assertEqual(record["agentId"], "python-simple-reference")
            self.assertTrue(record["artifactDigest"].startswith("sha256:"))


class LocalImportGuardTests(unittest.TestCase):
    """A package must contain every local module its code imports."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.kit = Path(self._temporary.name) / "kit"
        self.output = Path(self._temporary.name) / "out"
        shutil.copytree(ROOT / "arena_clients", self.kit / "arena_clients")
        for name in package.DEFAULT_RUNTIME_FILES:
            shutil.copy(ROOT / name, self.kit / name)
        (self.kit / "requirements").mkdir()
        shutil.copy(ROOT / package.DEFAULT_LOCKFILE_SOURCE, self.kit / package.DEFAULT_LOCKFILE_SOURCE)
        (self.kit / "agents").mkdir()

    def _write(self, relative: str, text: str) -> None:
        path = self.kit / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _package(self, **kwargs):
        return package.package_submission(
            self.kit, self.output, agent_id="guard-test", agent_name="Guard", **kwargs
        )

    def test_unpackaged_local_import_is_rejected_with_the_include_to_add(self) -> None:
        self._write("agents/agent.py", "import helper\n")
        self._write("agents/helper.py", "VALUE = 1\n")

        with self.assertRaises(package.PackagingError) as raised:
            self._package(entrypoint="agents/agent.py")

        message = str(raised.exception)
        self.assertIn("agents/agent.py imports agents/helper.py", message)
        self.assertIn("--include agents/helper.py", message)

    def test_rejected_build_leaves_nothing_that_blocks_the_rerun(self) -> None:
        self._write("agents/agent.py", "from helper import VALUE\n")
        self._write("agents/helper.py", "VALUE = 1\n")
        with self.assertRaises(package.PackagingError):
            self._package(entrypoint="agents/agent.py")
        self.assertFalse((self.output / package.SUBMISSION_DIR_NAME).exists())

        result = self._package(entrypoint="agents/agent.py", extra_files=["agents/helper.py"])
        self.assertIn("agents/helper.py", result["manifest"]["artifact"]["files"])

    def test_local_package_import_at_kit_root_is_checked(self) -> None:
        self._write("agents/agent.py", "import toolbox.extra\n")
        self._write("toolbox/__init__.py", "")

        with self.assertRaises(package.PackagingError) as raised:
            self._package(entrypoint="agents/agent.py")
        self.assertIn("--include toolbox/__init__.py", str(raised.exception))

    def test_import_guarded_by_import_error_is_optional(self) -> None:
        self._write(
            "agents/agent.py",
            "try:\n    import helper\nexcept ImportError:\n    helper = None\n",
        )
        self._write("agents/helper.py", "VALUE = 1\n")
        self._package(entrypoint="agents/agent.py")

    def test_standard_library_and_installed_imports_are_ignored(self) -> None:
        self._write("agents/agent.py", "import json\nimport openai\nfrom pathlib import Path\n")
        self._package(entrypoint="agents/agent.py")

    def test_crewai_example_needs_its_helper_module(self) -> None:
        with self.assertRaises(package.PackagingError) as raised:
            package.package_submission(
                ROOT, self.output, agent_id="crew", agent_name="Crew",
                entrypoint="examples/crewai/agent.py",
            )
        self.assertIn("--include examples/crewai/arena_tools.py", str(raised.exception))

        result = package.package_submission(
            ROOT, self.output, agent_id="crew", agent_name="Crew",
            entrypoint="examples/crewai/agent.py",
            extra_files=["examples/crewai/arena_tools.py"],
            lockfile_source="requirements/requirements-all.lock",
        )
        self.assertIn("examples/crewai/arena_tools.py", result["manifest"]["artifact"]["files"])

    def test_every_other_bundled_example_packages_with_only_its_entrypoint(self) -> None:
        for example in ("python_simple", "python_reference", "langgraph"):
            with self.subTest(example=example):
                package.package_submission(
                    ROOT, self.output, agent_id="ex", agent_name="Ex",
                    entrypoint=f"examples/{example}/agent.py", force=True,
                )


class ReferenceSubmissionTests(unittest.TestCase):
    def test_committed_reference_matches_generated_python_simple_manifest(self) -> None:
        reference = (
            ROOT
            / "examples"
            / "python_simple"
            / "reference_submission"
            / "submission.json"
        )
        self.assertTrue(reference.is_file(), "reference submission.json must ship")
        with tempfile.TemporaryDirectory() as temporary:
            result = package.package_submission(
                ROOT,
                Path(temporary),
                agent_id="python-simple-reference",
                agent_name="Python Simple Reference",
            )
            directory = result["directory"]
            assert isinstance(directory, Path)
            generated = json.loads((directory / "submission.json").read_text())
        expected = json.loads(reference.read_text(encoding="utf-8"))
        self.assertEqual(generated, expected)


if __name__ == "__main__":
    unittest.main()
