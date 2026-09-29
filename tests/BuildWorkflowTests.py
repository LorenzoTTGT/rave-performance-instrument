"""Exercise package integrity and installation failure handling without host writes."""

import json
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import artifacts
from build import read_version


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        binary = self.package / "VST3/Test.vst3/Contents/test.bin"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"plugin")
        self.manifest = {
            "schema": 1,
            "version": "0.1.1",
            "system": platform.system(),
            "files": {"VST3/Test.vst3/Contents/test.bin": artifacts.digest(binary)},
            "artifacts": [
                {
                    "format": "VST3",
                    "path": "VST3/Test.vst3",
                    "binary": "VST3/Test.vst3/Contents/test.bin",
                }
            ],
        }
        self.save()
        self.destination = self.root / "installed"

    def save(self):
        (self.package / "build-identity.json").write_text(json.dumps(self.manifest))

    def test_tampering_and_unlisted_files_rejected(self):
        artifacts.verify_package(self.package)
        binary = self.package / next(iter(self.manifest["files"]))
        binary.write_bytes(b"modified")
        with self.assertRaises(ValueError):
            artifacts.verify_package(self.package)
        binary.write_bytes(b"plugin")
        (binary.parent / "build-identity.json").write_text("{}")
        with self.assertRaises(ValueError):
            artifacts.verify_package(self.package)

    def test_unsafe_paths_rejected(self):
        for name in ("../outside", "/outside", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                artifacts.safe_relative(self.package, name)
        self.manifest["artifacts"][0]["format"] = "../../outside"
        self.save()
        with self.assertRaises(ValueError):
            artifacts.verify_package(self.package)

    def test_explicit_replacement_preserves_unrelated_plugins(self):
        artifacts.install(self.package, self.destination)
        target = self.destination / "VST3/Test.vst3/Contents/test.bin"
        target.write_bytes(b"old")
        unrelated = target.parents[2] / "Other.vst3"
        unrelated.mkdir()
        with self.assertRaises(FileExistsError):
            artifacts.install(self.package, self.destination)
        self.assertEqual(target.read_bytes(), b"old")
        artifacts.install(self.package, self.destination, replace=True)
        self.assertEqual(target.read_bytes(), b"plugin")
        self.assertTrue(unrelated.is_dir())

    def test_failed_replacement_restores_previous_installation(self):
        artifacts.install(self.package, self.destination)
        target = self.destination / "VST3/Test.vst3/Contents/test.bin"
        target.write_bytes(b"old")
        rename = Path.rename

        def fail_new(path, destination):
            if path.name == "new":
                raise OSError("simulated failed replacement")
            return rename(path, destination)

        with patch.object(Path, "rename", fail_new), self.assertRaises(OSError):
            artifacts.install(self.package, self.destination, replace=True)
        self.assertEqual(target.read_bytes(), b"old")

    def test_second_artifact_failure_rolls_back_first(self):
        binary = self.package / "Standalone/App/test.bin"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"app")
        self.manifest["files"]["Standalone/App/test.bin"] = artifacts.digest(binary)
        self.manifest["artifacts"].append(
            {
                "format": "Standalone",
                "path": "Standalone/App",
                "binary": "Standalone/App/test.bin",
            }
        )
        self.save()
        artifacts.install(self.package, self.destination)
        plugin = self.destination / "VST3/Test.vst3/Contents/test.bin"
        plugin.write_bytes(b"old plugin")
        rename = Path.rename

        def fail_second(path, destination):
            if path.name == "new" and Path(destination).parent.name == "Standalone":
                raise OSError("second artifact failed")
            return rename(path, destination)

        with patch.object(Path, "rename", fail_second), self.assertRaises(OSError):
            artifacts.install(self.package, self.destination, replace=True)
        self.assertEqual(plugin.read_bytes(), b"old plugin")
        self.assertEqual(
            (
                self.destination
                / (
                    "Standalone/App/test.bin"
                    if platform.system() == "Darwin"
                    else "Standalone/Standalone/App/test.bin"
                )
            ).read_bytes(),
            b"app",
        )

    def test_copy_failure_does_not_remove_existing_installation(self):
        artifacts.install(self.package, self.destination)
        target = self.destination / "VST3/Test.vst3/Contents/test.bin"
        target.write_bytes(b"old")
        with patch.object(
            artifacts.shutil, "copytree", side_effect=OSError("copy failed")
        ), self.assertRaises(OSError):
            artifacts.install(self.package, self.destination, replace=True)
        self.assertEqual(target.read_bytes(), b"old")

    def test_failed_rollback_keeps_backup(self):
        artifacts.install(self.package, self.destination)
        rename = Path.rename

        def fail_new_and_restore(path, destination):
            if path.name in ("new", "old"):
                raise OSError("simulated filesystem failure")
            return rename(path, destination)

        with patch.object(Path, "rename", fail_new_and_restore), self.assertRaisesRegex(
            RuntimeError, "backup preserved"
        ):
            artifacts.install(self.package, self.destination, replace=True)
        backups = list(
            self.destination.glob("VST3/.rave-install-*/old/Contents/test.bin")
        )
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b"plugin")

    def test_macos_dylib_install_id_is_not_a_dependency(self):
        binary = self.root / "libomp.dylib"
        binary.touch()
        stale_id = "/vendor/build/libomp.dylib"

        def otool(args, **kwargs):
            if args[1] == "-L":
                return f"{binary}:\n{stale_id} (compatibility version 1.0.0)\n/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)"
            if args[1] == "-D":
                return f"{binary}:\n{stale_id}"
            return ""

        with patch.object(artifacts, "run", otool):
            self.assertEqual(
                artifacts.dependencies(
                    binary, {"system": "Darwin", "torch_lib": str(self.root)}
                ),
                {},
            )

    def test_windows_environment_uses_normalized_keys(self):
        with patch.dict(
            artifacts.os.environ,
            {
                "SYSTEMROOT": "C:/Windows",
                "PATH": "developer-torch",
                "LD_LIBRARY_PATH": "developer-libs",
            },
            clear=True,
        ), patch.object(artifacts.os, "name", "nt"):
            env = artifacts.clean_environment()
        self.assertNotIn("developer-torch", env["PATH"])
        self.assertIn("System32", env["PATH"])
        self.assertNotIn("LD_LIBRARY_PATH", env)

    @unittest.skipUnless(platform.system() == "Linux", "Linux ELF path semantics")
    def test_linux_dependency_paths_with_spaces(self):
        library = self.root / "library folder/libtorch.so"
        library.parent.mkdir()
        library.touch()
        output = f"libtorch.so => {library} (0x123456)"
        with patch.object(artifacts, "run", return_value=output):
            self.assertEqual(
                artifacts.dependencies(
                    self.root / "plugin.so",
                    {"system": "Linux", "torch_lib": str(self.root)},
                ),
                {"libtorch.so": library.resolve()},
            )

    def test_model_resources_survive_install_and_replacement(self):
        resource = "VST3/Test.vst3/Contents/Resources/Models/starter.ts"
        model = self.package / resource
        model.parent.mkdir(parents=True)
        model.write_bytes(b"qualified-model")
        self.manifest["files"][resource] = artifacts.digest(model)
        self.save()
        artifacts.install(self.package, self.destination)
        self.assertEqual((self.destination / resource).read_bytes(), b"qualified-model")
        artifacts.install(self.package, self.destination, replace=True)
        self.assertEqual((self.destination / resource).read_bytes(), b"qualified-model")

    def test_version_validation(self):
        for value in ("1.2", "1.2.3\nextra", "1.256.0", "../bad", "01.2.3"):
            (self.root / "VERSION").write_text(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_version(self.root)
        (self.root / "VERSION").write_text("0.1.1\n")
        self.assertEqual(read_version(self.root), "0.1.1")


if __name__ == "__main__":
    unittest.main()
