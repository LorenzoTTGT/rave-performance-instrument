"""Stage relocatable native artifacts and explicitly install verified packages."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile


def run(args, **kwargs):
    return subprocess.check_output(
        [str(arg) for arg in args], text=True, **kwargs
    ).strip()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def identity(source):
    source = Path(source)
    names = run(
        ["git", "-C", source, "ls-files", "-co", "--exclude-standard", "-z"]
    ).split("\0")
    result = hashlib.sha256()
    for name in sorted(set(filter(None, names))):
        path = source / name
        if path.is_file():
            result.update(name.encode() + b"\0" + digest(path).encode())
    return {
        "revision": run(["git", "-C", source, "rev-parse", "HEAD"]),
        "dirty": bool(run(["git", "-C", source, "status", "--porcelain"])),
        "source_sha256": result.hexdigest(),
    }


def clean_environment():
    env = dict(os.environ)
    for name in (
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "PYTHONPATH",
    ):
        env.pop(name, None)
    if os.name == "nt":
        # os.environ is case-insensitive on Windows; a plain dict is not.
        system_root = env.get("SYSTEMROOT") or env["SystemRoot"]
        env["PATH"] = os.path.join(system_root, "System32") + os.pathsep + system_root
    return env


def mac_rpaths(path):
    return re.findall(
        r"cmd LC_RPATH\n.*?\n\s*path (.*?) \(offset", run(["otool", "-l", path])
    )


def dependencies(binary, config):
    """Return import-name -> resolved non-system runtime; missing imports fail closed."""
    binary = Path(binary)
    system = config["system"]
    torch = Path(config["torch_lib"])
    found = {}
    if system == "Windows":
        dumpbin = Path(config["compiler_path"]).with_name("dumpbin.exe")
        names = re.findall(
            r"^\s+([^\s]+\.dll)\s*$", run([dumpbin, "/DEPENDENTS", binary]), re.M | re.I
        )
        lookup = {
            p.name.lower(): p
            for root in (binary.parent, torch)
            if root.is_dir()
            for p in root.glob("*.dll")
        }
        for name in names:
            if name.lower() in lookup:
                found[name] = lookup[name.lower()].resolve()
            elif (
                not name.lower().startswith(("api-ms-", "ext-ms-"))
                and not (Path(os.environ["SystemRoot"]) / "System32" / name).exists()
            ):
                raise RuntimeError(f"Unresolved Windows import {name} in {binary}")
    elif system == "Darwin":
        names = [
            line.strip().split(" (compatibility")[0]
            for line in run(["otool", "-L", binary]).splitlines()[1:]
        ]
        rpaths = mac_rpaths(binary)
        identifiers = run(["otool", "-D", binary]).splitlines()[1:]
        for name in names:
            if name in identifiers:
                continue
            if name.startswith(("/usr/lib/", "/System/Library/")):
                continue
            candidates = [
                Path(
                    name.replace("@loader_path", str(binary.parent)).replace(
                        "@executable_path", str(binary.parent)
                    )
                )
            ]
            if name.startswith("@rpath/"):
                suffix = name[len("@rpath/") :]
                candidates = [
                    Path(
                        p.replace("@loader_path", str(binary.parent)).replace(
                            "@executable_path", str(binary.parent)
                        )
                    )
                    / suffix
                    for p in rpaths
                ]
                candidates += [torch / suffix]
            resolved = next((p.resolve() for p in candidates if p.is_file()), None)
            if resolved is None:
                raise RuntimeError(f"Unresolved macOS import {name} in {binary}")
            if resolved != binary.resolve():
                found[name] = resolved
    else:
        output = run(["ldd", binary])
        if "not found" in output:
            raise RuntimeError(f"Unresolved Linux imports: {output}")
        for line in output.splitlines():
            match = re.match(r"\s*(\S+) => (/.+?)\s+\(0x", line)
            if match:
                name, filename = match.groups()
                path = Path(filename).resolve()
                if not str(path).startswith(("/usr/lib/", "/lib/")):
                    found[name] = path
    return found


def bundle_runtime(binary, runtime, config):
    runtime.mkdir(parents=True, exist_ok=True)
    graph = {}
    todo = [Path(binary)]
    copies = {}
    while todo:
        original = todo.pop()
        if original in graph:
            continue
        graph[original] = dependencies(original, config)
        for dep in graph[original].values():
            if (
                dep.name in copies
                and copies[dep.name] != dep
                and digest(copies[dep.name]) != digest(dep)
            ):
                raise RuntimeError(f"Conflicting runtime library: {dep.name}")
            copies[dep.name] = dep
            todo.append(dep)
    for name, original in copies.items():
        shutil.copy2(original, runtime / name)
    for original, imports in graph.items():
        target = Path(binary) if original == Path(binary) else runtime / original.name
        if config["system"] == "Linux":
            for name, resolved in imports.items():
                if name != resolved.name:
                    subprocess.check_call(
                        [
                            "patchelf",
                            "--replace-needed",
                            name,
                            resolved.name,
                            str(target),
                        ]
                    )
            relative = os.path.relpath(runtime, target.parent)
            subprocess.check_call(
                [
                    "patchelf",
                    "--set-rpath",
                    "$ORIGIN" if relative == "." else "$ORIGIN/" + relative,
                    str(target),
                ]
            )
        elif config["system"] == "Darwin":
            for name, resolved in imports.items():
                relative = os.path.relpath(runtime / resolved.name, target.parent)
                subprocess.check_call(
                    [
                        "install_name_tool",
                        "-change",
                        name,
                        "@loader_path/" + relative,
                        str(target),
                    ]
                )
            for rpath in mac_rpaths(target):
                if rpath.startswith("/") and not rpath.startswith(
                    ("/System/", "/usr/lib")
                ):
                    subprocess.check_call(
                        ["install_name_tool", "-delete_rpath", rpath, str(target)]
                    )
            if target != Path(binary):
                subprocess.check_call(
                    ["install_name_tool", "-id", "@rpath/" + target.name, str(target)]
                )
            # Sign dependencies before signing the containing app/plugin bundle.
            # Signing its main executable early makes codesign inspect unsigned children.
            if target != Path(binary):
                subprocess.check_call(
                    ["codesign", "--force", "--sign", "-", str(target)],
                    stdout=subprocess.DEVNULL,
                )
    return {name: str(path) for name, path in copies.items()}


def safe_relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe package path: {name}")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Package path escapes root: {name}")
    return result


def verify_package(root):
    root = Path(root)
    manifest = json.loads((root / "build-identity.json").read_text())
    if manifest.get("schema") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty package manifest")
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Package symlinks are not supported")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", manifest.get("version", "")):
        raise ValueError("Invalid package version")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "build-identity.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Package file inventory differs from manifest")
    for name, expected in manifest["files"].items():
        path = safe_relative(root, name)
        if path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Package hash mismatch: {name}")
    formats = set()
    for artifact in manifest["artifacts"]:
        fmt = artifact["format"]
        if fmt not in ("VST3", "AU", "Standalone") or fmt in formats:
            raise ValueError("Invalid or duplicate artifact format")
        formats.add(fmt)
        path = safe_relative(root, artifact["path"])
        binary = safe_relative(root, artifact["binary"])
        if (
            Path(artifact["path"]).parts[0] != fmt
            or not path.exists()
            or not binary.is_file()
        ):
            raise ValueError("Invalid artifact layout")
        if path != binary and not binary.is_relative_to(path):
            raise ValueError("Binary is outside its artifact")
    if not formats:
        raise ValueError("No artifacts in package")
    return manifest


def stage(descriptor, destination):
    config = json.loads(Path(descriptor).read_text())
    if identity(config["source"]) != config["identity"]:
        raise RuntimeError(
            "Sources changed since configuration; configure and build before staging"
        )
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Staging requires a fresh destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".rave-stage-", dir=destination.parent
    ) as scratch:
        root = Path(scratch) / "package"
        root.mkdir()
        entries = []
        libraries = {}
        for artifact in config["artifacts"]:
            source = Path(artifact["path"])
            if not source.exists():
                raise FileNotFoundError(f"Build artifact is missing: {source}")
            target = root / artifact["format"] / source.name
            target.parent.mkdir()
            if source.is_dir():
                shutil.copytree(source, target)
                binary = target / Path(artifact["binary"]).relative_to(source)
            else:
                shutil.copy2(source, target)
                binary = target
            if config["torch_version"]:
                import models

                factory = Path(config["source"]) / "assets/models"
                models.verify(factory)
                resources = (
                    target / "Contents/Resources" if target.is_dir() else target.parent
                )
                shutil.copytree(
                    factory,
                    resources / "Models",
                    ignore=shutil.ignore_patterns("*.part"),
                )
            if config["system"] == "Darwin":
                runtime = target / "Contents/Frameworks"
            elif config["system"] == "Windows" and artifact["format"] == "VST3":
                runtime = binary.parent / "RaveRuntime"
                runtime.mkdir()
                payload = runtime / "RaveImplementation.dll"
                binary.rename(payload)
                shutil.copy2(config["loader"], binary)
                libraries.update(bundle_runtime(payload, runtime, config))
                runtime = None
            else:
                runtime = (
                    binary.parent
                    if config["system"] == "Windows"
                    else binary.parent / "lib"
                )
            if runtime is not None:
                libraries.update(bundle_runtime(binary, runtime, config))
            if config["system"] == "Darwin":
                subprocess.check_call(
                    ["codesign", "--force", "--deep", "--sign", "-", str(target)],
                    stdout=subprocess.DEVNULL,
                )
                subprocess.check_call(["codesign", "--verify", "--strict", str(target)])
            entries.append(
                {
                    "format": artifact["format"],
                    "path": target.relative_to(root).as_posix(),
                    "binary": binary.relative_to(root).as_posix(),
                }
            )
        # Keep the runtime's own redistribution notices, not just its binaries.
        torch_root = Path(config["torch_lib"]).parent
        notices = root / "licenses"
        notices.mkdir()
        license_paths = list(torch_root.glob("*LICENSE*"))
        for distribution in torch_root.parent.glob("torch-*.dist-info"):
            license_paths.extend(distribution.glob("LICENSE*"))
            license_paths.extend(distribution.glob("licenses/*"))
        for path in license_paths:
            if path.is_file():
                shutil.copy2(path, notices / path.name)
        if config["torch_version"] and not list(notices.iterdir()):
            raise RuntimeError("LibTorch license notice missing from the distribution")
        juce_license = Path(config["juce_license"])
        if not juce_license.is_file():
            raise RuntimeError(
                "The selected JUCE source license is required for staging"
            )
        shutil.copy2(juce_license, notices / "JUCE-LICENSE.md")
        juce_source = Path(config["source"]) / "assets/icon/LICENSE.md"
        shutil.copy2(juce_source, notices / "icon-LICENSE.md")
        for name in ("LICENSE", "COPYING.md"):
            shutil.copy2(Path(config["source"]) / name, notices / name)
        # Verify a relocated copy, so no original output location can satisfy imports.
        relocated = Path(scratch) / "relocated package"
        root.rename(relocated)
        vst = next(a for a in entries if a["format"] == "VST3")
        output = Path(scratch) / "moduleinfo.json"
        subprocess.check_call(
            [
                config["scanner"],
                "-create",
                "-version",
                config["version"],
                "-path",
                str(relocated / vst["path"]),
                "-output",
                str(output),
            ],
            env=clean_environment(),
            cwd=scratch,
        )
        metadata = output.read_text()
        scanned_versions = re.findall(r'"Version"\s*:\s*"([^"\n]+)"', metadata)
        if len(scanned_versions) < 3 or any(
            v != config["version"] for v in scanned_versions
        ):
            raise RuntimeError("Scanned version differs from VERSION")
        if config["system"] == "Linux":
            for entry in entries:
                result = run(
                    ["ldd", relocated / entry["binary"]], env=clean_environment()
                )
                if "not found" in result:
                    raise RuntimeError(result)
                for name in libraries:
                    match = re.search(re.escape(name) + r" => (.*?) \(0x", result)
                    if match and not Path(match[1]).resolve().is_relative_to(
                        relocated.resolve()
                    ):
                        raise RuntimeError(
                            f"External runtime dependency remains: {name}"
                        )
        manifest = {
            key: config[key]
            for key in (
                "schema",
                "version",
                "system",
                "architecture",
                "configuration",
                "compiler",
                "torch_version",
                "identity",
            )
        }
        manifest.update(
            artifacts=entries, runtime_libraries=libraries, relocation_scan="passed"
        )
        manifest["files"] = {
            p.relative_to(relocated).as_posix(): digest(p)
            for p in sorted(relocated.rglob("*"))
            if p.is_file()
        }
        (relocated / "build-identity.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        verify_package(relocated)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Destination appeared during staging: {destination}")
        relocated.rename(destination)
    print(f"Verified package: {destination}")


def install(package, install_root=None, replace=False):
    package = Path(package).resolve()
    manifest = verify_package(package)
    if manifest["system"] != platform.system():
        raise ValueError("Cannot install a package for a different operating system")
    plans = []
    for artifact in manifest["artifacts"]:
        fmt = artifact["format"]
        source = package / artifact["path"]
        if fmt == "Standalone" and manifest["system"] != "Darwin":
            source = source.parent
        if install_root:
            target = Path(install_root).resolve() / fmt / source.name
        elif manifest["system"] == "Darwin":
            parent = (
                Path.home()
                / {
                    "VST3": "Library/Audio/Plug-Ins/VST3",
                    "AU": "Library/Audio/Plug-Ins/Components",
                    "Standalone": "Applications",
                }[fmt]
            )
            target = parent / source.name
        elif manifest["system"] == "Windows":
            target = (
                Path(os.environ["CommonProgramFiles"]) / "VST3" / source.name
                if fmt == "VST3"
                else Path(os.environ["LOCALAPPDATA"])
                / "Programs/RAVE Performance Instrument"
            )
        else:
            target = (
                Path.home() / ".vst3" / source.name
                if fmt == "VST3"
                else Path.home()
                / ".local/lib/rave-performance-instrument"
                / manifest["version"]
            )
        if target.is_symlink() or (target.exists() and not replace):
            raise FileExistsError(
                f"Refusing to overwrite {target}; explicit --replace is required"
            )
        if target.resolve().is_relative_to(package) or package.is_relative_to(
            target.resolve()
        ):
            raise ValueError("Installation destination overlaps the package")
        plans.append((source, target))
    prepared = []
    preserve = set()
    installation_started = False
    try:
        for source, target in plans:
            target.parent.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(prefix=".rave-install-", dir=target.parent))
            prepared.append((scratch, target))
            shutil.copytree(source, scratch / "new")
            # Verify every copied byte before replacing any previous installation.
            for path in source.rglob("*"):
                if path.is_file() and manifest["files"][
                    path.relative_to(package).as_posix()
                ] != digest(scratch / "new" / path.relative_to(source)):
                    raise RuntimeError("Installation copy verification failed")
            if manifest["system"] == "Windows":
                # Temporary directories have a private ACL on Windows. Reset it
                # while elevated so the installed tree inherits the destination's
                # normal user/host permissions after the atomic rename.
                subprocess.run(
                    ["icacls", str(scratch), "/reset", "/T", "/Q"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
        installation_started = True
        for scratch, target in prepared:
            if target.is_symlink() or (target.exists() and not replace):
                raise FileExistsError(f"Installation destination appeared: {target}")
            if target.exists():
                target.rename(scratch / "old")
            (scratch / "new").rename(target)
    except BaseException:
        rollback_errors = []
        for scratch, target in reversed(prepared if installation_started else []):
            try:
                # Infer completed renames from the filesystem, including interrupts
                # immediately after a rename returns. Never remove an untouched target.
                if (scratch / "old").exists():
                    if target.exists():
                        shutil.rmtree(target)
                    (scratch / "old").rename(target)
                elif not (scratch / "new").exists() and target.exists():
                    shutil.rmtree(target)
            except OSError as error:
                preserve.add(scratch)
                rollback_errors.append(f"{error}; backup preserved at {scratch}")
        if rollback_errors:
            raise RuntimeError("; ".join(rollback_errors))
        raise
    finally:
        for scratch, _ in prepared:
            if scratch not in preserve:
                shutil.rmtree(scratch)
    for _, target in plans:
        print(f"Installed: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("identity").add_argument("source")
    st = sub.add_parser("stage")
    st.add_argument("descriptor")
    st.add_argument("destination")
    ve = sub.add_parser("verify")
    ve.add_argument("package")
    ins = sub.add_parser("install")
    ins.add_argument("package")
    ins.add_argument("--install-root")
    ins.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.command == "identity":
        print(json.dumps(identity(args.source)))
    elif args.command == "stage":
        stage(args.descriptor, args.destination)
    elif args.command == "verify":
        verify_package(args.package)
        print("Package hashes verified")
    else:
        install(args.package, args.install_root, args.replace)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
