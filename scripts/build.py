"""Configure, build, test and optionally stage a native RAVE package."""

import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import artifacts
import models


def read_version(source):
    version = (Path(source) / "VERSION").read_text().strip()
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("VERSION must contain MAJOR.MINOR.PATCH")
    if any(int(part) > 255 for part in version.split(".")):
        raise ValueError(
            "Version components must fit JUCE packed version fields (0..255)"
        )
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["Debug", "Release"], default="Release")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--stage", type=Path, help="Fresh package destination; also runs tests"
    )
    parser.add_argument("--without-torch", action="store_true")
    parser.add_argument("--juce-source", type=Path)
    parser.add_argument(
        "--icon-assets-only",
        action="store_true",
        help="Check committed icons without regeneration",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    source = Path(__file__).resolve().parents[1]
    version = read_version(source)
    build = (
        args.build_dir
        or source
        / "build"
        / f"{version}-{platform.system().lower()}-{platform.machine()}-{args.config}"
    ).resolve()
    command = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=" + args.config,
        "-DRAVE_USE_SYSTEM_JUCE=OFF",
        "-DBUILD_TESTING=ON",
        "-DPython3_EXECUTABLE=" + sys.executable,
        "-DRAVE_ENABLE_LIBTORCH=" + ("OFF" if args.without_torch else "ON"),
        "-DRAVE_ICON_REGENERATION=" + ("OFF" if args.icon_assets_only else "ON"),
        "-U",
        "CMAKE_PROJECT_RavePerformanceInstrument_INCLUDE",
    ]
    env = dict(os.environ)
    if not args.without_torch:
        import torch

        command.append("-DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path)
        if os.name == "nt":
            env["PATH"] = (
                str(Path(torch.__file__).parent / "lib") + os.pathsep + env["PATH"]
            )
    if args.juce_source:
        if platform.system() == "Darwin":
            from patch_juce_macos15 import patch

            patch(args.juce_source.resolve())
        command.append(
            "-DFETCHCONTENT_SOURCE_DIR_JUCE=" + str(args.juce_source.resolve())
        )
    if os.name == "nt":
        command += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    subprocess.check_call(command, env=env)
    subprocess.check_call(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            args.config,
            "--parallel",
            str(args.jobs),
        ]
        + (["--", "/p:CL_MPCount=1"] if os.name == "nt" else []),
        env=env,
    )
    if args.test or args.stage:
        subprocess.check_call(
            [
                "ctest",
                "--test-dir",
                str(build),
                "-C",
                args.config,
                "--output-on-failure",
            ],
            env=env,
        )
    if args.stage:
        descriptor = build / f"artifacts-{args.config}.json"
        if not args.without_torch:
            manifest = models.verify(source / "assets/models")
            config = json.loads(descriptor.read_text())
            for model in manifest["models"]:
                subprocess.check_call(
                    [
                        config["model_tester"],
                        "--factory",
                        str(source / "assets/models" / model["file"]),
                    ],
                    env=env,
                )
        artifacts.stage(descriptor, args.stage)
    print(f"Build: {build}")


if __name__ == "__main__":
    try:
        main()
    except (
        ImportError,
        OSError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        sys.exit(str(error))
