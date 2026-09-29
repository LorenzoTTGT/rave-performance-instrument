#!/usr/bin/env python3
"""CTest entry point for deterministic project-owned icon assets."""
from pathlib import Path
import subprocess
import sys

repository = Path(__file__).resolve().parents[1]
raise SystemExit(
    subprocess.call(
        [
            sys.executable,
            str(repository / "assets/icon/generate_icon.py"),
            "--check",
            *[arg for arg in sys.argv[1:] if arg],
        ],
        cwd=repository,
    )
)
