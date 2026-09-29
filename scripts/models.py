"""Fetch and verify the pinned factory models; no training checkpoints are downloaded."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for model in manifest["models"]:
        path = root / model["file"]
        if Path(model["file"]).name != model["file"]:
            raise ValueError("Invalid model filename")
        if (
            not path.is_file()
            or path.stat().st_size != model["size"]
            or digest(path) != model["sha256"]
        ):
            raise ValueError(
                f"Missing or invalid model: {path}; run scripts/models.py fetch"
            )
    return manifest


def fetch(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    for model in manifest["models"]:
        path = root / model["file"]
        if path.is_file() and digest(path) == model["sha256"]:
            continue
        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(
                model["url"], timeout=60
            ) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            if (
                temporary.stat().st_size != model["size"]
                or digest(temporary) != model["sha256"]
            ):
                raise ValueError(f"Download integrity failure: {model['file']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    verify(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "verify"])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets/models",
    )
    args = parser.parse_args()
    (fetch if args.action == "fetch" else verify)(args.root)
    print("Factory models verified")
