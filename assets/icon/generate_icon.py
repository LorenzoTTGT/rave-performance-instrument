#!/usr/bin/env python3
"""Generate or verify the RAVE Performance Instrument icon PNG derivatives.

The SVG beside this script is the sole artwork source. Rendering uses
rsvg-convert with explicit dimensions so every output is a square RGBA PNG.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import sys
import zlib
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "rave-instrument-icon.svg"
MANIFEST = ROOT / "SHA256SUMS"
SIZES = (16, 32, 64, 128, 256, 512, 1024)


def output_for(directory: Path, size: int) -> Path:
    return directory / f"rave-instrument-icon-{size}.png"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(directory: Path) -> None:
    for size in SIZES:
        output = output_for(directory, size)
        subprocess.run(
            ["rsvg-convert", "--width", str(size), "--height", str(size),
             "--output", str(output), str(SOURCE)],
            check=True,
        )


def png_details(path: Path) -> tuple[int, int, int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError(f"{path}: not a PNG with an IHDR header")
    width, height, bit_depth, colour_type = struct.unpack(">IIBB", data[16:26])
    position, idat = 8, bytearray()
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        if kind == b"IDAT":
            idat.extend(payload)
        position += 12 + length
    return width, height, bit_depth, colour_type, zlib.decompress(idat)


def has_transparent_margin(width: int, height: int, scanlines: bytes) -> bool:
    """Decode the PNG row filters enough to prove the SVG transparent margin survived."""
    stride, position, previous = width * 4, 0, bytearray(width * 4)
    alpha_values: list[int] = []
    for _ in range(height):
        filter_type, filtered = scanlines[position], scanlines[position + 1:position + 1 + stride]
        position += stride + 1
        row = bytearray(stride)
        for index, value in enumerate(filtered):
            left = row[index - 4] if index >= 4 else 0
            above = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 0:
                decoded = value
            elif filter_type == 1:
                decoded = value + left
            elif filter_type == 2:
                decoded = value + above
            elif filter_type == 3:
                decoded = value + ((left + above) // 2)
            elif filter_type == 4:
                candidate = left + above - upper_left
                distances = (abs(candidate - left), abs(candidate - above), abs(candidate - upper_left))
                decoded = value + (left if distances[0] <= distances[1] and distances[0] <= distances[2]
                                   else above if distances[1] <= distances[2] else upper_left)
            else:
                raise ValueError(f"unsupported PNG filter {filter_type}")
            row[index] = decoded & 0xff
        alpha_values.extend(row[3::4])
        previous = row
    return 0 in alpha_values and 255 in alpha_values


def expected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        hashes[name] = digest
    return hashes


def verify(directory: Path, require_manifest: bool) -> None:
    expected = expected_hashes() if require_manifest else {}
    for size in SIZES:
        path = output_for(directory, size)
        width, height, bit_depth, colour_type, scanlines = png_details(path)
        if (width, height) != (size, size):
            raise ValueError(f"{path}: expected {size}x{size}, got {width}x{height}")
        if (bit_depth, colour_type) != (8, 6):
            raise ValueError(f"{path}: expected 8-bit RGBA PNG, got {bit_depth}-bit type {colour_type}")
        if not has_transparent_margin(width, height, scanlines):
            raise ValueError(f"{path}: expected both transparent and opaque pixels")
        if require_manifest and expected.get(path.name) != sha256(path):
            raise ValueError(f"{path}: SHA-256 differs from {MANIFEST.name}")


def write_manifest() -> None:
    MANIFEST.write_text(
        "".join(f"{sha256(output_for(ROOT, size))}  {output_for(ROOT, size).name}\n"
                for size in SIZES),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="render assets and update SHA256SUMS")
    mode.add_argument("--check", action="store_true", help="verify checked-in assets and reproducibility")
    args = parser.parse_args()

    if args.write:
        render(ROOT)
        verify(ROOT, require_manifest=False)
        write_manifest()
        return 0

    verify(ROOT, require_manifest=True)
    with tempfile.TemporaryDirectory(prefix="rave-icon-") as temporary:
        rendered = Path(temporary)
        render(rendered)
        verify(rendered, require_manifest=False)
        for size in SIZES:
            checked_in = output_for(ROOT, size)
            regenerated = output_for(rendered, size)
            if sha256(checked_in) != sha256(regenerated):
                raise ValueError(
                    f"{checked_in.name}: regenerated bytes differ; use --write after reviewing the renderer change"
                )
    print("RAVE icon assets: dimensions, RGBA transparency, hashes, and regeneration match")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"icon asset check failed: {error}", file=sys.stderr)
        raise SystemExit(1)
