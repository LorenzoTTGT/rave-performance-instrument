#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import tempfile

import torch


class IdentityModel(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio


class RaveLikeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("forward_params", torch.tensor([1, 1, 1, 1]))
        self.register_buffer("encode_params", torch.tensor([1, 1, 2, 1]))
        self.register_buffer("decode_params", torch.tensor([2, 1, 1, 1]))

    @torch.jit.export
    def get_sample_rate(self) -> int:
        return 48000

    @torch.jit.export
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.cat((audio, audio * 2.0), dim=1)

    @torch.jit.export
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, 0:1] + latent[:, 1:2]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(audio))


class MalformedForwardParamsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("forward_params", torch.tensor([0, 1, 1, 1]))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio


class MalformedEncodeParamsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("encode_params", torch.tensor([1, 1, -2, 1]))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio


class BadShapeModel(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.cat((audio, audio), dim=1)


class NonFiniteModel(torch.nn.Module):
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return audio + float("nan")


def trace(path: pathlib.Path, model: torch.nn.Module) -> None:
    example = torch.zeros((1, 1, 4), dtype=torch.float32)
    torch.jit.trace(model.eval(), example).save(str(path))


def script(path: pathlib.Path, model: torch.nn.Module) -> None:
    torch.jit.script(model.eval()).save(str(path))


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_torch_backend_test.py <test-executable>")

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        identity_path = root / "identity.pt"
        rave_path = root / "rave-like.ts"
        malformed_forward_path = root / "malformed-forward.ts"
        malformed_encode_path = root / "malformed-encode.ts"
        bad_shape_path = root / "bad-shape.pt"
        non_finite_path = root / "non-finite.pt"

        trace(identity_path, IdentityModel())
        script(rave_path, RaveLikeModel())
        script(malformed_forward_path, MalformedForwardParamsModel())
        script(malformed_encode_path, MalformedEncodeParamsModel())
        trace(bad_shape_path, BadShapeModel())
        trace(non_finite_path, NonFiniteModel())

        return subprocess.run(
            [
                sys.argv[1],
                str(identity_path),
                str(rave_path),
                str(malformed_forward_path),
                str(malformed_encode_path),
                str(bad_shape_path),
                str(non_finite_path),
            ],
            check=False,
        ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
