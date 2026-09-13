#!/usr/bin/env python3
import pathlib
import subprocess
import sys
import tempfile

import torch


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


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_plugin_model_recall_test.py <test-executable>")

    with tempfile.TemporaryDirectory() as directory:
        model_path = pathlib.Path(directory) / "recall-model.ts"
        torch.jit.script(RaveLikeModel().eval()).save(str(model_path))
        # Capture the child output so any JUCE assertion - including ones fired
        # inside the plugin target's own compilation units - fails the test.
        completed = subprocess.run(
            [sys.argv[1], str(model_path)], check=False, capture_output=True, text=True
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        if "JUCE Assertion" in completed.stdout or "JUCE Assertion" in completed.stderr:
            print("FAILED: JUCE assertion fired during the plugin test", file=sys.stderr)
            return 1
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
