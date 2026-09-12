#include "model/TorchScriptBackend.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{
void require(const bool condition, const char* const message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void testGenericTorchScriptModel(const char* const modelPath)
{
    rave::TorchScriptBackend backend;
    backend.prepare(48000.0, 8);

    std::string error;
    require(!backend.load("/path/that/does/not/exist.pt", error), "missing model rejected");
    require(!error.empty(), "missing model reports an error");
    require(backend.load(modelPath, error), "generic TorchScript model loads");
    require(error.empty(), "successful load clears prior error");
    require(backend.latentDimensionCount() == 0, "generic model has no inferred latents");

    const std::array<float, 4> input { -1.0f, -0.25f, 0.5f, 1.0f };
    std::array<float, 4> output {};
    require(backend.process(input, {}, output), "identity model processes audio");
    require(output == input, "identity model output matches input");

    const std::array<float, 1> unsupportedLatent { 0.5f };
    require(!backend.process(input, unsupportedLatent, output), "generic model rejects latent controls");
}

void testRaveMetadataAndLatentOffsets(const char* const modelPath)
{
    rave::TorchScriptBackend backend;
    std::string error;
    require(backend.load(modelPath, error), "RAVE-like model loads");
    require(backend.latentDimensionCount() == 2, "latent count read from encode metadata");
    require(backend.modelSampleRate() == 48000, "sample rate read from exported method");
    require(backend.inputChannelCount() == 1 && backend.outputChannelCount() == 1,
            "forward channel metadata read");

    backend.prepare(48000.0, 8);
    const std::array<float, 4> input { 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 4> output {};
    require(backend.process(input, {}, output), "RAVE forward processes audio");
    require(output == std::array<float, 4> { 3.0f, 6.0f, 9.0f, 12.0f },
            "RAVE forward output");

    const std::array<float, 2> latentOffsets { 1.0f, 2.0f };
    require(backend.process(input, latentOffsets, output), "encode-offset-decode processes audio");
    require(output == std::array<float, 4> { 6.0f, 9.0f, 12.0f, 15.0f },
            "per-dimension offsets broadcast over latent frames");

    backend.prepare(44100.0, 8);
    require(!backend.process(input, {}, output), "sample-rate mismatch fails safely");
}
} // namespace

int main(const int argc, const char* const* argv)
{
    require(argc == 3, "expected generic and RAVE-like model fixture paths");
    testGenericTorchScriptModel(argv[1]);
    testRaveMetadataAndLatentOffsets(argv[2]);
    std::cout << "TorchScriptBackend tests passed\n";
}
