#include "model/ModelQualification.h"
#include "model/TorchScriptBackend.h"

#include "model/BackgroundModelLoader.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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

void testMalformedMetadataIsRejected(const char* const malformedForwardPath,
                                     const char* const malformedEncodePath)
{
    rave::TorchScriptBackend backend;
    std::string error;
    require(!backend.load(malformedForwardPath, error), "malformed forward_params rejected");
    require(error.find("malformed") != std::string::npos,
            "malformed forward_params message is actionable");

    error.clear();
    require(!backend.load(malformedEncodePath, error), "malformed encode_params rejected");
    require(error.find("malformed") != std::string::npos,
            "malformed encode_params message is actionable");
}

void testSampleRateQualification(const char* const raveModelPath)
{
    rave::TorchScriptBackend backend;
    std::string error;
    require(backend.load(raveModelPath, error), "RAVE model loads for qualification");
    require(backend.modelSampleRate() == 48000, "declared sample rate exposed for messages");
    require(backend.supportsConfiguration(48000.0, 8), "matching rate supported");
    require(!backend.supportsConfiguration(44100.0, 8), "mismatched rate unsupported");

    const auto mismatch = rave::qualifyModelBackend(backend, rave::ModelRuntimeConfiguration { 44100.0, 8 });
    require(!mismatch.empty(), "mismatched candidate fails qualification");
    require(mismatch.find("44100") != std::string::npos
                && mismatch.find("48000") != std::string::npos,
            "mismatch diagnostic names both sample rates");

    const auto matched = rave::qualifyModelBackend(backend, rave::ModelRuntimeConfiguration { 48000.0, 8 });
    require(matched.empty(), "matching candidate qualifies");

    // Qualification also runs inside the background loader.
    rave::BackgroundModelLoader loader([] {
        return std::make_shared<rave::TorchScriptBackend>();
    });
    const auto waitForTerminal = [&loader](const char* const what) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (loader.state() == rave::BackgroundModelLoader::State::loading
               && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        require(loader.state() != rave::BackgroundModelLoader::State::loading, what);
    };

    require(loader.start(raveModelPath, rave::ModelRuntimeConfiguration { 44100.0, 8 }),
            "mismatched loader run starts");
    waitForTerminal("mismatched loader run completes");
    auto result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::failed,
            "loader rejects sample-rate mismatch");
    require(result.backend == nullptr, "loader returns no backend for mismatch");
    require(result.errorMessage.find("sample rate") != std::string::npos,
            "loader mismatch message is actionable");

    require(loader.start(raveModelPath, rave::ModelRuntimeConfiguration { 48000.0, 8 }),
            "matching loader run starts");
    waitForTerminal("matching loader run completes");
    result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::succeeded,
            "loader activates matching candidate");
    require(result.backend != nullptr, "loader returns qualified backend");
}

void testDefectiveModelsFailQualification(const char* const badShapePath,
                                          const char* const nonFinitePath,
                                          const char* const throwingResetPath)
{
    rave::TorchScriptBackend badShape;
    std::string error;
    require(badShape.load(badShapePath, error), "unexpected-shape model loads generically");
    auto qualification = rave::qualifyModelBackend(badShape, rave::ModelRuntimeConfiguration { 48000.0, 8 });
    require(!qualification.empty(), "unexpected output shape fails qualification");
    require(qualification.find("warm-up") != std::string::npos,
            "unexpected shape diagnostic mentions warm-up");

    rave::TorchScriptBackend nonFinite;
    require(nonFinite.load(nonFinitePath, error), "non-finite model loads generically");
    qualification = rave::qualifyModelBackend(nonFinite, rave::ModelRuntimeConfiguration { 48000.0, 8 });
    require(!qualification.empty(), "non-finite model fails qualification");
    require(qualification.find("non-finite") != std::string::npos,
            "non-finite diagnostic is actionable");

    rave::TorchScriptBackend throwingReset;
    require(throwingReset.load(throwingResetPath, error), "throwing-reset model loads");
    qualification = rave::qualifyModelBackend(throwingReset, rave::ModelRuntimeConfiguration { 48000.0, 8 });
    require(!qualification.empty(), "failing reset fails qualification");
    require(qualification.find("reset") != std::string::npos,
            "reset diagnostic is actionable");
}

void testStrictMetadataRejection(const char* const wrongRankPath,
                                 const char* const fractionalParamsPath,
                                 const char* const floatRatePath,
                                 const char* const negativeRatePath)
{
    std::string error;

    rave::TorchScriptBackend wrongRank;
    require(!wrongRank.load(wrongRankPath, error), "wrong-rank forward_params rejected");
    require(error.find("malformed") != std::string::npos,
            "wrong-rank diagnostic is actionable");

    error.clear();
    rave::TorchScriptBackend fractional;
    require(!fractional.load(fractionalParamsPath, error), "fractional forward_params rejected");
    require(error.find("malformed") != std::string::npos,
            "fractional values are not lossily coerced");

    error.clear();
    rave::TorchScriptBackend floatRate;
    require(!floatRate.load(floatRatePath, error), "non-integer get_sample_rate rejected");
    require(error.find("malformed") != std::string::npos,
            "wrong get_sample_rate type is actionable");

    error.clear();
    rave::TorchScriptBackend negativeRate;
    require(!negativeRate.load(negativeRatePath, error), "out-of-range get_sample_rate rejected");
    require(error.find("out of range") != std::string::npos,
            "invalid get_sample_rate range is actionable");
}

void testSamplingRateBufferShapes(const char* const scalarPath,
                                  const char* const vectorPath,
                                  const char* const rank2Path)
{
    std::string error;

    rave::TorchScriptBackend scalar;
    require(scalar.load(scalarPath, error), "scalar sampling_rate buffer is accepted");
    require(error.empty(), "scalar sampling_rate load reports no error");
    require(scalar.modelSampleRate() == 48000, "scalar sampling_rate value is read");

    error.clear();
    rave::TorchScriptBackend vector;
    require(vector.load(vectorPath, error), "single-element 1-D sampling_rate buffer is accepted");
    require(error.empty(), "vector sampling_rate load reports no error");
    require(vector.modelSampleRate() == 48000, "vector sampling_rate value is read");

    error.clear();
    rave::TorchScriptBackend rank2;
    require(!rank2.load(rank2Path, error), "rank-2 singleton sampling_rate buffer is rejected");
    require(error.find("malformed") != std::string::npos,
            "rank-2 sampling_rate diagnostic is actionable");
}

void testRealModelSmoke(const char* const modelPath)
{
    rave::TorchScriptBackend backend;
    std::string error;
    require(backend.load(modelPath, error), "real model loads");
    require(error.empty(), "real model load has no error");
    require(backend.modelSampleRate() == 48000, "real model declares 48 kHz");
    require(backend.latentDimensionCount() == 12, "birds_pluma exposes 12 latent dimensions");
    require(backend.supportsConfiguration(48000.0, 2048), "real model supports the runtime configuration");

    const auto qualification =
        rave::qualifyModelBackend(backend, rave::ModelRuntimeConfiguration { 48000.0, 2048 });
    require(qualification.empty(), "real model qualifies at 48 kHz / 2048 samples");

    std::string resetError;
    require(backend.reset(resetError), "real model resets cleanly");
    require(resetError.empty(), "real model reset reports no error");

    const std::vector<float> input(2048, 0.01f);
    std::vector<float> output(2048);
    require(backend.process(input, {}, output), "real model processes a 2048-sample forward block");
    for (const auto sample : output)
        require(std::isfinite(sample), "real model forward output is finite");

    require(backend.reset(resetError), "real model resets before the latent path");
    const std::vector<float> latentOffsets(backend.latentDimensionCount(), 0.0f);
    require(backend.process(input, latentOffsets, output),
            "real model processes a 2048-sample encode/decode block");
    for (const auto sample : output)
        require(std::isfinite(sample), "real model latent output is finite");
}

int main(const int argc, const char* const* argv)
{
    constexpr int expectedBaseArguments = 15; // argv[0] plus fourteen fixtures
    const char* const realModelPath = std::getenv("RAVE_TEST_MODEL_PATH");
    const bool realModelRequested = realModelPath != nullptr && realModelPath[0] != '\0';
    require(argc == expectedBaseArguments + (realModelRequested ? 1 : 0),
            "expected fourteen fixture paths plus an optional real-model path");

    testGenericTorchScriptModel(argv[1]);
    testRaveMetadataAndLatentOffsets(argv[2]);
    testMalformedMetadataIsRejected(argv[3], argv[4]);
    testSampleRateQualification(argv[2]);
    testDefectiveModelsFailQualification(argv[5], argv[6], argv[11]);
    testStrictMetadataRejection(argv[7], argv[8], argv[9], argv[10]);
    testSamplingRateBufferShapes(argv[12], argv[13], argv[14]);
    if (realModelRequested)
        testRealModelSmoke(realModelPath);
    std::cout << "TorchScriptBackend tests passed\n";
}
