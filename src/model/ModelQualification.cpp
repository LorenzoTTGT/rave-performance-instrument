#include "model/ModelQualification.h"

#include <cmath>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace rave
{
namespace
{
// Warm up a couple of blocks so stateful streaming models populate internal
// caches at the intended configuration before activation.
constexpr std::size_t warmUpBlockCount = 2;
constexpr float warmUpAmplitude = 0.01f;

float warmUpSample(const std::size_t index, const int blockIndex) noexcept
{
    const auto polarity = (index / 8) % 2 == 0 ? 1.0f : -1.0f;
    return polarity * warmUpAmplitude * static_cast<float>(blockIndex + 1);
}
} // namespace

std::string checkModelConfiguration(ModelBackend& backend, const ModelRuntimeConfiguration& configuration)
{
    if (!(configuration.sampleRate > 0.0) || configuration.maximumBlockSize == 0)
        return "No audio runtime configuration is available yet; model loading is deferred";

    if (!backend.supportsConfiguration(configuration.sampleRate, configuration.maximumBlockSize))
    {
        const auto modelRate = backend.modelSampleRate();
        if (modelRate > 0)
            return "Model sample rate " + std::to_string(modelRate)
                + " Hz does not match the active "
                + std::to_string(std::lround(configuration.sampleRate)) + " Hz configuration";
        return "Model rejected the active runtime configuration";
    }

    return {};
}

std::string qualifyModelBackend(ModelBackend& backend, const ModelRuntimeConfiguration& configuration)
{
    try
    {
        if (const auto configurationError = checkModelConfiguration(backend, configuration);
            !configurationError.empty())
            return configurationError;

        backend.prepare(configuration.sampleRate, configuration.maximumBlockSize);

        std::string resetError;
        if (!backend.reset(resetError))
            return "Model reset failed: "
                + (resetError.empty() ? std::string("model rejected reset") : std::move(resetError));

        std::vector<float> input(configuration.maximumBlockSize);
        std::vector<float> output(configuration.maximumBlockSize);
        const std::vector<float> latent(backend.latentDimensionCount(), 0.0f);

        for (int blockIndex = 0; blockIndex < static_cast<int>(warmUpBlockCount); ++blockIndex)
        {
            for (std::size_t index = 0; index < input.size(); ++index)
                input[index] = warmUpSample(index, blockIndex);

            if (!backend.process(input, latent, output))
                return "Model warm-up failed at the active runtime configuration";

            for (const auto sample : output)
            {
                if (!std::isfinite(sample))
                    return "Model warm-up produced non-finite output at the active runtime configuration";
            }
        }
    }
    catch (const std::exception& exception)
    {
        return std::string("Model qualification threw an exception: ") + exception.what();
    }
    catch (...)
    {
        return "Model qualification threw an unknown exception";
    }

    return {};
}
} // namespace rave
