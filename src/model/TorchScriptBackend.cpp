#include "model/TorchScriptBackend.h"

#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <optional>
#include <utility>
#include <vector>

namespace rave
{
namespace
{
using MethodParameters = std::array<std::int64_t, 4>;

std::optional<MethodParameters> readMethodParameters(const torch::jit::Module& module,
                                                     const char* const attributeName)
{
    if (!module.hasattr(attributeName))
        return std::nullopt;

    const auto value = module.attr(attributeName);
    if (!value.isTensor())
        return std::nullopt;

    const auto tensor = value.toTensor().to(torch::kCPU).to(torch::kInt64).contiguous();
    if (tensor.numel() != 4)
        return std::nullopt;

    const auto* data = tensor.data_ptr<std::int64_t>();
    MethodParameters parameters { data[0], data[1], data[2], data[3] };
    if (std::any_of(parameters.begin(), parameters.end(), [](const auto item) { return item <= 0; }))
        return std::nullopt;
    return parameters;
}
} // namespace

struct TorchScriptBackend::Impl
{
    std::unique_ptr<torch::jit::Module> module;
    double preparedSampleRate = 0.0;
    std::size_t maximumBlockSize = 0;
    std::size_t latentDimensions = 0;
    std::size_t inputChannels = 1;
    std::size_t outputChannels = 1;
    int exportedSampleRate = -1;
    bool supportsLatentProcessing = false;
    bool sampleRateMatches = true;
};

TorchScriptBackend::TorchScriptBackend()
    : impl(std::make_unique<Impl>())
{
}

TorchScriptBackend::~TorchScriptBackend() = default;

bool TorchScriptBackend::load(const std::string& modelPath, std::string& errorMessage)
{
    errorMessage.clear();
    impl->module.reset();
    impl->latentDimensions = 0;
    impl->inputChannels = 1;
    impl->outputChannels = 1;
    impl->exportedSampleRate = -1;
    impl->supportsLatentProcessing = false;
    impl->sampleRateMatches = true;

    try
    {
        torch::InferenceMode inferenceMode;
        auto candidate = std::make_unique<torch::jit::Module>(torch::jit::load(modelPath));
        candidate->eval();

        if (!candidate->find_method("forward").has_value())
        {
            errorMessage = "TorchScript model does not expose forward(audio)";
            return false;
        }

        std::size_t inputChannels = 1;
        std::size_t outputChannels = 1;
        std::size_t latentDimensions = 0;
        bool supportsLatentProcessing = false;

        const auto forwardParameters = readMethodParameters(*candidate, "forward_params");
        if (forwardParameters.has_value())
        {
            inputChannels = static_cast<std::size_t>((*forwardParameters)[0]);
            outputChannels = static_cast<std::size_t>((*forwardParameters)[2]);
            if ((*forwardParameters)[1] != 1 || (*forwardParameters)[3] != 1)
            {
                errorMessage = "RAVE forward metadata uses an unsupported temporal ratio";
                return false;
            }
        }

        const auto encodeParameters = readMethodParameters(*candidate, "encode_params");
        const auto decodeParameters = readMethodParameters(*candidate, "decode_params");
        const auto encodeMethod = candidate->find_method("encode");
        const auto decodeMethod = candidate->find_method("decode");
        if (encodeParameters.has_value() && decodeParameters.has_value()
            && encodeMethod.has_value() && decodeMethod.has_value())
        {
            const auto encodeOutputChannels = (*encodeParameters)[2];
            const auto decodeInputChannels = (*decodeParameters)[0];
            const auto encodeOutputRatio = (*encodeParameters)[3];
            const auto decodeInputRatio = (*decodeParameters)[1];
            if (encodeOutputChannels != decodeInputChannels || encodeOutputRatio != decodeInputRatio
                || (*encodeParameters)[0] != static_cast<std::int64_t>(inputChannels)
                || (*decodeParameters)[2] != static_cast<std::int64_t>(outputChannels))
            {
                errorMessage = "RAVE encode/decode metadata is inconsistent";
                return false;
            }
            latentDimensions = static_cast<std::size_t>(encodeOutputChannels);
            supportsLatentProcessing = true;
        }

        if (inputChannels != 1 || outputChannels != 1)
        {
            errorMessage = "Initial audio engine supports mono RAVE model input and output only";
            return false;
        }

        int exportedSampleRate = -1;
        if (const auto sampleRateMethod = candidate->find_method("get_sample_rate");
            sampleRateMethod.has_value())
        {
            const auto value = (*sampleRateMethod)({});
            if (value.isInt())
                exportedSampleRate = static_cast<int>(value.toInt());
        }

        impl->module = std::move(candidate);
        impl->inputChannels = inputChannels;
        impl->outputChannels = outputChannels;
        impl->latentDimensions = latentDimensions;
        impl->supportsLatentProcessing = supportsLatentProcessing;
        impl->exportedSampleRate = exportedSampleRate;
        impl->sampleRateMatches = true;
        return true;
    }
    catch (const c10::Error& error)
    {
        errorMessage = error.what_without_backtrace();
    }
    catch (const std::exception& error)
    {
        errorMessage = error.what();
    }

    return false;
}

void TorchScriptBackend::prepare(const double sampleRate, const std::size_t maximumBlockSize)
{
    impl->preparedSampleRate = sampleRate;
    impl->maximumBlockSize = maximumBlockSize;
    impl->sampleRateMatches = impl->exportedSampleRate <= 0
        || std::abs(sampleRate - static_cast<double>(impl->exportedSampleRate)) < 0.5;
}

void TorchScriptBackend::reset() noexcept
{
    if (impl->module == nullptr)
        return;

    try
    {
        torch::InferenceMode inferenceMode;
        if (const auto resetMethod = impl->module->find_method("reset"); resetMethod.has_value())
            (*resetMethod)({});
    }
    catch (...)
    {
        // reset is best-effort because this interface cannot report an error.
    }
}

std::size_t TorchScriptBackend::latentDimensionCount() const noexcept
{
    return impl->latentDimensions;
}

int TorchScriptBackend::modelSampleRate() const noexcept
{
    return impl->exportedSampleRate;
}

std::size_t TorchScriptBackend::inputChannelCount() const noexcept
{
    return impl->inputChannels;
}

std::size_t TorchScriptBackend::outputChannelCount() const noexcept
{
    return impl->outputChannels;
}

bool TorchScriptBackend::process(const std::span<const float> input,
                                 const std::span<const float> latentControls,
                                 const std::span<float> output)
{
    if (impl->module == nullptr || input.empty() || output.size() < input.size()
        || !impl->sampleRateMatches
        || (impl->maximumBlockSize != 0 && input.size() > impl->maximumBlockSize))
        return false;

    const auto useLatentControls = !latentControls.empty();
    if (useLatentControls
        && (!impl->supportsLatentProcessing || latentControls.size() != impl->latentDimensions))
        return false;

    try
    {
        torch::InferenceMode inferenceMode;
        const auto sampleCount = static_cast<std::int64_t>(input.size());
        auto inputTensor = torch::from_blob(const_cast<float*>(input.data()),
                                            { 1, 1, sampleCount },
                                            torch::TensorOptions().dtype(torch::kFloat32));

        torch::Tensor result;
        if (!useLatentControls)
        {
            result = impl->module->forward({ inputTensor }).toTensor();
        }
        else
        {
            auto latent = impl->module->get_method("encode")({ inputTensor }).toTensor();
            if (latent.dim() != 3 || latent.size(0) != 1
                || latent.size(1) != static_cast<std::int64_t>(impl->latentDimensions))
                return false;

            auto controls = torch::from_blob(const_cast<float*>(latentControls.data()),
                                             { 1, static_cast<std::int64_t>(latentControls.size()), 1 },
                                             torch::TensorOptions().dtype(torch::kFloat32));
            latent = latent + controls;
            result = impl->module->get_method("decode")({ latent }).toTensor();
        }

        result = result.to(torch::kCPU).contiguous();
        if (result.scalar_type() != torch::kFloat32 || result.dim() != 3
            || result.size(0) != 1 || result.size(1) != 1 || result.size(2) != sampleCount)
            return false;

        std::copy_n(result.data_ptr<float>(), input.size(), output.data());
        return true;
    }
    catch (const c10::Error&)
    {
        return false;
    }
    catch (const std::exception&)
    {
        return false;
    }
}
} // namespace rave
