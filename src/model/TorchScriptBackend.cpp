#include "model/TorchScriptBackend.h"

#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace rave
{
namespace
{
using MethodParameters = std::array<std::int64_t, 4>;

// Upper sanity bound for a declared model sample rate; far above any audio
// interface rate, so values beyond it indicate malformed metadata.
constexpr int maximumPlausibleSampleRate = 1'000'000;

// Result of probing an optional nn~/RAVE metadata buffer. "present but not
// valid" distinguishes malformed metadata, which must reject the model, from
// absent metadata, which simply means the capability is unavailable.
struct MethodParameterProbe
{
    bool present = false;
    bool valid = false;
    MethodParameters parameters {};
};

MethodParameterProbe readMethodParameters(const torch::jit::Module& module,
                                          const char* const attributeName)
{
    MethodParameterProbe probe;
    if (!module.hasattr(attributeName))
        return probe;

    probe.present = true;
    const auto value = module.attr(attributeName);
    // Present metadata must be a strictly valid parameter spec; anything else
    // (non-tensor, fractional/wrong dtype, wrong rank or shape, non-positive
    // values) is malformed and rejects the model instead of being coerced.
    if (!value.isTensor())
        return probe;

    const auto tensor = value.toTensor();
    if (!at::isIntegralType(tensor.scalar_type(), /*includeBool=*/false))
        return probe;

    const auto cpu = tensor.to(torch::kCPU).to(torch::kInt64).contiguous();
    if (cpu.dim() != 1 || cpu.numel() != 4)
        return probe;

    const auto* data = cpu.data_ptr<std::int64_t>();
    MethodParameters parameters { data[0], data[1], data[2], data[3] };
    if (std::any_of(parameters.begin(), parameters.end(), [](const auto item) { return item <= 0; }))
        return probe;

    probe.valid = true;
    probe.parameters = parameters;
    return probe;
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
        if (forwardParameters.present && !forwardParameters.valid)
        {
            errorMessage = "RAVE forward_params metadata is malformed";
            return false;
        }
        if (forwardParameters.valid)
        {
            inputChannels = static_cast<std::size_t>(forwardParameters.parameters[0]);
            outputChannels = static_cast<std::size_t>(forwardParameters.parameters[2]);
            if (forwardParameters.parameters[1] != 1 || forwardParameters.parameters[3] != 1)
            {
                errorMessage = "RAVE forward metadata uses an unsupported temporal ratio";
                return false;
            }
        }

        const auto encodeParameters = readMethodParameters(*candidate, "encode_params");
        if (encodeParameters.present && !encodeParameters.valid)
        {
            errorMessage = "RAVE encode_params metadata is malformed";
            return false;
        }
        const auto decodeParameters = readMethodParameters(*candidate, "decode_params");
        if (decodeParameters.present && !decodeParameters.valid)
        {
            errorMessage = "RAVE decode_params metadata is malformed";
            return false;
        }
        const auto encodeMethod = candidate->find_method("encode");
        const auto decodeMethod = candidate->find_method("decode");
        if (encodeParameters.valid && decodeParameters.valid
            && encodeMethod.has_value() && decodeMethod.has_value())
        {
            const auto encodeOutputChannels = encodeParameters.parameters[2];
            const auto decodeInputChannels = decodeParameters.parameters[0];
            const auto encodeOutputRatio = encodeParameters.parameters[3];
            const auto decodeInputRatio = decodeParameters.parameters[1];
            if (encodeOutputChannels != decodeInputChannels || encodeOutputRatio != decodeInputRatio
                || encodeParameters.parameters[0] != static_cast<std::int64_t>(inputChannels)
                || decodeParameters.parameters[2] != static_cast<std::int64_t>(outputChannels))
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
            if (!value.isInt())
            {
                errorMessage = "RAVE get_sample_rate metadata is malformed";
                return false;
            }
            const auto declaredRate = value.toInt();
            // A declared sample rate must be a plausible audio rate; anything
            // else is malformed metadata rather than an unknown rate.
            if (declaredRate <= 0 || declaredRate > maximumPlausibleSampleRate)
            {
                errorMessage = "RAVE get_sample_rate metadata is out of range";
                return false;
            }
            exportedSampleRate = static_cast<int>(declaredRate);
        }
        else if (candidate->hasattr("sampling_rate"))
        {
            // nn~/RAVE exports also declare the rate as an integer buffer.
            const auto value = candidate->attr("sampling_rate");
            if (!value.isTensor())
            {
                errorMessage = "RAVE sampling_rate metadata is malformed";
                return false;
            }
            const auto tensor = value.toTensor();
            if (!at::isIntegralType(tensor.scalar_type(), /*includeBool=*/false))
            {
                errorMessage = "RAVE sampling_rate metadata is malformed";
                return false;
            }
            // Validate rank and shape before any conversion or flattening: a
            // declared rate must be a scalar or a single-element 1-D buffer.
            // Anything else (for example a rank-2 singleton) is malformed
            // metadata rather than something to coerce.
            if (tensor.dim() != 0 && tensor.dim() != 1)
            {
                errorMessage = "RAVE sampling_rate metadata is malformed";
                return false;
            }
            if (tensor.dim() == 1 && tensor.size(0) != 1)
            {
                errorMessage = "RAVE sampling_rate metadata is malformed";
                return false;
            }
            const auto cpu = tensor.to(torch::kCPU).to(torch::kInt64);
            const auto declaredRate = cpu.item<std::int64_t>();
            if (declaredRate <= 0 || declaredRate > maximumPlausibleSampleRate)
            {
                errorMessage = "RAVE sampling_rate metadata is out of range";
                return false;
            }
            exportedSampleRate = static_cast<int>(declaredRate);
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
    impl->sampleRateMatches = supportsConfiguration(sampleRate, maximumBlockSize);
}

bool TorchScriptBackend::supportsConfiguration(const double sampleRate,
                                               const std::size_t /*maximumBlockSize*/) const noexcept
{
    if (sampleRate <= 0.0)
        return false;
    return impl->exportedSampleRate <= 0
        || std::abs(sampleRate - static_cast<double>(impl->exportedSampleRate)) < 0.5;
}

bool TorchScriptBackend::reset(std::string& errorMessage)
{
    errorMessage.clear();
    if (impl->module == nullptr)
        return true;

    try
    {
        torch::InferenceMode inferenceMode;
        if (const auto resetMethod = impl->module->find_method("reset"); resetMethod.has_value())
            (*resetMethod)({});
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
    catch (...)
    {
        errorMessage = "unknown reset failure";
    }
    return false;
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
