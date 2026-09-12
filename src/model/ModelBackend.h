#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace rave
{
// Device configuration a candidate model must be qualified against before it
// may be reported active. The defaults match the documented primary milestone
// configuration (48 kHz) and are used when no device has been prepared yet.
struct ModelRuntimeConfiguration
{
    double sampleRate = 48000.0;
    std::size_t maximumBlockSize = 512;
};

class ModelBackend
{
public:
    virtual ~ModelBackend() = default;

    virtual bool load(const std::string& modelPath, std::string& errorMessage) = 0;
    virtual void prepare(double sampleRate, std::size_t maximumBlockSize) = 0;
    virtual void reset() noexcept = 0;

    [[nodiscard]] virtual std::size_t latentDimensionCount() const noexcept = 0;

    // Sample rate the model was exported for, or a non-positive value when the
    // model does not declare one. Consulted during candidate qualification so
    // incompatible sample rates are rejected before activation.
    [[nodiscard]] virtual int modelSampleRate() const noexcept { return -1; }

    // Returns false when the model cannot run at the intended device
    // configuration, for example an exported sample rate mismatch. Called
    // during off-audio-thread candidate qualification, never in the callback.
    [[nodiscard]] virtual bool supportsConfiguration(double sampleRate, std::size_t) const noexcept
    {
        return sampleRate > 0.0;
    }

    // Called by an inference worker, never directly by the real-time audio callback.
    virtual bool process(std::span<const float> input,
                         std::span<const float> latentControls,
                         std::span<float> output) = 0;
};

using ModelBackendPtr = std::shared_ptr<ModelBackend>;
} // namespace rave
