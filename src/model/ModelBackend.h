#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace rave
{
// Device configuration a candidate model must be qualified against before it
// may be reported active. A non-positive sample rate or zero block size means
// "not configured yet": no real host/device configuration has been observed,
// and neither qualification nor activation may run against invented defaults.
struct ModelRuntimeConfiguration
{
    double sampleRate = 0.0;
    std::size_t maximumBlockSize = 0;
};

class ModelBackend
{
public:
    virtual ~ModelBackend() = default;

    virtual bool load(const std::string& modelPath, std::string& errorMessage) = 0;
    virtual void prepare(double sampleRate, std::size_t maximumBlockSize) = 0;

    // Resets streaming state before first use. Returns false with a diagnostic
    // when the model's reset cannot be performed; such candidates must fail
    // qualification instead of becoming active with stale or unknown state.
    virtual bool reset(std::string& errorMessage) = 0;

    [[nodiscard]] virtual std::size_t latentDimensionCount() const noexcept = 0;

    // Sample rate the model was exported for, or a non-positive value when the
    // model does not declare one. Consulted during candidate qualification so
    // incompatible sample rates are rejected before activation.
    [[nodiscard]] virtual int modelSampleRate() const noexcept { return -1; }

    // Returns false when the model cannot run at the intended device
    // configuration, for example an exported sample rate mismatch. Called
    // during off-audio-thread candidate qualification and again under
    // activation ownership before a candidate replaces the active model.
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
