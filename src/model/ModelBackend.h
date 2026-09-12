#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace rave
{
class ModelBackend
{
public:
    virtual ~ModelBackend() = default;

    virtual bool load(const std::string& modelPath, std::string& errorMessage) = 0;
    virtual void prepare(double sampleRate, std::size_t maximumBlockSize) = 0;
    virtual void reset() noexcept = 0;

    [[nodiscard]] virtual std::size_t latentDimensionCount() const noexcept = 0;

    // Called by an inference worker, never directly by the real-time audio callback.
    virtual bool process(std::span<const float> input,
                         std::span<const float> latentControls,
                         std::span<float> output) = 0;
};

using ModelBackendPtr = std::shared_ptr<ModelBackend>;
} // namespace rave
