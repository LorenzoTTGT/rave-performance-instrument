#pragma once

#include "model/ModelBackend.h"

#include <memory>

namespace rave
{
// Initial LibTorch backend for TorchScript models exposing forward(audio).
// Latent encode/decode support will be layered on after model metadata probing.
class TorchScriptBackend final : public ModelBackend
{
public:
    TorchScriptBackend();
    ~TorchScriptBackend() override;

    bool load(const std::string& modelPath, std::string& errorMessage) override;
    void prepare(double sampleRate, std::size_t maximumBlockSize) override;
    void reset() noexcept override;

    [[nodiscard]] std::size_t latentDimensionCount() const noexcept override;
    [[nodiscard]] int modelSampleRate() const noexcept;
    [[nodiscard]] std::size_t inputChannelCount() const noexcept;
    [[nodiscard]] std::size_t outputChannelCount() const noexcept;

    bool process(std::span<const float> input,
                 std::span<const float> latentControls,
                 std::span<float> output) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace rave
