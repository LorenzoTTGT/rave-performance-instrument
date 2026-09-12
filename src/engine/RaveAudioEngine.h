#pragma once

#include <juce_audio_devices/juce_audio_devices.h>

#include "engine/AudioBlockQueue.h"
#include "engine/InferenceWorker.h"
#include "model/ModelBackend.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace rave
{
class RaveAudioEngine final : public juce::AudioIODeviceCallback
{
public:
    // Replace models only while the audio device callback is stopped.
    void setModelBackend(ModelBackendPtr backend);
    [[nodiscard]] bool hasModelBackend() const noexcept;

    void setDryWet(float newValue) noexcept;
    [[nodiscard]] float dryWet() const noexcept;
    [[nodiscard]] std::size_t latentDimensionCount() const noexcept;
    [[nodiscard]] bool setLatentControl(std::size_t index, float value) noexcept;
    [[nodiscard]] std::uint64_t missedInferenceDeadlineCount() const noexcept;
    [[nodiscard]] std::uint64_t alignmentErrorCount() const noexcept;

    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;
    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                          int numInputChannels,
                                          float* const* outputChannelData,
                                          int numOutputChannels,
                                          int numSamples,
                                          const juce::AudioIODeviceCallbackContext&) override;

private:
    void renderDry(const float* const* inputChannelData,
                   int numInputChannels,
                   float* const* outputChannelData,
                   int numOutputChannels,
                   int numSamples) noexcept;

    InferenceWorker inferenceWorker;
    AudioBlockQueue dryBlockQueue;
    std::vector<float> wetBuffer;
    std::vector<float> delayedDryBuffer;
    std::vector<const float*> dryInputPointers;
    std::vector<float*> dryOutputPointers;
    std::atomic<float> dryWetValue { 0.0f };
    std::atomic<bool> modelBackendConfigured { false };
    std::atomic<std::size_t> modelLatentDimensionCount { 0 };
    std::atomic<std::uint64_t> missedDeadlines { 0 };
    std::atomic<std::uint64_t> alignmentErrors { 0 };
    std::uint64_t nextSequence = 1;
    std::size_t maximumBlockSize = 0;
    int configuredOutputChannels = 0;
};
} // namespace rave
