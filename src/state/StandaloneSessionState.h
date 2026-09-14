#pragma once

#include "model/ModelQualification.h"

#include <juce_core/juce_core.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <vector>

namespace rave
{
class StandaloneSessionState final
{
public:
    static constexpr int schemaVersion = 1;
    static constexpr std::size_t maximumLatents = maximumLatentDimensions;
    static constexpr std::size_t maximumSerializedBytes = 1024 * 1024;
    static constexpr int dryWetTarget = -1;

    struct Snapshot
    {
        juce::String modelPath;
        juce::String midiInputId;
        juce::String audioDeviceType;
        juce::String audioOutputId;
        juce::String audioInputId;
        float dryWet = 0.0f;
        std::vector<float> latents;
        std::vector<int> midiControllers; // element 0 dry/wet, then latent index + 1
    };

    StandaloneSessionState();
    void setLatentCount(std::size_t count);
    [[nodiscard]] std::size_t latentCount() const noexcept;
    [[nodiscard]] float latent(std::size_t index) const noexcept;
    [[nodiscard]] bool setLatent(std::size_t index, float value) noexcept;
    void setDryWet(float value) noexcept;
    [[nodiscard]] float dryWet() const noexcept;

    void beginMidiLearn(int target) noexcept;
    void clearMidiMapping(int target) noexcept;
    [[nodiscard]] int midiController(int target) const noexcept;
    [[nodiscard]] bool applyMidiCc(int controller, int value) noexcept;

    void setModelPath(juce::String value);
    void setMidiInputId(juce::String value);
    void setAudioSetup(juce::String type, juce::String output, juce::String input);
    [[nodiscard]] Snapshot snapshot() const;
    [[nodiscard]] bool restore(const Snapshot& value);
    [[nodiscard]] bool serialize(juce::MemoryBlock& output) const;
    [[nodiscard]] bool deserialize(const void* data, std::size_t bytes);

private:
    static constexpr std::size_t maximumStringLength = 4096;
    static constexpr int noLearningTarget = -2;

    mutable juce::CriticalSection controlLock;
    Snapshot identity;
    // Fixed-capacity storage is allocated with this object and never replaced,
    // so realtime readers remain lifetime-safe while the message thread changes count.
    std::array<std::atomic<float>, maximumLatents> latentValues {};
    std::array<std::atomic<int>, maximumLatents + 1> mappings {};
    std::atomic<std::size_t> count { 0 };
    std::atomic<float> mix { 0.0f };
    std::atomic<int> learning { noLearningTarget };
};
}
