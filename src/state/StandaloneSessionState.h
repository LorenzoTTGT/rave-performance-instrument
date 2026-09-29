#pragma once

#include "model/ModelQualification.h"

#include <juce_core/juce_core.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
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
        bool generate = false;
        float motionDepth = 0.5f, motionRate = 0.1f;
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

    void setGenerator(bool enabled, float depth, float rate);
    void setModelPath(juce::String value);
    void setMidiInputId(juce::String value);
    void setAudioSetup(juce::String type, juce::String output, juce::String input);
    [[nodiscard]] Snapshot snapshot() const;
    [[nodiscard]] bool restore(const Snapshot& value);
    [[nodiscard]] bool serialize(juce::MemoryBlock& output) const;
    [[nodiscard]] bool deserialize(const void* data, std::size_t bytes);
    // Test-only seams for deterministic bank-publication schedules.
    // Production never sets them.
    std::function<void()> restoreBeforePublishForTesting;
    mutable std::function<void()> readerAdmissionInterleaveForTesting;
    // Reads at most maximumSerializedBytes + 1, so size validation never
    // follows an unbounded whole-file allocation.
    [[nodiscard]] static bool readBoundedPresetFile(const juce::File& file,
                                                    juce::MemoryBlock& output);

private:
    static constexpr std::size_t maximumStringLength = 4096;
    static constexpr int noLearningTarget = -2;

    struct ControlBank
    {
        std::array<std::atomic<float>, maximumLatents> latentValues {};
        std::array<std::atomic<int>, maximumLatents + 1> mappings {};
        std::atomic<std::size_t> count { 0 };
        std::atomic<float> mix { 0.0f };
        // High bit closes admission; low bits count admitted lock-free readers.
        std::atomic<std::uint32_t> readerState { 0 };
    };

    [[nodiscard]] ControlBank* acquireActiveControlBank() const noexcept;
    static void releaseControlBank(ControlBank& bank) noexcept;

    mutable juce::CriticalSection controlLock;
    Snapshot identity;
    // Restore fills the inactive fixed-capacity bank and atomically publishes
    // it. Realtime readers never observe a partly restored mapping/value set.
    mutable std::array<ControlBank, 2> controlBanks {};
    std::atomic<std::size_t> activeControlBank { 0 };
    std::atomic<int> learning { noLearningTarget };
};
}
