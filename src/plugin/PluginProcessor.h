#pragma once

#include "engine/RaveAudioEngine.h"
#include "model/ModelQualification.h"

#include <juce_audio_processors/juce_audio_processors.h>

#if RAVE_HAS_LIBTORCH
#include "model/BackgroundModelLoader.h"
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

class RavePluginProcessor final : public juce::AudioProcessor,
                                  private juce::Timer
{
public:
    static constexpr std::size_t macroCount = 8;
    static constexpr std::size_t midiTargetCount = macroCount + 1;
    static constexpr int stateSchemaVersion = 2;
    static constexpr std::size_t maximumStateBytes = 1024 * 1024;
    static constexpr std::size_t maximumLatentCount = rave::maximumLatentDimensions;

    // The optional factory injects a test backend; production uses the
    // TorchScript backend factory.
    explicit RavePluginProcessor(std::function<rave::ModelBackendPtr()> backendFactory = {});
    ~RavePluginProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destinationData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    [[nodiscard]] juce::RangedAudioParameter& dryWetParameterReference() const;
    [[nodiscard]] juce::RangedAudioParameter& macroParameterReference(std::size_t index) const;
    void beginMidiLearn(std::size_t targetIndex) noexcept;
    [[nodiscard]] int midiControllerForTarget(std::size_t targetIndex) const noexcept;
    [[nodiscard]] bool isLearningMidiTarget(std::size_t targetIndex) const noexcept;

    [[nodiscard]] bool startModelLoad(const juce::File& modelFile);
    [[nodiscard]] bool relinkMissingModel(const juce::File& modelFile);
    [[nodiscard]] bool isRelinkRequired() const noexcept;
    [[nodiscard]] juce::String requestedModelPath() const;
    [[nodiscard]] bool finishModelLoadIfReady();
    [[nodiscard]] juce::String modelStatus() const;
    [[nodiscard]] bool isModelLoading() const noexcept;
    [[nodiscard]] std::uint64_t modelRevision() const noexcept;
    [[nodiscard]] std::size_t latentDimensionCount() const noexcept;
    [[nodiscard]] float latentControl(std::size_t index) const noexcept;
    [[nodiscard]] bool setLatentControl(std::size_t index, float value) noexcept;

    // Message-thread handoff for MIDI changes captured by processBlock().
    // Coalesces each parameter to its newest callback value.
    void publishPendingMidiParameterChanges();
    // Test seam exercising the exact raw parser used by applyMidi; required
    // because MidiBuffer canonicalizes malformed byte sequences on insertion.
    void applyRawMidiEventForTesting(const std::uint8_t* data, int byteCount) noexcept;

    // Surfaces engine lifecycle transitions (incompatible reprepare, failing
    // checked reset/start, release/device stop) in the visible status instead
    // of leaving an active claim while dry fallback is used. Renders through
    // the shared lifecycle presenter and only acts on unseen revisions, so it
    // never overwrites an explicit activation status that already observed
    // the current revision. Message thread only.
    void refreshLifecycleStatus();

    // Deterministic test seam invoked after a failed-load lifecycle snapshot
    // and before its status commit. Production leaves this empty.
    std::function<void()> failedLoadStatusInterleaveForTesting;
    // Invoked by the non-realtime publisher after capturing a mailbox value
    // and before notifying JUCE. Tests use this to force a newer callback.
    std::function<void()> midiPublishInterleaveForTesting;
    // Invoked after validated state parameters commit and before captured MIDI
    // generations are acknowledged.
    std::function<void()> stateRestoreMidiAcknowledgeInterleaveForTesting;
    // Invoked after a completed backend has been paired with its exact request
    // metadata but before activation commits, allowing a newer restore to win
    // deterministically without mixing request generations.
    std::function<void()> activationMutationInterleaveForTesting;
    // Invoked while getStateInformation owns the outer mutation lock.
    std::function<void()> stateSnapshotInterleaveForTesting;
    void runActivationMutationTransactionForTesting(const std::function<void()>& action);

private:
    void timerCallback() override;
    void applyMidi(juce::MidiBuffer& midiMessages) noexcept;
    void applyRawMidiEvent(const std::uint8_t* data, int byteCount) noexcept;
    [[nodiscard]] float authoritativeParameterValue(
        std::size_t target, const juce::AudioParameterFloat* parameter) const noexcept;
    void activateModel(rave::ModelBackendPtr backend,
                       const juce::File& modelFile,
                       const std::vector<float>& restoredLatents,
                       std::uint64_t generation,
                       bool relink);
    bool queueModelRequest(const juce::File&, std::vector<float>, bool relink,
                           std::uint64_t generation);
    void publishModelLoadFailure(const std::string& errorMessage, bool relink);

    // Outermost lock for message/non-realtime mutations. Recursive because
    // JUCE listener callbacks and test seams may synchronously re-enter.
    mutable std::recursive_mutex nonRealtimeMutationMutex;
    std::uint64_t parameterCommitEpoch = 0;
    std::size_t parameterCommitDepth = 0;
    rave::RaveAudioEngine engine;
    juce::AudioParameterFloat* dryWetParameter = nullptr;
    std::array<juce::AudioParameterFloat*, macroCount> macroParameters {};
    std::array<std::atomic<int>, midiTargetCount> midiControllers;
    std::atomic<int> learningMidiTarget { -1 };
    std::array<std::atomic<float>, midiTargetCount> midiLatestValues {};
    std::array<std::atomic<std::uint64_t>, midiTargetCount> midiValueSequences {};
    std::array<std::atomic<std::uint64_t>, midiTargetCount> midiPublishedSequences {};
#if RAVE_HAS_LIBTORCH
    std::unique_ptr<rave::BackgroundModelLoader> modelLoader;
#endif
    mutable juce::CriticalSection modelStateLock;
    // Event text is stored separately from engine truth. When the flag is set,
    // modelStatus() appends lifecycle state from a fresh coherent snapshot.
    juce::String currentModelStatus;
    bool currentStatusNeedsLifecycle = false;
    juce::File pendingModelFile;
    juce::File activeModelFile;
    juce::File queuedRestoreModelFile;
    std::vector<float> queuedRestoreLatents;
    std::vector<float> pendingLatentRestore;
    std::uint64_t pendingGeneration = 0;
    std::uint64_t queuedGeneration = 0;
    bool pendingRelink = false;
    bool queuedRelink = false;
    std::atomic<std::uint64_t> requestGeneration { 0 };
    juce::File requestedMissingModelFile;
    std::vector<float> retainedMissingLatents;
    bool relinkRequired = false;
    std::atomic<bool> hasQueuedModelRestore { false };
    std::atomic<std::uint64_t> currentModelRevision { 0 };
    // Lifecycle status refresh state (message thread only, guarded by
    // modelStateLock when the status itself is written).
    std::uint64_t lastSeenLifecycleRevision = 0;
};
