#pragma once

#include <juce_audio_devices/juce_audio_devices.h>

#include "engine/AudioBlockQueue.h"
#include "engine/InferenceWorker.h"
#include "model/ModelBackend.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace rave
{
class RaveAudioEngine final : public juce::AudioIODeviceCallback
{
public:
    struct LifecycleStatusSnapshot
    {
        std::uint64_t revision = 0;
        bool modelInstalled = false;
        bool modelUsable = false;
        std::string diagnostic;
        std::size_t latentDimensionCount = 0;
    };

    // Replace models only while the audio device callback is stopped.
    void setModelBackend(ModelBackendPtr backend);

    // Transactional replacement for callers that already suspend or stop the
    // device callback. Activation holds lifecycleMutex for the whole
    // sequence, reads the known host configuration under that ownership, and
    // — whenever a configuration is known — prepares the candidate, performs
    // its checked reset, and starts the worker before reporting success. This
    // holds even after release or a previous rollback failure: a known host
    // configuration survives release and callback detachment. On any failure
    // a verified rollback restores the previous backend and restarts it;
    // diagnostics (failureReason, when non-null) truthfully distinguish
    // previous model retained, no previous model, and rollback failure, each
    // naming bounded dry pass-through when nothing usable is running.
    // Candidate checking and preparation never happen inside the audio
    // callback.
    [[nodiscard]] bool activateModelBackend(ModelBackendPtr candidate,
                                            std::string* failureReason = nullptr);

    // Test seam for deterministic lifecycle interleaving tests. Production
    // code never sets it. When set, it is invoked while lifecycleMutex is
    // held, after activation has captured its exact configuration snapshot
    // and before the candidate is committed. Coordinate it with
    // prepareLockBoundaryForTesting: activation waits there until a
    // concurrent prepare has reached the lock boundary, proving the prepare
    // is pending on serialization before activation proceeds; the mutex then
    // orders the two critical sections without scheduler timing.
    std::function<void()> activationInterleaveForTesting;

    // Test seam fired on the calling thread immediately before prepare()
    // acquires lifecycleMutex — the lifecycle-lock acquisition boundary.
    // Deterministic lifecycle tests count down a latch here so activation can
    // prove the concurrent prepare reached the boundary and is demonstrably
    // pending on serialization. Test-only and non-realtime: production never
    // sets it, and prepare never runs on the audio thread.
    std::function<void()> prepareLockBoundaryForTesting;

    // Test-only seam forwarded to the inference worker immediately before
    // worker start, so engine/status-level tests can exercise thread-start
    // failure containment deterministically. Production never sets it.
    std::function<void()> workerThreadStartHookForTesting;

    // Thread-safe lifecycle outcome of the most recent prepare: empty when
    // nothing is degraded, otherwise an actionable description of why the
    // installed model is not usable (incompatible configuration, or a failing
    // checked reset/start) together with the bounded dry pass-through notice.
    // Non-realtime: takes lifecycleMutex.
    [[nodiscard]] std::string lifecycleDiagnostic() const;

    // Monotonic counter bumped whenever the lifecycle diagnostic or usability
    // state changes, so non-realtime status surfaces can poll cheaply.
    [[nodiscard]] std::uint64_t lifecycleRevision() const noexcept;

    // Coherent non-realtime presentation state captured under lifecycle
    // ownership. Status surfaces use this instead of racing independent reads.
    [[nodiscard]] LifecycleStatusSnapshot lifecycleStatusSnapshot() const;

    // The host configuration models are qualified against, or an unset
    // configuration before the first prepare. The configuration is known once
    // any prepare ran and survives release and callback detachment.
    // Non-realtime: takes lifecycleMutex and must never be called from the
    // audio callback.
    [[nodiscard]] ModelRuntimeConfiguration runtimeConfiguration() const;

    // Whether a backend is installed (loaded), regardless of usability.
    [[nodiscard]] bool hasModelBackend() const;

    // Whether the installed backend is compatible with the current
    // configuration and actually running (the wet path is live). Incompatible
    // repares, failing resets, and release all clear usability; the engine
    // then renders bounded dry pass-through.
    [[nodiscard]] bool hasUsableModel() const;

    void setDryWet(float newValue) noexcept;
    [[nodiscard]] float dryWet() const noexcept;
    [[nodiscard]] std::size_t latentDimensionCount() const noexcept;
    [[nodiscard]] float latentControl(std::size_t index) const noexcept;
    [[nodiscard]] bool setLatentControl(std::size_t index, float value) noexcept;
    [[nodiscard]] std::uint64_t missedInferenceDeadlineCount() const noexcept;
    [[nodiscard]] std::uint64_t alignmentErrorCount() const noexcept;

    // Device-independent lifecycle used by standalone and plugin adapters.
    // Device-lifecycle reprepares are best-effort: on every prepare the
    // installed backend is checked against the new exact configuration before
    // the worker starts, and an incompatible model or failing reset leaves
    // bounded dry pass-through. Activation reports such failures through
    // activateModelBackend instead.
    void prepare(double sampleRate,
                 std::size_t maximumSamplesPerBlock,
                 int outputChannelCount);

    // Stops the worker and clears usability, but deliberately keeps the known
    // host configuration so later activations can still prepare, reset, and
    // start against it.
    void release() noexcept;

    void processAudio(const float* const* inputChannelData,
                      int numInputChannels,
                      float* const* outputChannelData,
                      int numOutputChannels,
                      int numSamples) noexcept;

    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;
    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                          int numInputChannels,
                                          float* const* outputChannelData,
                                          int numOutputChannels,
                                          int numSamples,
                                          const juce::AudioIODeviceCallbackContext&) override;

private:
    // One ownership mechanism for every non-audio lifecycle mutation
    // (prepare, release, setModelBackend, activateModelBackend) and for the
    // runtimeConfiguration() snapshot. The real-time audio path
    // (processAudio/renderDry) never takes this mutex; it reads only single
    // atomic copies published under the mutex.
    mutable std::mutex lifecycleMutex;

    // Re-prepares the engine at the given exact configuration, verifies the
    // installed backend supports it, and starts the worker after the
    // backend's checked reset succeeds. lifecycleMutex must be held by the
    // caller. Returns false with a diagnostic in startError (when non-null)
    // when the installed backend is incompatible or its reset/start fails;
    // the engine then renders bounded dry pass-through and records the
    // lifecycle diagnostic.
    [[nodiscard]] bool prepareAndStart(double sampleRate,
                                       std::size_t maximumSamplesPerBlock,
                                       int outputChannelCount,
                                       std::string* startError);

    // Updates the lifecycle diagnostic and usability together and bumps the
    // revision whenever either changes, so every transition — including
    // release/device stop making a usable model stop running — is published
    // for non-realtime pollers. lifecycleMutex must be held by the caller.
    void setLifecycleStateLocked(std::string diagnostic, bool modelUsable);

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
    std::atomic<std::size_t> modelLatentDimensionCount { 0 };
    std::atomic<std::uint64_t> missedDeadlines { 0 };
    std::atomic<std::uint64_t> alignmentErrors { 0 };
    std::uint64_t nextSequence = 1;

    // Single-value copies published under lifecycleMutex for the lock-free
    // audio callback; one atomic store per field cannot tear, so no multi-
    // field snapshot protocol is needed on the real-time path.
    std::atomic<std::size_t> callbackMaximumBlockSize { 0 };
    std::atomic<int> callbackOutputChannels { 1 };

    // Non-audio lifecycle state, guarded by lifecycleMutex. configurationKnown
    // is separate from callback activity: release and callback detachment
    // keep the configuration, rate, block size, and channel count intact.
    double configuredSampleRate = 0.0;
    int configuredOutputChannels = 1;
    bool configurationKnown = false;
    bool modelInstalled = false;
    bool modelUsable = false;
    std::string lifecycleDiagnosticValue;
    std::atomic<std::uint64_t> lifecycleRevisionValue { 0 };
};
} // namespace rave
