#include "engine/RaveAudioEngine.h"

#include "model/ModelQualification.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rave
{
void RaveAudioEngine::setModelBackend(ModelBackendPtr backend)
{
    const std::scoped_lock lock(lifecycleMutex);

    if (inferenceWorker.isRunning())
        throw std::logic_error("Stop the audio device before replacing the model backend");

    modelLatentDimensionCount.store(
        backend != nullptr ? backend->latentDimensionCount() : 0,
        std::memory_order_release);
    modelInstalled = backend != nullptr;
    modelUsable = false;
    inferenceWorker.setBackend(std::move(backend));
}

bool RaveAudioEngine::hasModelBackend() const
{
    const std::scoped_lock lock(lifecycleMutex);
    return modelInstalled;
}

bool RaveAudioEngine::hasUsableModel() const
{
    const std::scoped_lock lock(lifecycleMutex);
    return modelUsable;
}

ModelRuntimeConfiguration RaveAudioEngine::runtimeConfiguration() const
{
    // Non-realtime snapshot under lifecycle ownership; never call this from
    // the audio callback.
    const std::scoped_lock lock(lifecycleMutex);
    if (!configurationKnown)
        return {};
    return { configuredSampleRate, inferenceQuantumSamples };
}

std::string RaveAudioEngine::lifecycleDiagnostic() const
{
    const std::scoped_lock lock(lifecycleMutex);
    return lifecycleDiagnosticValue;
}

std::uint64_t RaveAudioEngine::lifecycleRevision() const noexcept
{
    return lifecycleRevisionValue.load(std::memory_order_acquire);
}

RaveAudioEngine::LifecycleStatusSnapshot RaveAudioEngine::lifecycleStatusSnapshot() const
{
    const std::scoped_lock lock(lifecycleMutex);
    return { lifecycleRevisionValue.load(std::memory_order_relaxed),
             modelInstalled,
             modelUsable,
             lifecycleDiagnosticValue,
             modelLatentDimensionCount.load(std::memory_order_relaxed) };
}

void RaveAudioEngine::setLifecycleStateLocked(std::string diagnostic, const bool modelUsableNew)
{
    // lifecycleMutex must be held by the caller. The revision is bumped
    // whenever either the diagnostic or usability changes, so every usability
    // transition (including release/device stop) is published for pollers.
    if (lifecycleDiagnosticValue == diagnostic && modelUsable == modelUsableNew)
        return;
    lifecycleDiagnosticValue = std::move(diagnostic);
    modelUsable = modelUsableNew;
    lifecycleRevisionValue.fetch_add(1, std::memory_order_release);
}

bool RaveAudioEngine::activateModelBackend(ModelBackendPtr candidate, std::string* failureReason)
{
    const auto fail = [failureReason](std::string reason) {
        if (failureReason != nullptr)
            *failureReason = std::move(reason);
        return false;
    };

    if (candidate == nullptr)
        return fail("candidate backend was empty");

    // The diagnostic is always defined: cleared up front, set on any failure.
    if (failureReason != nullptr)
        failureReason->clear();

    // Lifecycle ownership: prepare, release, and replacement cannot interleave
    // with this activation, so the snapshot below is exact and the commit
    // happens under the same ownership that validated it.
    const std::scoped_lock lock(lifecycleMutex);

    const auto withRetentionSuffix = [&](std::string reason) {
        // Truthful outcome classes: usable models are genuinely playing,
        // installed-but-unusable models survive only as bounded dry
        // pass-through, and without any model there is nothing to retain.
        if (modelUsable)
            return reason + "; previous model retained";
        if (modelInstalled)
            return reason
                + "; previous model is installed but not usable — remaining in bounded dry pass-through";
        return reason + "; no previous model active";
    };

    // Exact-snapshot validation: the configuration is read once under
    // lifecycle ownership and reused verbatim for the commit below.
    const ModelRuntimeConfiguration snapshot {
        configuredSampleRate,
        inferenceQuantumSamples
    };
    if (const auto configurationError = checkModelConfiguration(*candidate, snapshot);
        !configurationError.empty())
        return fail(withRetentionSuffix(configurationError));

    if (activationInterleaveForTesting)
        activationInterleaveForTesting();

    // Callers suspend or stop the device callback first, so this bounded join
    // happens on the message thread and never inside the audio callback.
    inferenceWorker.stop();

    ModelBackendPtr previous = inferenceWorker.currentBackend();
    const auto previousLatentCount = modelLatentDimensionCount.load(std::memory_order_acquire);
    const bool previousInstalled = modelInstalled;
    std::vector<float> previousLatents(previousLatentCount);
    for (std::size_t index = 0; index < previousLatentCount; ++index)
        previousLatents[index] = inferenceWorker.latentControl(index);

    // Install and start the candidate at the exact known configuration:
    // prepare, checked reset, and worker start must all succeed before
    // activation reports success.
    std::string candidateError;
    try
    {
        modelLatentDimensionCount.store(candidate->latentDimensionCount(), std::memory_order_release);
        modelInstalled = true;
        inferenceWorker.setBackend(std::move(candidate));

        std::string startError;
        if (prepareAndStart(snapshot.sampleRate,
                            snapshot.maximumBlockSize,
                            configuredOutputChannels,
                            &startError))
            return true;

        candidateError = startError.empty()
            ? std::string("candidate worker could not be started")
            : startError;
    }
    catch (const std::exception& exception)
    {
        candidateError = exception.what();
    }
    catch (...)
    {
        candidateError = "unknown activation error";
    }

    // Rollback: restore the previous backend and its latent values, and verify
    // the restart (compatibility plus checked reset plus start) before
    // claiming retention. Anything else is reported as bounded dry
    // pass-through.
    bool rollbackRestarted = false;
    std::string rollbackDetail;
    try
    {
        modelLatentDimensionCount.store(previousLatentCount, std::memory_order_release);
        modelInstalled = previousInstalled;
        inferenceWorker.setBackend(std::move(previous));

        std::string restartError;
        rollbackRestarted = prepareAndStart(snapshot.sampleRate,
                                            snapshot.maximumBlockSize,
                                            configuredOutputChannels,
                                            &restartError);
        if (!rollbackRestarted)
            rollbackDetail = restartError.empty()
                ? std::string("previous model could not be restarted")
                : "previous model could not be restarted: " + restartError;

        // setBackend(previous) rebuilt worker latent storage with defaults, so
        // restore every previous control value after the verified restart.
        for (std::size_t index = 0; index < previousLatents.size(); ++index)
            static_cast<void>(inferenceWorker.setLatentControl(index, previousLatents[index]));
    }
    catch (const std::exception& exception)
    {
        rollbackRestarted = false;
        rollbackDetail = std::string("rollback threw: ") + exception.what();
    }
    catch (...)
    {
        rollbackRestarted = false;
        rollbackDetail = "rollback threw an unknown error";
    }

    if (previousInstalled && rollbackRestarted)
        return fail(candidateError + "; previous model retained");

    if (!previousInstalled)
        return fail(candidateError
                    + "; no previous model to retain — remaining in bounded dry pass-through");

    // The rollback restart itself failed: do not claim retention. Abandon the
    // backend atomically: clear the installed flag, the engine latent count,
    // and the worker latent storage/count (via setBackend(nullptr)) so
    // hasModelBackend false implies zero latents and no stale values.
    inferenceWorker.stop();
    modelInstalled = false;
    modelLatentDimensionCount.store(0, std::memory_order_release);
    inferenceWorker.setBackend(nullptr);
    setLifecycleStateLocked(candidateError + "; " + rollbackDetail
                                 + "; no usable model — bounded dry pass-through",
                            false);
    return fail(candidateError + "; " + rollbackDetail
                + " — remaining in bounded dry pass-through");
}

void RaveAudioEngine::setDryWet(const float newValue) noexcept
{
    dryWetValue.store(juce::jlimit(0.0f, 1.0f, newValue), std::memory_order_relaxed);
}

float RaveAudioEngine::dryWet() const noexcept
{
    return dryWetValue.load(std::memory_order_relaxed);
}

std::size_t RaveAudioEngine::latentDimensionCount() const noexcept
{
    return modelLatentDimensionCount.load(std::memory_order_acquire);
}

float RaveAudioEngine::latentControl(const std::size_t index) const noexcept
{
    return inferenceWorker.latentControl(index);
}

bool RaveAudioEngine::setLatentControl(const std::size_t index, const float value) noexcept
{
    return inferenceWorker.setLatentControl(index, value);
}

std::uint64_t RaveAudioEngine::missedInferenceDeadlineCount() const noexcept
{
    return missedDeadlines.load(std::memory_order_relaxed);
}

std::uint64_t RaveAudioEngine::alignmentErrorCount() const noexcept
{
    return alignmentErrors.load(std::memory_order_relaxed);
}

void RaveAudioEngine::prepare(const double sampleRate,
                              const std::size_t maximumSamplesPerBlock,
                              const int outputChannelCount)
{
    // Test-only seam at the lifecycle-lock acquisition boundary: fired before
    // this prepare can touch the mutex, so deterministic tests can prove a
    // concurrent prepare reached the boundary and is pending on
    // serialization. Never set in production; prepare never runs on the
    // audio thread.
    if (prepareLockBoundaryForTesting != nullptr)
        prepareLockBoundaryForTesting();

    const std::scoped_lock lock(lifecycleMutex);

    // Device-lifecycle reprepares are best-effort: if the installed backend
    // does not support the new configuration or its reset fails, the worker
    // stays stopped, the diagnostic is recorded, and the engine renders
    // bounded dry audio. Activation reports such failures through its own
    // verified path. Host/device prepare paths must also never throw into the
    // caller, so contain anything prepareAndStart did not convert and record
    // it as a stopped, unusable state.
    std::string startError;
    try
    {
        static_cast<void>(prepareAndStart(sampleRate, maximumSamplesPerBlock, outputChannelCount, &startError));
    }
    catch (const std::exception& exception)
    {
        inferenceWorker.stop();
        setLifecycleStateLocked(std::string("prepare failed: ") + exception.what()
                                    + "; the installed model stays silent — bounded dry pass-through",
                                false);
    }
    catch (...)
    {
        inferenceWorker.stop();
        setLifecycleStateLocked("prepare failed with an unknown exception; the installed model "
                                    "stays silent — bounded dry pass-through",
                                false);
    }
}

bool RaveAudioEngine::prepareAndStart(const double sampleRate,
                                      const std::size_t maximumSamplesPerBlock,
                                      const int outputChannelCount,
                                      std::string* const startError)
{
    // lifecycleMutex must be held by the caller.
    if (startError != nullptr)
        startError->clear();

    inferenceWorker.stop();
    sampleClock = 0;
    submittedFrame = 0;
    smoothedDryWet = dryWet();
    availability = 0.0f;
    missedDeadlines.store(0, std::memory_order_relaxed);
    alignmentErrors.store(0, std::memory_order_relaxed);
    lateResults.store(0, std::memory_order_relaxed);
    renderedSamples.store(0, std::memory_order_relaxed);

    configurationKnown = true;
    configuredSampleRate = sampleRate;
    configuredOutputChannels = std::max(1, outputChannelCount);
    const auto blockSize = std::max<std::size_t>(1, maximumSamplesPerBlock);

    // Single atomic copies for the lock-free audio callback; one store per
    // field cannot tear, so no multi-field snapshot protocol is needed there.
    callbackMaximumBlockSize.store(blockSize, std::memory_order_relaxed);
    callbackOutputChannels.store(configuredOutputChannels, std::memory_order_relaxed);

    constexpr std::size_t inferenceQueueCapacity = 4;
    dryTimeline.assign(static_cast<std::size_t>(configuredOutputChannels) * dryRingSamples, 0.0f);
    inputFrame.assign(inferenceQuantumSamples, 0.0f);
    receiveFrame.assign(inferenceQuantumSamples, 0.0f);
    wetTimeline.assign(timelineFrameCount * inferenceQuantumSamples, 0.0f);
    wetFrameTags.assign(timelineFrameCount, std::numeric_limits<std::uint64_t>::max());
    smoothingStep = static_cast<float>(1.0 / std::max(1.0, sampleRate * 0.005));

    if (modelInstalled)
    {
        auto* const installed = inferenceWorker.currentBackend().get();
        // Verify the installed backend supports the new exact configuration
        // before starting; otherwise keep it installed, record the lifecycle
        // diagnostic, and report bounded dry pass-through.
        if (const auto configurationError =
                checkModelConfiguration(*installed, { sampleRate, inferenceQuantumSamples });
            !configurationError.empty())
        {
            if (startError != nullptr)
                *startError = configurationError;
            setLifecycleStateLocked(configurationError
                                        + "; the installed model stays silent — bounded dry pass-through",
                                    false);
            return false;
        }

        try
        {
            inferenceWorker.prepare(sampleRate, inferenceQuantumSamples, inferenceQueueCapacity);
        }
        catch (const std::exception& exception)
        {
            const auto cause = std::string("backend prepare failed: ") + exception.what();
            setLifecycleStateLocked(
                cause + "; the installed model stays silent — bounded dry pass-through", false);
            if (startError != nullptr)
                *startError = cause;
            return false;
        }
        catch (...)
        {
            const auto cause = std::string("backend prepare failed with an unknown exception");
            setLifecycleStateLocked(
                cause + "; the installed model stays silent — bounded dry pass-through", false);
            if (startError != nullptr)
                *startError = cause;
            return false;
        }

        inferenceWorker.threadStartHookForTesting = workerThreadStartHookForTesting;
        if (!inferenceWorker.start(startError))
        {
            const auto detail = startError != nullptr && !startError->empty()
                ? *startError
                : std::string("model could not be started");
            setLifecycleStateLocked(detail
                                        + "; the installed model stays silent — bounded dry pass-through",
                                    false);
            return false;
        }

        setLifecycleStateLocked({}, true);
        return true;
    }

    setLifecycleStateLocked({}, false);
    return true;
}

void RaveAudioEngine::release() noexcept
{
    const std::scoped_lock lock(lifecycleMutex);
    inferenceWorker.stop();
    // Publishing the usability transition is the point: pollers must render
    // installed-but-not-running after release/device stop instead of keeping
    // a stale active claim. The known host configuration is deliberately
    // kept, so later activations can still prepare, checked-reset, and start
    // against it. (Assigning an empty diagnostic cannot allocate, so this
    // stays noexcept-safe.)
    setLifecycleStateLocked({}, false);
}

bool RaveAudioEngine::commitResult(const float* const samples,
                                   const std::size_t count,
                                   const std::uint64_t frame,
                                   const std::uint64_t currentOutputFrame) noexcept
{
    if (samples == nullptr || count != inferenceQuantumSamples || frame >= submittedFrame)
    {
        alignmentErrors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (frame < currentOutputFrame)
    {
        lateResults.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const auto slot = static_cast<std::size_t>(frame % timelineFrameCount);
    if (wetFrameTags[slot] == frame)
    {
        alignmentErrors.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    std::copy_n(samples, inferenceQuantumSamples,
                wetTimeline.begin() + static_cast<std::ptrdiff_t>(slot * inferenceQuantumSamples));
    wetFrameTags[slot] = frame;
    return true;
}

bool RaveAudioEngine::publishResultForTesting(const float* const samples,
                                              const std::size_t count,
                                              const std::uint64_t frame) noexcept
{
    const auto outputFrame = sampleClock >= transportLatencySamples
        ? (sampleClock - transportLatencySamples) / inferenceQuantumSamples : 0;
    return commitResult(samples, count, frame, outputFrame);
}

void RaveAudioEngine::drainResults(const std::uint64_t currentOutputFrame) noexcept
{
    for (std::size_t drained = 0; drained < timelineFrameCount; ++drained)
    {
        std::size_t count = 0;
        std::uint64_t frame = 0;
        if (!inferenceWorker.tryReceive(receiveFrame.data(), receiveFrame.size(), count, &frame))
            break;
        static_cast<void>(commitResult(receiveFrame.data(), count, frame, currentOutputFrame));
    }
}

RaveAudioEngine::RuntimeTelemetry RaveAudioEngine::runtimeTelemetry() const noexcept
{
    return { transportLatencySamples,
             renderedSamples.load(std::memory_order_relaxed) >= transportLatencySamples,
             missedDeadlines.load(std::memory_order_relaxed),
             inferenceWorker.droppedInputBlockCount() + inferenceWorker.droppedOutputBlockCount(),
             lateResults.load(std::memory_order_relaxed),
             inferenceWorker.processingErrorCount(),
             inferenceWorker.resetErrorCount(),
             alignmentErrors.load(std::memory_order_relaxed) };
}

void RaveAudioEngine::processAudio(
    const float* const* inputChannelData,
    const int numInputChannels,
    float* const* outputChannelData,
    const int numOutputChannels,
    const int numSamples) noexcept
{
    if (numSamples <= 0 || outputChannelData == nullptr)
        return;

    const auto channels = callbackOutputChannels.load(std::memory_order_relaxed);
    if (dryTimeline.empty())
    {
        // Before the first prepare there is no declared transport; preserve
        // the legacy safe pass-through. Every prepared state uses the delay.
        for (int channel = 0; channel < numOutputChannels; ++channel)
            if (outputChannelData[channel] != nullptr)
            {
                const auto* input = inputChannelData != nullptr && channel < numInputChannels
                    ? inputChannelData[channel] : nullptr;
                if (input != nullptr)
                    juce::FloatVectorOperations::copy(outputChannelData[channel], input, numSamples);
                else
                    juce::FloatVectorOperations::clear(outputChannelData[channel], numSamples);
            }
        return;
    }
    if (numOutputChannels != channels)
    {
        for (int channel = 0; channel < numOutputChannels; ++channel)
            if (outputChannelData[channel] != nullptr)
                juce::FloatVectorOperations::clear(outputChannelData[channel], numSamples);
        alignmentErrors.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    for (int sample = 0; sample < numSamples; ++sample, ++sampleClock)
    {
        const auto dryIndex = static_cast<std::size_t>(sampleClock % dryRingSamples);
        for (int channel = 0; channel < channels; ++channel)
        {
            const auto* input = inputChannelData != nullptr && channel < numInputChannels
                ? inputChannelData[channel] : nullptr;
            dryTimeline[static_cast<std::size_t>(channel) * dryRingSamples + dryIndex]
                = input != nullptr ? input[sample] : 0.0f;
        }

        inputFrame[static_cast<std::size_t>(sampleClock % inferenceQuantumSamples)] =
            inputChannelData != nullptr && numInputChannels > 0 && inputChannelData[0] != nullptr
                ? inputChannelData[0][sample] : 0.0f;

        if ((sampleClock + 1) % inferenceQuantumSamples == 0)
        {
            const auto frame = submittedFrame++;
            if (inferenceWorker.isRunning()
                && !inferenceWorker.trySubmit(inputFrame.data(), inferenceQuantumSamples, frame))
            {
                // Queue-drop telemetry is owned by the worker. No timeline
                // state is changed: this frame deterministically falls back.
            }
        }

        const bool delayed = sampleClock >= transportLatencySamples;
        const auto delayedClock = delayed ? sampleClock - transportLatencySamples : 0;
        const auto outputFrame = delayedClock / inferenceQuantumSamples;
        if (delayed && delayedClock % inferenceQuantumSamples == 0)
        {
            drainResults(outputFrame);
            const auto slot = static_cast<std::size_t>(outputFrame % timelineFrameCount);
            if (inferenceWorker.isRunning() && wetFrameTags[slot] != outputFrame)
                missedDeadlines.fetch_add(1, std::memory_order_relaxed);
        }

        const auto targetMix = dryWet();
        if (smoothedDryWet < targetMix)
            smoothedDryWet = std::min(targetMix, smoothedDryWet + smoothingStep);
        else
            smoothedDryWet = std::max(targetMix, smoothedDryWet - smoothingStep);

        const auto wetSlot = static_cast<std::size_t>(outputFrame % timelineFrameCount);
        const bool wetAvailable = delayed && wetFrameTags[wetSlot] == outputFrame;
        const auto availabilityTarget = wetAvailable ? 1.0f : 0.0f;
        if (availability < availabilityTarget)
            availability = std::min(availabilityTarget, availability + smoothingStep);
        else
            availability = std::max(availabilityTarget, availability - smoothingStep);
        const auto wetGain = smoothedDryWet * availability;

        for (int channel = 0; channel < channels; ++channel)
        {
            auto* output = outputChannelData[channel];
            if (output == nullptr)
                continue;
            const auto dry = delayed
                ? dryTimeline[static_cast<std::size_t>(channel) * dryRingSamples
                              + static_cast<std::size_t>(delayedClock % dryRingSamples)]
                : 0.0f;
            const auto wet = wetAvailable
                ? wetTimeline[wetSlot * inferenceQuantumSamples
                              + static_cast<std::size_t>(delayedClock % inferenceQuantumSamples)]
                : dry;
            output[sample] = dry + (wet - dry) * wetGain;
        }
    }
    renderedSamples.store(sampleClock, std::memory_order_relaxed);
}

void RaveAudioEngine::audioDeviceAboutToStart(juce::AudioIODevice* const device)
{
    if (device == nullptr)
    {
        release();
        return;
    }

    prepare(device->getCurrentSampleRate(),
            static_cast<std::size_t>(std::max(1, device->getCurrentBufferSizeSamples())),
            device->getActiveOutputChannels().countNumberOfSetBits());
}

void RaveAudioEngine::audioDeviceStopped()
{
    release();
}

void RaveAudioEngine::audioDeviceIOCallbackWithContext(
    const float* const* inputChannelData,
    const int numInputChannels,
    float* const* outputChannelData,
    const int numOutputChannels,
    const int numSamples,
    const juce::AudioIODeviceCallbackContext&)
{
    processAudio(inputChannelData,
                 numInputChannels,
                 outputChannelData,
                 numOutputChannels,
                 numSamples);
}


} // namespace rave
