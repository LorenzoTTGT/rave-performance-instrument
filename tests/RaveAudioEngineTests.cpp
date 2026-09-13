#include "engine/InferenceWorker.h"
#include "engine/LifecycleStatusText.h"
#include "engine/RaveAudioEngine.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <latch>
#include <limits>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
void require(const bool condition, const char* const message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class TestAudioDevice final : public juce::AudioIODevice
{
public:
    TestAudioDevice() : AudioIODevice("test", "test") {}

    juce::StringArray getOutputChannelNames() override { return { "left", "right" }; }
    juce::StringArray getInputChannelNames() override { return { "left", "right" }; }
    juce::Array<double> getAvailableSampleRates() override { return { 48000.0 }; }
    juce::Array<int> getAvailableBufferSizes() override { return { 4 }; }
    int getDefaultBufferSize() override { return 4; }
    juce::String open(const juce::BigInteger&, const juce::BigInteger&, double, int) override { return {}; }
    void close() override {}
    bool isOpen() override { return true; }
    void start(juce::AudioIODeviceCallback*) override {}
    void stop() override {}
    bool isPlaying() override { return true; }
    juce::String getLastError() override { return {}; }
    int getCurrentBufferSizeSamples() override { return 4; }
    double getCurrentSampleRate() override { return 48000.0; }
    int getCurrentBitDepth() override { return 32; }
    juce::BigInteger getActiveOutputChannels() const override
    {
        juce::BigInteger channels;
        channels.setRange(0, 2, true);
        return channels;
    }
    juce::BigInteger getActiveInputChannels() const override { return getActiveOutputChannels(); }
    int getOutputLatencyInSamples() override { return 0; }
    int getInputLatencyInSamples() override { return 0; }
};

class GainBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f;
        return true;
    }
};

class LatentGainBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 2; }

    bool process(const std::span<const float> input,
                 const std::span<const float> latentControls,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f + latentControls[0];
        return true;
    }
};

class ThrowingPrepareBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override { throw std::runtime_error("prepare exploded"); }
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 5; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index];
        return true;
    }
};

class UnknownThrowingPrepareBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override { throw 7; }
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }
    bool process(std::span<const float>, std::span<const float>, std::span<float>) override
    {
        return true;
    }
};

class IncompatibleRateBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override { ++prepareCalls; }
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    [[nodiscard]] int modelSampleRate() const noexcept override { return 44100; }

    [[nodiscard]] bool supportsConfiguration(const double sampleRate,
                                             const std::size_t) const noexcept override
    {
        return sampleRate > 0.0
            && std::abs(sampleRate - static_cast<double>(modelSampleRate())) < 0.5;
    }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index];
        return true;
    }

    int prepareCalls = 0;
};

class IntermittentNonFiniteBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f;
        if (diverged.exchange(true))
            output[0] = std::numeric_limits<float>::quiet_NaN();
        return true;
    }

    std::atomic<bool> diverged { false };
};

// Reset refuses for calls in [resetCallsBeforeFailure, +resetFailCount);
// other lifecycle calls succeed. Optionally carries latent dimensions for
// rollback-coverage with a latent-capable previous backend.
class ResetCountingBackend final : public rave::ModelBackend
{
public:
    explicit ResetCountingBackend(const int resetCallsBeforeFailure,
                                  const std::size_t latentDimensions = 0,
                                  const int resetFailCount = std::numeric_limits<int>::max())
        : failFromCall(resetCallsBeforeFailure), failCount(resetFailCount), latentDimensions(latentDimensions) {}

    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}

    bool reset(std::string& errorMessage) override
    {
        const auto call = resetCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto offset = call - failFromCall;
        if (failCount > 0 && offset >= 0 && offset < failCount)
        {
            errorMessage = "reset refused on call " + std::to_string(call);
            return false;
        }
        return true;
    }

    std::size_t latentDimensionCount() const noexcept override { return latentDimensions; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index];
        return true;
    }

    std::atomic<int> resetCalls { 0 };
    int failFromCall = 1;
    int failCount = std::numeric_limits<int>::max();
    std::size_t latentDimensions = 0;
};

// A misbehaving backend whose reset actually throws from the Nth call,
// violating the checked-reset contract; containment must convert the throw.
class ThrowFromResetBackend final : public rave::ModelBackend
{
public:
    explicit ThrowFromResetBackend(const int throwFromCall, const int throwCount = 1)
        : throwFromCall(throwFromCall), throwCount(throwCount) {}

    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}

    bool reset(std::string&) override
    {
        const auto call = resetCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto offset = call - throwFromCall;
        if (throwCount > 0 && offset >= 0 && offset < throwCount)
            throw std::runtime_error("reset exploded on call " + std::to_string(call));
        return true;
    }

    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index];
        return true;
    }

    std::atomic<int> resetCalls { 0 };
    int throwFromCall = 1;
    int throwCount = 1;
};

class RateLoggingGainBackend final : public rave::ModelBackend
{
public:
    std::function<void(double)> onPrepare;

    bool load(const std::string&, std::string&) override { return true; }

    void prepare(const double sampleRate, const std::size_t) override
    {
        if (onPrepare != nullptr)
            onPrepare(sampleRate);
    }

    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f;
        return true;
    }
};

// A backend that only supports one exact sample rate, for incompatible-reprepare tests.
class FixedRateBackend final : public rave::ModelBackend
{
public:
    explicit FixedRateBackend(const int rate)
        : sampleRate(rate) {}

    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    [[nodiscard]] int modelSampleRate() const noexcept override { return sampleRate; }

    [[nodiscard]] bool supportsConfiguration(const double runtimeRate,
                                             const std::size_t) const noexcept override
    {
        return runtimeRate > 0.0
            && std::abs(runtimeRate - static_cast<double>(sampleRate)) < 0.5;
    }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f;
        return true;
    }

    int sampleRate;
};

// Prepare throws for calls in [prepareThrowFromCall, prepareThrowFromCall + prepareThrowCount); other lifecycle calls succeed.
class NthCallThrowingBackend final : public rave::ModelBackend
{
public:
    void prepare(double, std::size_t) override
    {
        const auto call = prepareCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto offset = call - prepareThrowFromCall;
        if (prepareThrowCount > 0 && offset >= 0 && offset < prepareThrowCount)
            throw std::runtime_error("prepare refuses on call " + std::to_string(call));
    }

    bool load(const std::string&, std::string&) override { return true; }
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index];
        return true;
    }

    std::atomic<int> prepareCalls { 0 };
    int prepareThrowFromCall = 0;
    int prepareThrowCount = 0;
};

void testDryFallbackWithoutModel()
{
    rave::RaveAudioEngine engine;
    std::array<float, 4> left { 1.0f, 2.0f, 3.0f, 4.0f };
    std::array<float, 4> right { 5.0f, 6.0f, 7.0f, 8.0f };
    const float* inputs[] { left.data(), right.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };

    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == left && outputRight == right, "no-model callback passes dry audio");
}

void testProcessedAudioUsesMatchingDelayedDryBlock()
{
    rave::RaveAudioEngine engine;
    engine.setModelBackend(std::make_shared<GainBackend>());
    engine.setDryWet(0.5f);

    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);

    bool observedProcessedBlock = false;
    for (int block = 1; block <= 40 && !observedProcessedBlock; ++block)
    {
        std::array<float, 4> left;
        std::array<float, 4> right;
        left.fill(static_cast<float>(100 + block));
        right.fill(static_cast<float>(1000 + block));
        const float* inputs[] { left.data(), right.data() };
        std::array<float, 4> outputLeft {};
        std::array<float, 4> outputRight {};
        float* outputs[] { outputLeft.data(), outputRight.data() };

        engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
        if (std::abs((outputRight[0] - outputLeft[0]) - 450.0f) < 0.001f)
            observedProcessedBlock = true;

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    engine.audioDeviceStopped();
    require(observedProcessedBlock, "engine emits aligned delayed dry/wet mix");
}
void testRuntimeConfigurationReporting()
{
    rave::RaveAudioEngine engine;
    auto configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 0.0 && configuration.maximumBlockSize == 0,
            "configuration is unset before the first prepare");

    engine.prepare(44100.0, 16, 1);
    configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 44100.0 && configuration.maximumBlockSize == 16,
            "prepared configuration reported to qualification");

    engine.release();
    configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 44100.0 && configuration.maximumBlockSize == 16,
            "last prepared configuration retained after release");
}

void testActivationRechecksCurrentConfiguration()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto good = std::make_shared<LatentGainBackend>();
    std::string failureReason;
    require(engine.activateModelBackend(good, &failureReason), "compatible candidate activates");
    require(failureReason.empty(), "successful activation reports no failure");
    static_cast<void>(engine.setLatentControl(0, 0.5f));

    // A candidate qualified for a different rate must be refused under
    // activation ownership without ever touching the active backend.
    auto incompatible = std::make_shared<IncompatibleRateBackend>();
    require(!engine.activateModelBackend(incompatible, &failureReason),
            "incompatible candidate is refused at activation");
    require(failureReason.find("sample rate") != std::string::npos
                && failureReason.find("44100") != std::string::npos
                && failureReason.find("48000") != std::string::npos,
            "refusal names the rate conflict");
    require(incompatible->prepareCalls == 0,
            "refused candidate is never prepared");
    require(engine.latentDimensionCount() == 2, "active model kept after refusal");
    require(engine.latentControl(0) == 0.5f, "active latent kept after refusal");

    require(!engine.activateModelBackend(nullptr, &failureReason), "empty candidate refused");
    require(failureReason.find("empty") != std::string::npos, "empty candidate diagnostic");

    rave::RaveAudioEngine unpreparedEngine;
    require(!unpreparedEngine.activateModelBackend(good, &failureReason),
            "activation is refused without a real runtime configuration");
    require(failureReason.find("No audio runtime configuration") != std::string::npos,
            "missing configuration diagnostic is actionable");

    engine.audioDeviceStopped();
}

bool runCallbacksUntilProcessed(rave::RaveAudioEngine& engine)
{
    for (int block = 1; block <= 80; ++block)
    {
        std::array<float, 4> left;
        std::array<float, 4> right;
        left.fill(static_cast<float>(100 + block));
        right.fill(static_cast<float>(1000 + block));
        const float* inputs[] { left.data(), right.data() };
        std::array<float, 4> outputLeft {};
        std::array<float, 4> outputRight {};
        float* outputs[] { outputLeft.data(), outputRight.data() };

        engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
        if (std::abs((outputRight[0] - outputLeft[0]) - 450.0f) < 0.001f)
            return true;

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return false;
}

void testFailedActivationPreservesPreviousBackend()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto good = std::make_shared<GainBackend>();
    require(engine.activateModelBackend(good), "first candidate activates");
    require(engine.hasModelBackend(), "active model configured");
    require(engine.runtimeConfiguration().sampleRate == 48000.0,
            "activation keeps the prepared configuration");

    require(!engine.activateModelBackend(nullptr), "empty candidate rejected");
    require(engine.hasModelBackend(), "empty candidate keeps previous model");

    // With the device still configured, the worker runs; activation stops it
    // internally and the rollback below re-prepares the previous backend.
    auto broken = std::make_shared<ThrowingPrepareBackend>();
    std::string failureReason;
    require(!engine.activateModelBackend(broken, &failureReason),
            "candidate whose prepare fails is rejected");
    require(failureReason.find("previous model retained") != std::string::npos,
            "rollback reports retention");
    require(failureReason.find("dry pass-through") == std::string::npos,
            "invariant prepare cause is not contaminated by transient dry state");
    const auto presentation = rave::contextualLifecycleStatus(
        "Model activation failed: "
            + juce::String(rave::candidateFailureCause(failureReason)),
        engine.lifecycleStatusSnapshot(), "previous.ts", "No model loaded");
    require(presentation.containsIgnoreCase("backend prepare failed")
                && presentation.containsIgnoreCase("active"),
            "throwing candidate cause is composed with the retained active state");
    require(!presentation.containsIgnoreCase("dry pass-through"),
            "retained rollback presentation has no stale dry-fallback claim");
    require(engine.hasModelBackend(), "previous model retained after failed activation");
    require(engine.latentDimensionCount() == 0, "previous latent count restored");

    auto unknownBroken = std::make_shared<UnknownThrowingPrepareBackend>();
    require(!engine.activateModelBackend(unknownBroken, &failureReason),
            "unknown candidate prepare exception is rejected");
    require(failureReason.find("unknown exception; previous model retained") != std::string::npos
                && failureReason.find("dry pass-through") == std::string::npos,
            "unknown prepare cause remains invariant before rollback outcome");

    engine.prepare(48000.0, 4, 2);
    engine.setDryWet(0.5f);
    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);
    require(runCallbacksUntilProcessed(engine), "restored backend still processes audio");

    // Activation while the worker is running joins it off the callback path.
    auto second = std::make_shared<GainBackend>();
    require(engine.activateModelBackend(second), "replacement activates while running");
    require(runCallbacksUntilProcessed(engine), "replacement backend processes audio");

    engine.audioDeviceStopped();
}

void testFailedActivationRestoresPreviousLatentControls()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto good = std::make_shared<LatentGainBackend>();
    std::string failureReason;
    require(engine.activateModelBackend(good, &failureReason), "latent model activates");

    require(engine.setLatentControl(0, 1.0f), "first latent set");
    require(engine.setLatentControl(1, -2.0f), "second latent set");

    auto broken = std::make_shared<ThrowingPrepareBackend>();
    require(!engine.activateModelBackend(broken, &failureReason),
            "candidate failure triggers rollback");
    require(engine.hasModelBackend(), "previous model retained");
    require(engine.latentDimensionCount() == 2, "previous latent count restored");
    require(engine.latentControl(0) == 1.0f, "first latent value restored");
    require(engine.latentControl(1) == -2.0f, "second latent value restored");

    // The restored worker must still process audio with the previous backend.
    engine.setDryWet(0.5f);
    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);
    require(runCallbacksUntilProcessed(engine), "restored backend still processes audio");
    engine.audioDeviceStopped();
}

void testCandidateResetFailureAtActivationRollsBackToPrevious()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto good = std::make_shared<GainBackend>();
    std::string failureReason;
    require(engine.activateModelBackend(good, &failureReason), "previous model activates");

    // The candidate's first reset models the qualification pass on the loader
    // thread; the second reset (performed at activation when the worker
    // starts) refuses, so activation must reject it and roll back.
    auto candidate = std::make_shared<ResetCountingBackend>(2);
    std::string qualificationResetError;
    require(candidate->reset(qualificationResetError),
            "candidate reset succeeds during qualification");
    require(candidate->resetCalls.load() == 1, "qualification reset consumed once");

    require(!engine.activateModelBackend(candidate, &failureReason),
            "candidate whose reset fails at activation is rejected");
    require(failureReason.find("model reset failed") != std::string::npos,
            "failure names the activation reset problem");
    require(failureReason.find("previous model retained") != std::string::npos,
            "failure claims retention only after a verified restart");
    require(engine.hasModelBackend(), "previous model installed after rollback");
    require(engine.hasUsableModel(), "previous model verified running after rollback");
    require(engine.latentDimensionCount() == 0, "previous latent metadata restored");

    // The verified rollback restart must have left the previous model playing.
    engine.setDryWet(0.5f);
    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);
    require(runCallbacksUntilProcessed(engine), "retained model plays after rollback restart");
    engine.audioDeviceStopped();
}

void testActivationFailureWithoutPreviousModelReportsDryFallback()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto candidate = std::make_shared<ResetCountingBackend>(1);
    std::string failureReason;
    require(!engine.activateModelBackend(candidate, &failureReason),
            "candidate with failing reset is rejected without a previous model");
    require(failureReason.find("no previous model") != std::string::npos
                && failureReason.find("dry pass-through") != std::string::npos,
            "no-previous-model diagnostic is truthful");
    require(!engine.hasModelBackend(), "no model is installed after the failure");

    // The engine remains in bounded dry pass-through.
    std::array<float, 4> input { 1.0f, 2.0f, 3.0f, 4.0f };
    const float* inputs[] { input.data(), input.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == input && outputRight == input,
            "engine passes dry audio after activation failure");
}

void testFailedRollbackRestartFallsBackToDry()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);

    // The previous model resets cleanly exactly once - its own activation -
    // so the rollback restart below will also fail.
    auto previous = std::make_shared<ResetCountingBackend>(2);
    std::string failureReason;
    require(engine.activateModelBackend(previous, &failureReason),
            "previous model with a one-shot reset activates");
    require(previous->resetCalls.load() == 1, "previous reset consumed by its activation");

    auto candidate = std::make_shared<ResetCountingBackend>(1);
    require(!engine.activateModelBackend(candidate, &failureReason),
            "candidate failure triggers a rollback that itself fails");
    require(failureReason.find("could not be restarted") != std::string::npos
                && failureReason.find("dry pass-through") != std::string::npos,
            "rollback failure diagnostic is truthful");
    require(!engine.hasModelBackend(), "unrestartable previous model is not claimed as retained");

    std::array<float, 4> input { 1.0f, 2.0f, 3.0f, 4.0f };
    const float* inputs[] { input.data(), input.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == input && outputRight == input,
            "engine passes dry audio after a failed rollback restart");
    require(engine.lifecycleDiagnostic().find("dry pass-through") != std::string::npos,
            "rollback failure leaves a lifecycle diagnostic");

    // A healthy replacement must still activate against the known
    // configuration after a failed rollback.
    auto replacement = std::make_shared<GainBackend>();
    require(engine.activateModelBackend(replacement, &failureReason),
            "healthy replacement activates after a failed rollback");
    require(engine.hasUsableModel(), "replacement is usable after activation");
    engine.setDryWet(0.5f);
    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);
    require(runCallbacksUntilProcessed(engine), "replacement processes audio after rollback failure");
    engine.audioDeviceStopped();
}

void testActivationCommitIsSerializedAgainstConcurrentReprepare()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);

    std::vector<std::string> log;
    std::mutex logMutex;
    const auto append = [&](const std::string& entry) {
        const std::scoped_lock logLock(logMutex);
        log.push_back(entry);
    };

    auto candidate = std::make_shared<RateLoggingGainBackend>();
    candidate->onPrepare = [&](const double rate) {
        append("prepare:" + std::to_string(static_cast<int>(rate)));
    };

    // Deterministic boundary protocol: prepare() fires the test seam at its
    // lifecycle-lock acquisition boundary; the seam counts down the latch so
    // the activation hook (holding lifecycleMutex) proceeds only once the
    // concurrent prepare has reached that boundary. Its next action is
    // acquiring the held mutex, so it is demonstrably pending on
    // serialization; the mutex - not the scheduler - then orders the two
    // critical sections.
    std::latch prepareAtBoundary { 1 };
    std::thread interleave;
    engine.prepareLockBoundaryForTesting = [&] { prepareAtBoundary.count_down(); };
    engine.activationInterleaveForTesting = [&]() {
        interleave = std::thread([&] {
            engine.prepare(44100.0, 4, 2);
            append("reprepare-done");
        });

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!prepareAtBoundary.try_wait()
               && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(prepareAtBoundary.try_wait(),
                "concurrent prepare reached the lifecycle-lock boundary before activation proceeded");
    };

    std::string failureReason;
    const bool activated = engine.activateModelBackend(candidate, &failureReason);
    require(activated && failureReason.empty(), "activation succeeds");
    require(interleave.joinable(), "interleave hook ran during activation");
    interleave.join();
    engine.activationInterleaveForTesting = nullptr;
    engine.prepareLockBoundaryForTesting = nullptr;

    // The concurrent prepare was provably pending at the lock boundary before
    // activation proceeded, so the mutex guarantees the ordering below: the
    // commit prepared the candidate at the exact activation snapshot (48 kHz),
    // and the 44.1 kHz reprepare ran only after the commit.
    require(log.size() == 3, "both lifecycle commits logged exactly once");
    require(log[0] == "prepare:48000", "activation committed the exact snapshot");
    require(log[1] == "prepare:44100", "concurrent reprepare ran after the commit");
    require(log[2] == "reprepare-done", "concurrent reprepare completed last");

    const auto configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 44100.0 && configuration.maximumBlockSize == 4,
            "serialized reprepare publishes its configuration");
    require(engine.hasUsableModel(), "candidate remains usable after the reprepare");

    engine.audioDeviceStopped();
}

void testActivationAfterReleaseChecksResetAndStart()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    engine.release();

    // Releasing the callback must not erase the known host configuration.
    const auto configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 48000.0 && configuration.maximumBlockSize == 4,
            "known configuration survives release");

    // A candidate whose reset fails is rejected truthfully even after release.
    auto failing = std::make_shared<ResetCountingBackend>(1);
    std::string failureReason;
    require(!engine.activateModelBackend(failing, &failureReason),
            "failing reset is rejected after release");
    require(failureReason.find("model reset failed") != std::string::npos
                && failureReason.find("no previous model to retain") != std::string::npos
                && failureReason.find("dry pass-through") != std::string::npos,
            "post-release reset failure reports truthful dry fallback");
    require(!engine.hasUsableModel(), "nothing usable after rejected post-release activation");

    // A healthy candidate is prepared, checked-reset, and started before
    // activation reports success, even though the callback is detached.
    auto healthy = std::make_shared<GainBackend>();
    require(engine.activateModelBackend(healthy, &failureReason),
            "healthy candidate activates after release");
    require(failureReason.empty(), "post-release activation reports no failure");
    require(engine.hasUsableModel(), "activated model is usable after release");

    engine.setDryWet(0.5f);
    require(runCallbacksUntilProcessed(engine),
            "worker started by post-release activation plays once audio returns");
    engine.audioDeviceStopped();
}

void testIncompatibleReprepareYieldsTruthfulDryFallback()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto fixed = std::make_shared<FixedRateBackend>(48000);
    std::string failureReason;
    require(engine.activateModelBackend(fixed, &failureReason),
            "48 kHz-only model activates at 48 kHz");
    require(engine.hasUsableModel(), "model usable after activation");

    engine.setDryWet(0.5f);
    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);
    require(runCallbacksUntilProcessed(engine), "model processes wet audio at its native rate");
    engine.audioDeviceStopped();

    // The host reproves at an incompatible rate: the installed model must be
    // reported as not usable while staying installed, the lifecycle diagnostic
    // must name the conflict, and the callback must pass nonzero dry input
    // through untouched.
    engine.prepare(44100.0, 4, 2);
    require(!engine.hasUsableModel(), "incompatible reprepare reports the model as not usable");
    require(engine.hasModelBackend(), "model stays installed across the incompatible reprepare");
    require(engine.lifecycleDiagnostic().find("44100") != std::string::npos
                && engine.lifecycleDiagnostic().find("48000") != std::string::npos
                && engine.lifecycleDiagnostic().find("dry pass-through") != std::string::npos,
            "lifecycle diagnostic names the incompatible configuration and dry fallback");

    std::array<float, 4> input { 0.25f, -0.5f, 0.75f, -1.0f };
    const float* inputs[] { input.data(), input.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == input && outputRight == input,
            "nonzero input passes through as bounded dry audio after incompatible reprepare");

    // Returning to the compatible configuration restores usability.
    engine.prepare(48000.0, 4, 2);
    require(engine.hasUsableModel(), "model becomes usable again at the compatible rate");
    require(engine.lifecycleDiagnostic().empty(),
            "lifecycle diagnostic clears when usability recovers");

    engine.audioDeviceStopped();
}

void testReleasePublishesUsabilityTransition()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);

    // With no model there is no usability transition, so release must not
    // spuriously bump the lifecycle revision.
    const auto revisionBeforeEmptyRelease = engine.lifecycleRevision();
    engine.release();
    require(engine.lifecycleRevision() == revisionBeforeEmptyRelease,
            "release without a model does not bump the revision");

    engine.prepare(48000.0, 4, 2);
    require(engine.activateModelBackend(std::make_shared<GainBackend>()), "model activates");
    require(engine.hasUsableModel(), "model usable after activation");

    // Release/device stop must publish the usability transition so pollers
    // render installed-but-not-running instead of a stale active claim.
    const auto revisionBeforeRelease = engine.lifecycleRevision();
    engine.release();
    require(engine.lifecycleRevision() != revisionBeforeRelease,
            "release publishes the usability transition");
    require(!engine.hasUsableModel(), "model not usable after release");
    require(engine.hasModelBackend(), "model stays installed after release");
    require(engine.lifecycleDiagnostic().empty(), "release is not a failure diagnostic");

    const auto released = rave::lifecycleStatusText(engine.lifecycleDiagnostic(),
                                                    engine.hasUsableModel(),
                                                    engine.hasModelBackend(),
                                                    {},
                                                    engine.latentDimensionCount(),
                                                    "No model loaded");
    require(released.containsIgnoreCase("not running"),
            "formatter renders installed-but-not-running after release");
    require(!released.containsIgnoreCase("active —"), "no active claim after release");

    // Reprepare after release restores usability and the active claim.
    engine.prepare(48000.0, 4, 2);
    require(engine.hasUsableModel(), "model usable again after reprepare");
    const auto recovered = rave::lifecycleStatusText(engine.lifecycleDiagnostic(),
                                                     engine.hasUsableModel(),
                                                     engine.hasModelBackend(),
                                                     {},
                                                     engine.latentDimensionCount(),
                                                     "No model loaded");
    require(recovered.containsIgnoreCase("active"), "formatter renders the active claim after recovery");

    engine.audioDeviceStopped();
}

void testPrepareThrowAtReprepareIsContained()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto backend = std::make_shared<NthCallThrowingBackend>();
    backend->prepareThrowFromCall = 2;
    backend->prepareThrowCount = 1;
    require(engine.activateModelBackend(backend), "model activates (prepare call 1 succeeds)");

    // The reprepare throws inside the backend prepare; it must be contained,
    // mark the model unusable, and publish an actionable diagnostic.
    engine.prepare(48000.0, 4, 2);
    require(!engine.hasUsableModel(), "backend prepare throw marks the model unusable");
    require(engine.hasModelBackend(), "model stays installed after the prepare throw");
    require(engine.lifecycleDiagnostic().find("backend prepare failed") != std::string::npos
                && engine.lifecycleDiagnostic().find("prepare refuses on call 2") != std::string::npos
                && engine.lifecycleDiagnostic().find("dry pass-through") != std::string::npos,
            "prepare-throw diagnostic is actionable");

    std::array<float, 4> input { 0.5f, -0.25f, 0.75f, -1.0f };
    const float* inputs[] { input.data(), input.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == input && outputRight == input,
            "nonzero input passes through as bounded dry audio after the prepare throw");

    // A later prepare beyond the throw window recovers usability.
    engine.prepare(48000.0, 4, 2);
    require(engine.hasUsableModel(), "model recovers after the throw window");
    require(engine.lifecycleDiagnostic().empty(), "diagnostic cleared on recovery");

    engine.audioDeviceStopped();
}

void testResetThrowAtReprepareIsContained()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    // This backend's reset actually throws from its second call, violating the
    // checked-reset contract; containment must convert the throw.
    auto backend = std::make_shared<ThrowFromResetBackend>(2, 1);
    require(engine.activateModelBackend(backend), "activation reset succeeds");

    engine.prepare(48000.0, 4, 2); // worker restart performs reset call 2, which throws
    require(!engine.hasUsableModel(), "throwing reprepare reset marks the model unusable");
    require(engine.lifecycleDiagnostic().find("model reset failed") != std::string::npos
                && engine.lifecycleDiagnostic().find("reset exploded on call 2") != std::string::npos
                && engine.lifecycleDiagnostic().find("dry pass-through") != std::string::npos,
            "reset-throw diagnostic is actionable");

    std::array<float, 4> input { 0.5f, -0.25f, 0.75f, -1.0f };
    const float* inputs[] { input.data(), input.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == input && outputRight == input,
            "nonzero input passes through as bounded dry audio after the reset throw");

    // A later prepare beyond the throw window recovers usability.
    engine.prepare(48000.0, 4, 2);
    require(engine.hasUsableModel(), "model recovers after the reset throw window");
    require(engine.lifecycleDiagnostic().empty(), "diagnostic cleared on recovery");

    engine.audioDeviceStopped();
}

void testThreadStartFailureAtReprepareIsContained()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    require(engine.activateModelBackend(std::make_shared<GainBackend>()), "model activates");

    // Injected thread-start failure at engine level: the prepare must contain
    // it, mark the model unusable, and present it through the shared presenter.
    engine.workerThreadStartHookForTesting = [] { throw std::runtime_error("no thread resources"); };
    engine.prepare(48000.0, 4, 2);
    engine.workerThreadStartHookForTesting = nullptr;
    require(!engine.hasUsableModel(), "thread-start failure marks the model unusable");
    require(engine.lifecycleDiagnostic().find("worker thread could not start") != std::string::npos
                && engine.lifecycleDiagnostic().find("no thread resources") != std::string::npos
                && engine.lifecycleDiagnostic().find("dry pass-through") != std::string::npos,
            "thread-start failure diagnostic is actionable");

    const auto status = rave::lifecycleStatusText(engine.lifecycleDiagnostic(),
                                                  engine.hasUsableModel(),
                                                  engine.hasModelBackend(),
                                                  {},
                                                  engine.latentDimensionCount(),
                                                  "No model loaded");
    require(status.containsIgnoreCase("worker thread could not start")
                && status.containsIgnoreCase("dry pass-through"),
            "status presenter renders the thread-start failure");
    require(!status.containsIgnoreCase("active —"), "no active claim while the worker cannot start");

    // Recovery once the injected failure is cleared.
    engine.prepare(48000.0, 4, 2);
    require(engine.hasUsableModel(), "model recovers after the thread-start failure");

    engine.audioDeviceStopped();
}

void testFailedRollbackWithLatentPreviousClearsWorkerLatentState()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    // Latent-capable previous whose rollback restart (reset call 2) fails.
    auto previous = std::make_shared<ResetCountingBackend>(2, 2);
    std::string failureReason;
    require(engine.activateModelBackend(previous, &failureReason),
            "latent previous model activates");
    require(engine.setLatentControl(0, 0.75f), "first latent set on the previous model");
    require(engine.setLatentControl(1, -0.25f), "second latent set on the previous model");

    auto candidate = std::make_shared<ResetCountingBackend>(1);
    require(!engine.activateModelBackend(candidate, &failureReason),
            "candidate failure triggers a rollback that itself fails");
    require(!engine.hasModelBackend(), "abandoned backend is not installed");
    require(engine.latentDimensionCount() == 0, "engine latent dimension count cleared");
    require(rave::standaloneLatentControlCount(engine.lifecycleStatusSnapshot()) == 0,
            "standalone presentation removes controls after backend abandonment");
    require(engine.latentControl(0) == 0.0f && engine.latentControl(1) == 0.0f,
            "worker latent storage cleared with the abandoned backend");
    require(engine.lifecycleDiagnostic().find("no usable model") != std::string::npos
                && engine.lifecycleDiagnostic().find("dry pass-through") != std::string::npos,
            "abandonment diagnostic is actionable");

    std::array<float, 4> input { 0.5f, -0.25f, 0.75f, -1.0f };
    const float* inputs[] { input.data(), input.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == input && outputRight == input,
            "engine passes dry audio after abandoning the backend");

    engine.audioDeviceStopped();
}

void testPostReattachStatusDerivesFromCurrentLifecycleState()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    // Succeeds during activation (reset call 1), refuses the next prepare
    // (reset call 2) — the reattachment sequence.
    auto backend = std::make_shared<ResetCountingBackend>(2, 0, 1);
    std::string failureReason;
    require(engine.activateModelBackend(backend, &failureReason),
            "activation succeeds before reattachment");
    engine.release(); // detach the audio callback

    // Reattachment runs a full prepare: the checked reset refuses, so the
    // status derived from current state must show the failure instead of an
    // active claim.
    engine.prepare(48000.0, 4, 2);
    const auto failed = rave::lifecycleStatusText(engine.lifecycleDiagnostic(),
                                                  engine.hasUsableModel(),
                                                  engine.hasModelBackend(),
                                                  {},
                                                  engine.latentDimensionCount(),
                                                  "Audio pass-through ready — no model loaded");
    require(failed.containsIgnoreCase("dry pass-through"),
            "post-reattach status names the dry fallback");
    require(!failed.containsIgnoreCase("active —"),
            "post-reattach status keeps no active claim after the failed prepare");

    // A later successful prepare derives the active claim again.
    engine.prepare(48000.0, 4, 2);
    const auto recovered = rave::lifecycleStatusText(engine.lifecycleDiagnostic(),
                                                     engine.hasUsableModel(),
                                                     engine.hasModelBackend(),
                                                     {},
                                                     engine.latentDimensionCount(),
                                                     "Audio pass-through ready — no model loaded");
    require(recovered.containsIgnoreCase("active"), "post-reattach status shows the active claim on recovery");

    engine.audioDeviceStopped();
}

void testCandidateFailureStatusUsesPostReattachState()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);

    // Reset 1 activates, reset 2 successfully restarts this backend during
    // candidate rollback, and reset 3 fails on callback reattachment.
    auto previous = std::make_shared<ResetCountingBackend>(3, 0, 1);
    std::string failureReason;
    require(engine.activateModelBackend(previous, &failureReason), "previous model activates");

    auto candidate = std::make_shared<ResetCountingBackend>(1);
    require(!engine.activateModelBackend(candidate, &failureReason),
            "candidate reset failure rolls back successfully");
    require(failureReason.find("reset refused on call 1") != std::string::npos,
            "activation failure preserves the candidate cause");
    require(failureReason.find("previous model retained") != std::string::npos,
            "activation-time outcome records successful rollback");

    engine.release();
    engine.prepare(48000.0, 4, 2); // callback reattachment reset now fails
    const auto current = rave::lifecycleStatusText(engine.lifecycleDiagnostic(),
                                                   engine.hasUsableModel(),
                                                   engine.hasModelBackend(),
                                                   {},
                                                   engine.latentDimensionCount(),
                                                   "Audio pass-through ready — no model loaded");
    const auto snapshot = engine.lifecycleStatusSnapshot();
    const rave::RaveAudioEngine::LifecycleStatusSnapshot abandonedSnapshot {
        snapshot.revision, false, false, "dry pass-through", 2
    };
    require(rave::standaloneLatentControlCount(abandonedSnapshot) == 0,
            "abandoned model removes stale standalone latent controls even from stale metadata");
    require(snapshot.revision == engine.lifecycleRevision()
                && snapshot.modelInstalled == engine.hasModelBackend()
                && snapshot.modelUsable == engine.hasUsableModel(),
            "unchanged lifecycle is represented consistently by the coherent snapshot");
    const auto candidateCause = rave::candidateFailureCause(failureReason);
    const auto combined = juce::String("Model activation failed: ")
        + juce::String(candidateCause)
        + juce::String(juce::CharPointer_UTF8(" — current state: "))
        + current;

    require(combined.contains("reset refused on call 1"),
            "post-reattach status preserves the invariant candidate cause");
    require(combined.containsIgnoreCase("dry pass-through"),
            "post-reattach status reports the authoritative current dry state");
    require(!combined.containsIgnoreCase("previous model retained"),
            "post-reattach status drops the stale activation-time retention outcome");
    require(!combined.containsIgnoreCase("active —"),
            "post-reattach status makes no contradictory active claim");

    engine.audioDeviceStopped();
}

void testLifecycleStatusFormatterCoversAllStates()
{
    require(rave::candidateFailureCause("backend; detail; previous model retained")
                == "backend; detail",
            "only the exact retained suffix is removed");
    require(rave::candidateFailureCause(";leading; internal diagnostic")
                == ";leading; internal diagnostic",
            "arbitrary leading and internal semicolons are preserved");
    require(rave::candidateFailureCause("; previous model retained").empty(),
            "empty candidate cause before an exact suffix is preserved");
    require(rave::candidateFailureCause("").empty(), "empty diagnostic stays empty");
    require(rave::candidateFailureCause("backend; previous model retained; extra")
                == "backend; previous model retained; extra",
            "recognized text that is not an exact suffix is preserved");
    require(rave::candidateFailureCause(
                "backend; inner; previous model could not be restarted: reset failed"
                " — remaining in bounded dry pass-through")
                == "backend; inner",
            "variable rollback detail is removed without truncating internal semicolons");

    const rave::RaveAudioEngine::LifecycleStatusSnapshot released {
        7, true, false, {}, 2
    };
    const auto failedAfterRelease = rave::contextualLifecycleStatus(
        "Model load failed: malformed metadata", released, "model.ts", "No model loaded");
    require(failedAfterRelease.containsIgnoreCase("model load failed")
                && failedAfterRelease.containsIgnoreCase("installed but not running")
                && !failedAfterRelease.containsIgnoreCase("active —"),
            "standalone failed-result presenter uses the captured released state");
    const rave::RaveAudioEngine::LifecycleStatusSnapshot laterFailure {
        8, true, false, "later reset failed — bounded dry pass-through", 2
    };
    const auto later = rave::contextualLifecycleStatus({}, laterFailure, "model.ts", "No model loaded");
    require(later.containsIgnoreCase("later reset failed"),
            "a later standalone timer snapshot supersedes failed-result presentation");

    const std::string diagnostic("Model sample rate 48000 Hz does not match the active 44100 Hz "
                                 "configuration; the installed model stays silent — bounded dry "
                                 "pass-through");
    const auto failure = rave::lifecycleStatusText(diagnostic, false, true, "model.ts", 2,
                                                   "No model loaded");
    require(failure == juce::String(juce::CharPointer_UTF8(diagnostic.c_str())),
            "failure diagnostic is rendered verbatim");

    const auto active = rave::lifecycleStatusText({}, true, true, "model.ts", 12, "No model loaded");
    require(active == juce::String(juce::CharPointer_UTF8("model.ts active — 12 latent dimensions")),
            "usable named model renders the active claim");

    const auto activeUnnamed = rave::lifecycleStatusText({}, true, true, {}, 3, "No model loaded");
    require(activeUnnamed
                == juce::String(juce::CharPointer_UTF8("Model active — 3 latent dimensions")),
            "usable unnamed model renders the generic active claim");

    const auto stopped = rave::lifecycleStatusText({}, false, true, "model.ts", 2, "No model loaded");
    require(stopped.containsIgnoreCase("not running") && stopped.containsIgnoreCase("dry pass-through"),
            "installed-but-silent model renders the not-running claim");

    const auto none = rave::lifecycleStatusText({}, false, false, {}, 0,
                                                "Audio pass-through ready — no model loaded");
    require(none.containsIgnoreCase("no model loaded"), "missing model renders the no-model text");
}

void testRepeatedPrepareReleaseCyclesRemainBounded()
{
    rave::RaveAudioEngine engine;
    engine.setModelBackend(std::make_shared<GainBackend>());
    for (int cycle = 0; cycle < 5; ++cycle)
    {
        engine.prepare(48000.0, 4, 2);
        require(engine.runtimeConfiguration().sampleRate == 48000.0, "cycle configuration");
        require(engine.hasModelBackend(), "cycle model retained");
        engine.release();
    }

    std::array<float, 4> left { 1.0f, 2.0f, 3.0f, 4.0f };
    const float* inputs[] { left.data(), left.data() };
    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight {};
    float* outputs[] { outputLeft.data(), outputRight.data() };
    engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
    require(outputLeft == left, "callback passes dry audio after repeated cycles");
}

void testRuntimeNonFiniteOutputFallsBackToDry()
{
    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 4, 2);
    auto backend = std::make_shared<IntermittentNonFiniteBackend>();
    require(engine.activateModelBackend(backend), "non-finite test model activates");
    engine.setDryWet(0.5f);

    TestAudioDevice device;
    engine.audioDeviceAboutToStart(&device);

    bool observedProcessedBlock = false;
    for (int block = 1; block <= 80; ++block)
    {
        std::array<float, 4> left;
        std::array<float, 4> right;
        left.fill(static_cast<float>(100 + block));
        right.fill(static_cast<float>(1000 + block));
        const float* inputs[] { left.data(), right.data() };
        std::array<float, 4> outputLeft {};
        std::array<float, 4> outputRight {};
        float* outputs[] { outputLeft.data(), outputRight.data() };

        engine.audioDeviceIOCallbackWithContext(inputs, 2, outputs, 2, 4, {});
        for (const auto sample : outputLeft)
            require(std::isfinite(sample), "output stays finite after runtime divergence");
        for (const auto sample : outputRight)
            require(std::isfinite(sample), "output stays finite after runtime divergence");
        if (std::abs((outputRight[0] - outputLeft[0]) - 450.0f) < 0.001f)
            observedProcessedBlock = true;

        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    engine.audioDeviceStopped();
    require(observedProcessedBlock, "diverging model produced valid blocks before divergence");
    require(engine.missedInferenceDeadlineCount() > 0,
            "dropped non-finite blocks surface as missed deadlines");
}
} // namespace

int main()
{
    testDryFallbackWithoutModel();
    testProcessedAudioUsesMatchingDelayedDryBlock();
    testRuntimeConfigurationReporting();
    testActivationRechecksCurrentConfiguration();
    testFailedActivationPreservesPreviousBackend();
    testFailedActivationRestoresPreviousLatentControls();
    testCandidateResetFailureAtActivationRollsBackToPrevious();
    testActivationFailureWithoutPreviousModelReportsDryFallback();
    testFailedRollbackRestartFallsBackToDry();
    testActivationAfterReleaseChecksResetAndStart();
    testIncompatibleReprepareYieldsTruthfulDryFallback();
    testActivationCommitIsSerializedAgainstConcurrentReprepare();
    testReleasePublishesUsabilityTransition();
    testPrepareThrowAtReprepareIsContained();
    testResetThrowAtReprepareIsContained();
    testThreadStartFailureAtReprepareIsContained();
    testFailedRollbackWithLatentPreviousClearsWorkerLatentState();
    testPostReattachStatusDerivesFromCurrentLifecycleState();
    testCandidateFailureStatusUsesPostReattachState();
    testLifecycleStatusFormatterCoversAllStates();
    testRepeatedPrepareReleaseCyclesRemainBounded();
    testRuntimeNonFiniteOutputFallsBackToDry();
    std::cout << "RaveAudioEngine tests passed\n";
}
