#include "engine/RaveAudioEngine.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>

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
    void reset() noexcept override {}
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

class ThrowingPrepareBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override { throw std::runtime_error("prepare exploded"); }
    void reset() noexcept override {}
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

class IntermittentNonFiniteBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    void reset() noexcept override {}
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
    require(configuration.sampleRate == 48000.0 && configuration.maximumBlockSize == 512,
            "documented default configuration before prepare");

    engine.prepare(44100.0, 16, 1);
    configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 44100.0 && configuration.maximumBlockSize == 16,
            "prepared configuration reported to qualification");

    engine.release();
    configuration = engine.runtimeConfiguration();
    require(configuration.sampleRate == 44100.0 && configuration.maximumBlockSize == 16,
            "last prepared configuration retained after release");
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
    require(!engine.activateModelBackend(broken), "candidate whose prepare fails is rejected");
    require(engine.hasModelBackend(), "previous model retained after failed activation");
    require(engine.latentDimensionCount() == 0, "previous latent count restored");

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
    testFailedActivationPreservesPreviousBackend();
    testRepeatedPrepareReleaseCyclesRemainBounded();
    testRuntimeNonFiniteOutputFallsBackToDry();
    std::cout << "RaveAudioEngine tests passed\n";
}
