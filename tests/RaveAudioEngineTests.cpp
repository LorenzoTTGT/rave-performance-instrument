#include "engine/RaveAudioEngine.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
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
} // namespace

int main()
{
    testDryFallbackWithoutModel();
    testProcessedAudioUsesMatchingDelayedDryBlock();
    std::cout << "RaveAudioEngine tests passed\n";
}
