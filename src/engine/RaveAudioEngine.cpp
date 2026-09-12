#include "engine/RaveAudioEngine.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace rave
{
void RaveAudioEngine::setModelBackend(ModelBackendPtr backend)
{
    if (inferenceWorker.isRunning())
        throw std::logic_error("Stop the audio device before replacing the model backend");

    modelLatentDimensionCount.store(
        backend != nullptr ? backend->latentDimensionCount() : 0,
        std::memory_order_release);
    modelBackendConfigured.store(backend != nullptr, std::memory_order_release);
    inferenceWorker.setBackend(std::move(backend));
}

bool RaveAudioEngine::hasModelBackend() const noexcept
{
    return modelBackendConfigured.load(std::memory_order_acquire);
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
    inferenceWorker.stop();
    nextSequence = 1;
    missedDeadlines.store(0, std::memory_order_relaxed);
    alignmentErrors.store(0, std::memory_order_relaxed);

    maximumBlockSize = std::max<std::size_t>(1, maximumSamplesPerBlock);
    configuredOutputChannels = std::max(1, outputChannelCount);

    constexpr std::size_t inferenceQueueCapacity = 4;
    constexpr std::size_t dryQueueCapacity = inferenceQueueCapacity * 3;
    dryBlockQueue.prepare(static_cast<std::size_t>(configuredOutputChannels),
                          maximumBlockSize,
                          dryQueueCapacity);
    wetBuffer.assign(maximumBlockSize, 0.0f);
    delayedDryBuffer.assign(maximumBlockSize * static_cast<std::size_t>(configuredOutputChannels), 0.0f);
    dryInputPointers.assign(static_cast<std::size_t>(configuredOutputChannels), nullptr);
    dryOutputPointers.resize(static_cast<std::size_t>(configuredOutputChannels));
    for (int channel = 0; channel < configuredOutputChannels; ++channel)
    {
        dryOutputPointers[static_cast<std::size_t>(channel)] =
            delayedDryBuffer.data() + static_cast<std::size_t>(channel) * maximumBlockSize;
    }

    if (hasModelBackend())
    {
        inferenceWorker.prepare(sampleRate, maximumBlockSize, inferenceQueueCapacity);
        static_cast<void>(inferenceWorker.start());
    }
}

void RaveAudioEngine::release() noexcept
{
    inferenceWorker.stop();
}

void RaveAudioEngine::processAudio(
    const float* const* inputChannelData,
    const int numInputChannels,
    float* const* outputChannelData,
    const int numOutputChannels,
    const int numSamples) noexcept
{
    if (!inferenceWorker.isRunning() || numInputChannels <= 0 || inputChannelData == nullptr
        || inputChannelData[0] == nullptr || numSamples <= 0
        || static_cast<std::size_t>(numSamples) > maximumBlockSize
        || numOutputChannels != configuredOutputChannels)
    {
        renderDry(inputChannelData,
                  numInputChannels,
                  outputChannelData,
                  numOutputChannels,
                  numSamples);
        return;
    }

    const auto sequence = nextSequence++;
    const auto submitted = inferenceWorker.trySubmit(
        inputChannelData[0], static_cast<std::size_t>(numSamples), sequence);

    if (submitted)
    {
        for (int channel = 0; channel < configuredOutputChannels; ++channel)
        {
            dryInputPointers[static_cast<std::size_t>(channel)] =
                channel < numInputChannels ? inputChannelData[channel] : nullptr;
        }

        if (!dryBlockQueue.tryPush(dryInputPointers.data(),
                                   dryInputPointers.size(),
                                   static_cast<std::size_t>(numSamples),
                                   sequence))
            alignmentErrors.fetch_add(1, std::memory_order_relaxed);
    }

    std::size_t wetSampleCount = 0;
    std::uint64_t wetSequence = 0;
    if (!inferenceWorker.tryReceive(
            wetBuffer.data(), wetBuffer.size(), wetSampleCount, &wetSequence))
    {
        missedDeadlines.fetch_add(1, std::memory_order_relaxed);
        renderDry(inputChannelData,
                  numInputChannels,
                  outputChannelData,
                  numOutputChannels,
                  numSamples);
        return;
    }

    std::size_t drySampleCount = 0;
    std::uint64_t drySequence = 0;
    bool foundMatchingDryBlock = false;
    while (dryBlockQueue.tryPop(dryOutputPointers.data(),
                                dryOutputPointers.size(),
                                maximumBlockSize,
                                drySampleCount,
                                &drySequence))
    {
        if (drySequence == wetSequence)
        {
            foundMatchingDryBlock = true;
            break;
        }
        if (drySequence > wetSequence)
            break;
    }

    if (!foundMatchingDryBlock || wetSampleCount != static_cast<std::size_t>(numSamples)
        || drySampleCount != wetSampleCount)
    {
        alignmentErrors.fetch_add(1, std::memory_order_relaxed);
        renderDry(inputChannelData,
                  numInputChannels,
                  outputChannelData,
                  numOutputChannels,
                  numSamples);
        return;
    }

    const auto wetGain = dryWet();
    const auto dryGain = 1.0f - wetGain;
    for (int channel = 0; channel < numOutputChannels; ++channel)
    {
        auto* const output = outputChannelData[channel];
        if (output == nullptr)
            continue;

        const auto* const dry = dryOutputPointers[static_cast<std::size_t>(channel)];
        for (int sample = 0; sample < numSamples; ++sample)
            output[sample] = dry[sample] * dryGain + wetBuffer[static_cast<std::size_t>(sample)] * wetGain;
    }
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

void RaveAudioEngine::renderDry(const float* const* inputChannelData,
                                const int numInputChannels,
                                float* const* outputChannelData,
                                const int numOutputChannels,
                                const int numSamples) noexcept
{
    for (int channel = 0; channel < numOutputChannels; ++channel)
    {
        auto* const output = outputChannelData[channel];
        if (output == nullptr)
            continue;

        const auto* const input = inputChannelData != nullptr && channel < numInputChannels
            ? inputChannelData[channel]
            : nullptr;
        if (input == nullptr)
            juce::FloatVectorOperations::clear(output, numSamples);
        else
            juce::FloatVectorOperations::copy(output, input, numSamples);
    }
}
} // namespace rave
