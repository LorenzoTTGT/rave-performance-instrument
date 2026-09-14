#include "state/StandaloneSessionState.h"

#include <algorithm>
#include <cmath>

namespace rave
{
namespace
{
bool finite(const float value)
{
    return std::isfinite(value);
}
}

StandaloneSessionState::StandaloneSessionState()
{
    for (auto& mapping : mappings)
        mapping.store(-1, std::memory_order_relaxed);
}

void StandaloneSessionState::setLatentCount(std::size_t newCount)
{
    newCount = std::min(newCount, maximumLatents);
    const juce::ScopedLock lock(controlLock);
    const auto oldCount = count.load(std::memory_order_relaxed);

    if (newCount < oldCount)
    {
        // Publish a shrink before resetting inactive slots. A reader that observed
        // the old count can still safely access the fixed-capacity storage.
        count.store(newCount, std::memory_order_release);
        for (auto index = newCount; index < oldCount; ++index)
        {
            latentValues[index].store(0.0f, std::memory_order_relaxed);
            mappings[index + 1].store(-1, std::memory_order_relaxed);
        }
    }
    else if (newCount > oldCount)
    {
        // Initialize newly active slots before making them visible to readers.
        for (auto index = oldCount; index < newCount; ++index)
        {
            latentValues[index].store(0.0f, std::memory_order_relaxed);
            mappings[index + 1].store(-1, std::memory_order_relaxed);
        }
        count.store(newCount, std::memory_order_release);
    }
}

std::size_t StandaloneSessionState::latentCount() const noexcept
{
    return count.load(std::memory_order_acquire);
}

float StandaloneSessionState::latent(const std::size_t index) const noexcept
{
    const auto currentCount = latentCount();
    return index < currentCount
        ? latentValues[index].load(std::memory_order_relaxed)
        : 0.0f;
}

bool StandaloneSessionState::setLatent(const std::size_t index, const float value) noexcept
{
    const auto currentCount = latentCount();
    if (index >= currentCount || !finite(value))
        return false;

    latentValues[index].store(std::clamp(value, -4.0f, 4.0f), std::memory_order_relaxed);
    return true;
}

void StandaloneSessionState::setDryWet(const float value) noexcept
{
    if (finite(value))
        mix.store(std::clamp(value, 0.0f, 1.0f), std::memory_order_relaxed);
}

float StandaloneSessionState::dryWet() const noexcept
{
    return mix.load(std::memory_order_relaxed);
}

void StandaloneSessionState::beginMidiLearn(const int target) noexcept
{
    if (target == dryWetTarget
        || (target >= 0 && static_cast<std::size_t>(target) < latentCount()))
        learning.store(target, std::memory_order_release);
}

void StandaloneSessionState::clearMidiMapping(const int target) noexcept
{
    if (target == dryWetTarget)
    {
        mappings[0].store(-1, std::memory_order_relaxed);
        return;
    }

    if (target >= 0 && static_cast<std::size_t>(target) < latentCount())
        mappings[static_cast<std::size_t>(target) + 1].store(-1, std::memory_order_relaxed);
}

int StandaloneSessionState::midiController(const int target) const noexcept
{
    if (target == dryWetTarget)
        return mappings[0].load(std::memory_order_relaxed);

    if (target >= 0 && static_cast<std::size_t>(target) < latentCount())
        return mappings[static_cast<std::size_t>(target) + 1].load(std::memory_order_relaxed);

    return -1;
}

bool StandaloneSessionState::applyMidiCc(const int controller, const int value) noexcept
{
    if (controller < 0 || controller > 127 || value < 0 || value > 127)
        return false;

    const auto currentCount = latentCount();
    const auto learnedTarget = learning.exchange(noLearningTarget, std::memory_order_acq_rel);
    const bool validLearnTarget = learnedTarget == dryWetTarget
        || (learnedTarget >= 0
            && static_cast<std::size_t>(learnedTarget) < currentCount);
    if (validLearnTarget)
    {
        for (std::size_t index = 0; index <= currentCount; ++index)
        {
            if (mappings[index].load(std::memory_order_relaxed) == controller)
                mappings[index].store(-1, std::memory_order_relaxed);
        }

        const auto mappingIndex = learnedTarget == dryWetTarget
            ? std::size_t { 0 }
            : static_cast<std::size_t>(learnedTarget) + 1;
        mappings[mappingIndex].store(controller, std::memory_order_relaxed);
    }

    bool applied = false;
    const auto normalized = static_cast<float>(value) / 127.0f;
    if (mappings[0].load(std::memory_order_relaxed) == controller)
    {
        setDryWet(normalized);
        applied = true;
    }

    for (std::size_t index = 0; index < currentCount; ++index)
    {
        if (mappings[index + 1].load(std::memory_order_relaxed) == controller)
        {
            latentValues[index].store(-4.0f + 8.0f * normalized, std::memory_order_relaxed);
            applied = true;
        }
    }
    return applied;
}

void StandaloneSessionState::setModelPath(juce::String value)
{
    const juce::ScopedLock lock(controlLock);
    identity.modelPath = value.substring(0, static_cast<int>(maximumStringLength));
}

void StandaloneSessionState::setMidiInputId(juce::String value)
{
    const juce::ScopedLock lock(controlLock);
    identity.midiInputId = value.substring(0, static_cast<int>(maximumStringLength));
}

void StandaloneSessionState::setAudioSetup(juce::String type,
                                           juce::String output,
                                           juce::String input)
{
    const juce::ScopedLock lock(controlLock);
    identity.audioDeviceType = type.substring(0, static_cast<int>(maximumStringLength));
    identity.audioOutputId = output.substring(0, static_cast<int>(maximumStringLength));
    identity.audioInputId = input.substring(0, static_cast<int>(maximumStringLength));
}

StandaloneSessionState::Snapshot StandaloneSessionState::snapshot() const
{
    const juce::ScopedLock lock(controlLock);
    auto result = identity;
    const auto currentCount = latentCount();
    result.dryWet = dryWet();
    result.latents.resize(currentCount);
    result.midiControllers.resize(currentCount + 1);
    result.midiControllers[0] = mappings[0].load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < currentCount; ++index)
    {
        result.latents[index] = latentValues[index].load(std::memory_order_relaxed);
        result.midiControllers[index + 1]
            = mappings[index + 1].load(std::memory_order_relaxed);
    }
    return result;
}

bool StandaloneSessionState::restore(const Snapshot& value)
{
    if (value.latents.size() > maximumLatents
        || value.midiControllers.size() != value.latents.size() + 1
        || !finite(value.dryWet)
        || static_cast<std::size_t>(value.modelPath.length()) > maximumStringLength
        || static_cast<std::size_t>(value.midiInputId.length()) > maximumStringLength
        || static_cast<std::size_t>(value.audioDeviceType.length()) > maximumStringLength
        || static_cast<std::size_t>(value.audioOutputId.length()) > maximumStringLength
        || static_cast<std::size_t>(value.audioInputId.length()) > maximumStringLength)
        return false;

    for (const auto latentValue : value.latents)
        if (!finite(latentValue))
            return false;
    for (const auto controller : value.midiControllers)
        if (controller < -1 || controller > 127)
            return false;

    setLatentCount(value.latents.size());
    setDryWet(value.dryWet);
    for (std::size_t index = 0; index < value.latents.size(); ++index)
        static_cast<void>(setLatent(index, value.latents[index]));

    const juce::ScopedLock lock(controlLock);
    identity = value;
    for (std::size_t index = 0; index < value.midiControllers.size(); ++index)
        mappings[index].store(value.midiControllers[index], std::memory_order_relaxed);
    return true;
}

bool StandaloneSessionState::serialize(juce::MemoryBlock& output) const
{
    const auto state = snapshot();
    juce::XmlElement xml("RaveStandaloneState");
    xml.setAttribute("version", schemaVersion);
    xml.setAttribute("model", state.modelPath);
    xml.setAttribute("midiInput", state.midiInputId);
    xml.setAttribute("deviceType", state.audioDeviceType);
    xml.setAttribute("output", state.audioOutputId);
    xml.setAttribute("input", state.audioInputId);
    xml.setAttribute("dryWet", state.dryWet);
    xml.setAttribute("count", static_cast<int>(state.latents.size()));
    xml.setAttribute("cc0", state.midiControllers[0]);
    for (std::size_t index = 0; index < state.latents.size(); ++index)
    {
        xml.setAttribute("v" + juce::String(index), state.latents[index]);
        xml.setAttribute("cc" + juce::String(index + 1), state.midiControllers[index + 1]);
    }

    const auto text = xml.toString();
    const auto byteCount = text.getNumBytesAsUTF8();
    if (byteCount > maximumSerializedBytes)
        return false;
    output.replaceAll(text.toRawUTF8(), byteCount);
    return true;
}

bool StandaloneSessionState::readBoundedPresetFile(const juce::File& file,
                                                    juce::MemoryBlock& output)
{
    output.reset();
    auto stream = file.createInputStream();
    if (stream == nullptr)
        return false;

    constexpr auto readLimit = maximumSerializedBytes + 1;
    output.setSize(readLimit, false);
    auto bytesRead = std::size_t { 0 };
    while (bytesRead < readLimit)
    {
        const auto chunk = stream->read(static_cast<char*>(output.getData()) + bytesRead,
                                        static_cast<int>(readLimit - bytesRead));
        if (chunk <= 0)
            break;
        bytesRead += static_cast<std::size_t>(chunk);
    }
    if (bytesRead == 0 || bytesRead > maximumSerializedBytes)
    {
        output.reset();
        return false;
    }
    output.setSize(bytesRead, false);
    return true;
}

bool StandaloneSessionState::deserialize(const void* data, const std::size_t bytes)
{
    if (data == nullptr || bytes == 0 || bytes > maximumSerializedBytes)
        return false;

    const auto xml = juce::parseXML(juce::String::fromUTF8(
        static_cast<const char*>(data), static_cast<int>(bytes)));
    if (xml == nullptr || !xml->hasTagName("RaveStandaloneState")
        || xml->getIntAttribute("version") != schemaVersion)
        return false;

    Snapshot state;
    state.modelPath = xml->getStringAttribute("model");
    state.midiInputId = xml->getStringAttribute("midiInput");
    state.audioDeviceType = xml->getStringAttribute("deviceType");
    state.audioOutputId = xml->getStringAttribute("output");
    state.audioInputId = xml->getStringAttribute("input");
    state.dryWet = static_cast<float>(xml->getDoubleAttribute("dryWet"));

    const auto parsedCount = xml->getIntAttribute("count", -1);
    if (parsedCount < 0 || parsedCount > static_cast<int>(maximumLatents))
        return false;

    const auto parsedSize = static_cast<std::size_t>(parsedCount);
    state.latents.resize(parsedSize);
    state.midiControllers.resize(parsedSize + 1);
    state.midiControllers[0] = xml->getIntAttribute("cc0", -1);
    for (std::size_t index = 0; index < parsedSize; ++index)
    {
        state.latents[index] = static_cast<float>(
            xml->getDoubleAttribute("v" + juce::String(index)));
        state.midiControllers[index + 1] = xml->getIntAttribute(
            "cc" + juce::String(index + 1), -1);
    }
    return restore(state);
}
}
