#include "state/StandaloneSessionState.h"

#include <algorithm>
#include <cmath>
#include <thread>

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
    for (auto& bank : controlBanks)
        for (auto& mapping : bank.mappings)
            mapping.store(-1, std::memory_order_relaxed);
}

StandaloneSessionState::ControlBank*
StandaloneSessionState::acquireActiveControlBank() const noexcept
{
    constexpr auto closedBit = std::uint32_t { 1 } << 31;
    for (;;)
    {
        const auto index = activeControlBank.load(std::memory_order_acquire);
        auto& bank = controlBanks[index];
        auto state = bank.readerState.load(std::memory_order_acquire);
        if (readerAdmissionInterleaveForTesting)
            readerAdmissionInterleaveForTesting();
        if ((state & closedBit) != 0)
            continue;
        if (!bank.readerState.compare_exchange_weak(
                state, state + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            continue;
        if (activeControlBank.load(std::memory_order_acquire) == index)
            return &bank;
        bank.readerState.fetch_sub(1, std::memory_order_release);
    }
}

void StandaloneSessionState::releaseControlBank(ControlBank& bank) noexcept
{
    bank.readerState.fetch_sub(1, std::memory_order_release);
}

void StandaloneSessionState::setLatentCount(std::size_t newCount)
{
    newCount = std::min(newCount, maximumLatents);
    const juce::ScopedLock lock(controlLock);
    auto& bank = controlBanks[activeControlBank.load(std::memory_order_acquire)];
    const auto oldCount = bank.count.load(std::memory_order_relaxed);

    if (newCount < oldCount)
    {
        bank.count.store(newCount, std::memory_order_release);
        for (auto index = newCount; index < oldCount; ++index)
        {
            bank.latentValues[index].store(0.0f, std::memory_order_relaxed);
            bank.mappings[index + 1].store(-1, std::memory_order_relaxed);
        }
    }
    else if (newCount > oldCount)
    {
        for (auto index = oldCount; index < newCount; ++index)
        {
            bank.latentValues[index].store(0.0f, std::memory_order_relaxed);
            bank.mappings[index + 1].store(-1, std::memory_order_relaxed);
        }
        bank.count.store(newCount, std::memory_order_release);
    }
}

std::size_t StandaloneSessionState::latentCount() const noexcept
{
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return 0;
    const auto result = bank->count.load(std::memory_order_acquire);
    releaseControlBank(*bank);
    return result;
}

float StandaloneSessionState::latent(const std::size_t index) const noexcept
{
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return 0.0f;
    const auto currentCount = bank->count.load(std::memory_order_acquire);
    const auto result = index < currentCount
        ? bank->latentValues[index].load(std::memory_order_relaxed)
        : 0.0f;
    releaseControlBank(*bank);
    return result;
}

bool StandaloneSessionState::setLatent(const std::size_t index, const float value) noexcept
{
    if (!finite(value))
        return false;
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return false;
    const auto currentCount = bank->count.load(std::memory_order_acquire);
    const auto valid = index < currentCount;
    if (valid)
        bank->latentValues[index].store(
            std::clamp(value, -4.0f, 4.0f), std::memory_order_relaxed);
    releaseControlBank(*bank);
    return valid;
}

void StandaloneSessionState::setDryWet(const float value) noexcept
{
    if (!finite(value))
        return;
    if (auto* const bank = acquireActiveControlBank())
    {
        bank->mix.store(std::clamp(value, 0.0f, 1.0f), std::memory_order_relaxed);
        releaseControlBank(*bank);
    }
}

float StandaloneSessionState::dryWet() const noexcept
{
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return 0.0f;
    const auto result = bank->mix.load(std::memory_order_relaxed);
    releaseControlBank(*bank);
    return result;
}

void StandaloneSessionState::beginMidiLearn(const int target) noexcept
{
    if (target == dryWetTarget
        || (target >= 0 && static_cast<std::size_t>(target) < latentCount()))
        learning.store(target, std::memory_order_release);
}

void StandaloneSessionState::clearMidiMapping(const int target) noexcept
{
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return;
    const auto currentCount = bank->count.load(std::memory_order_acquire);
    if (target == dryWetTarget)
        bank->mappings[0].store(-1, std::memory_order_relaxed);
    else if (target >= 0 && static_cast<std::size_t>(target) < currentCount)
        bank->mappings[static_cast<std::size_t>(target) + 1].store(
            -1, std::memory_order_relaxed);
    releaseControlBank(*bank);
}

int StandaloneSessionState::midiController(const int target) const noexcept
{
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return -1;
    const auto currentCount = bank->count.load(std::memory_order_acquire);
    auto result = -1;
    if (target == dryWetTarget)
        result = bank->mappings[0].load(std::memory_order_relaxed);
    else if (target >= 0 && static_cast<std::size_t>(target) < currentCount)
        result = bank->mappings[static_cast<std::size_t>(target) + 1].load(
            std::memory_order_relaxed);
    releaseControlBank(*bank);
    return result;
}

bool StandaloneSessionState::applyMidiCc(const int controller, const int value) noexcept
{
    if (controller < 0 || controller > 127 || value < 0 || value > 127)
        return false;
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
        return false;

    const auto currentCount = bank->count.load(std::memory_order_acquire);
    const auto learnedTarget = learning.exchange(noLearningTarget, std::memory_order_acq_rel);
    const bool validLearnTarget = learnedTarget == dryWetTarget
        || (learnedTarget >= 0
            && static_cast<std::size_t>(learnedTarget) < currentCount);
    if (validLearnTarget)
    {
        for (std::size_t index = 0; index <= currentCount; ++index)
        {
            if (bank->mappings[index].load(std::memory_order_relaxed) == controller)
                bank->mappings[index].store(-1, std::memory_order_relaxed);
        }
        const auto mappingIndex = learnedTarget == dryWetTarget
            ? std::size_t { 0 }
            : static_cast<std::size_t>(learnedTarget) + 1;
        bank->mappings[mappingIndex].store(controller, std::memory_order_relaxed);
    }

    bool applied = false;
    const auto normalized = static_cast<float>(value) / 127.0f;
    if (bank->mappings[0].load(std::memory_order_relaxed) == controller)
    {
        bank->mix.store(normalized, std::memory_order_relaxed);
        applied = true;
    }
    for (std::size_t index = 0; index < currentCount; ++index)
    {
        if (bank->mappings[index + 1].load(std::memory_order_relaxed) == controller)
        {
            bank->latentValues[index].store(
                -4.0f + 8.0f * normalized, std::memory_order_relaxed);
            applied = true;
        }
    }
    releaseControlBank(*bank);
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

void StandaloneSessionState::setGenerator(bool enabled, float depth, float rate)
{
    if (!finite(depth) || !finite(rate))
        return;
    const juce::ScopedLock lock(controlLock);
    identity.generate = enabled;
    identity.motionDepth = std::clamp(depth, 0.0f, 2.0f);
    identity.motionRate = std::clamp(rate, 0.01f, 2.0f);
}

StandaloneSessionState::Snapshot StandaloneSessionState::snapshot() const
{
    const juce::ScopedLock lock(controlLock);
    auto result = identity;
    auto* const bank = acquireActiveControlBank();
    if (bank == nullptr)
    {
        result.dryWet = 0.0f;
        result.latents.clear();
        result.midiControllers.assign(1, -1);
        return result;
    }
    const auto currentCount = bank->count.load(std::memory_order_acquire);
    result.dryWet = bank->mix.load(std::memory_order_relaxed);
    result.latents.resize(currentCount);
    result.midiControllers.resize(currentCount + 1);
    result.midiControllers[0] = bank->mappings[0].load(std::memory_order_relaxed);
    for (std::size_t index = 0; index < currentCount; ++index)
    {
        result.latents[index] = bank->latentValues[index].load(std::memory_order_relaxed);
        result.midiControllers[index + 1]
            = bank->mappings[index + 1].load(std::memory_order_relaxed);
    }
    releaseControlBank(*bank);
    return result;
}

bool StandaloneSessionState::restore(const Snapshot& value)
{
    if (value.latents.size() > maximumLatents || value.midiControllers.size() != value.latents.size() + 1 ||
        !finite(value.dryWet) || !finite(value.motionDepth) || !finite(value.motionRate) || value.motionDepth < 0.0f ||
        value.motionDepth > 2.0f || value.motionRate < 0.01f || value.motionRate > 2.0f ||
        static_cast<std::size_t>(value.modelPath.length()) > maximumStringLength ||
        static_cast<std::size_t>(value.midiInputId.length()) > maximumStringLength ||
        static_cast<std::size_t>(value.audioDeviceType.length()) > maximumStringLength ||
        static_cast<std::size_t>(value.audioOutputId.length()) > maximumStringLength ||
        static_cast<std::size_t>(value.audioInputId.length()) > maximumStringLength)
        return false;

    for (const auto latentValue : value.latents)
        if (!finite(latentValue))
            return false;
    for (const auto controller : value.midiControllers)
        if (controller < -1 || controller > 127)
            return false;

    // Make all potentially allocating copies before entering the publication
    // transaction. The inactive bank then receives a complete fixed-capacity
    // control snapshot before one release-store makes it visible.
    auto restoredIdentity = value;
    const juce::ScopedLock lock(controlLock);
    const auto activeIndex = activeControlBank.load(std::memory_order_acquire);
    const auto inactiveIndex = std::size_t { 1 } - activeIndex;
    auto& bank = controlBanks[inactiveIndex];
    constexpr auto closedBit = std::uint32_t { 1 } << 31;
    constexpr auto readerMask = closedBit - 1;
    bank.readerState.fetch_or(closedBit, std::memory_order_acq_rel);
    while ((bank.readerState.load(std::memory_order_acquire) & readerMask) != 0)
        std::this_thread::yield();

    bank.count.store(value.latents.size(), std::memory_order_relaxed);
    bank.mix.store(std::clamp(value.dryWet, 0.0f, 1.0f), std::memory_order_relaxed);
    for (std::size_t index = 0; index < maximumLatents; ++index)
    {
        const auto enabled = index < value.latents.size();
        bank.latentValues[index].store(
            enabled ? std::clamp(value.latents[index], -4.0f, 4.0f) : 0.0f,
            std::memory_order_relaxed);
        bank.mappings[index + 1].store(
            enabled ? value.midiControllers[index + 1] : -1,
            std::memory_order_relaxed);
    }
    bank.mappings[0].store(value.midiControllers[0], std::memory_order_relaxed);
    identity = std::move(restoredIdentity);
    learning.store(noLearningTarget, std::memory_order_release);
    if (restoreBeforePublishForTesting)
        restoreBeforePublishForTesting();
    bank.readerState.store(0, std::memory_order_release);
    activeControlBank.store(inactiveIndex, std::memory_order_release);
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
    xml.setAttribute("generate", state.generate);
    xml.setAttribute("motionDepth", state.motionDepth);
    xml.setAttribute("motionRate", state.motionRate);
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
    state.generate = xml->getBoolAttribute("generate", false);
    state.motionDepth = static_cast<float>(xml->getDoubleAttribute("motionDepth", 0.5));
    state.motionRate = static_cast<float>(xml->getDoubleAttribute("motionRate", 0.1));

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
