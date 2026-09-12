#include "engine/AudioBlockQueue.h"

#include <algorithm>

namespace rave
{
void AudioBlockQueue::prepare(const std::size_t channelCount,
                              const std::size_t maximumSamplesPerBlock,
                              const std::size_t blockCapacity)
{
    configuredChannelCount = channelCount;
    configuredMaximumBlockSize = maximumSamplesPerBlock;

    // One slot is reserved so equal read/write indices unambiguously mean empty.
    slots.clear();
    slots.resize(blockCapacity > 0 ? blockCapacity + 1 : 0);
    for (auto& slot : slots)
    {
        slot.samples.assign(channelCount * maximumSamplesPerBlock, 0.0f);
        slot.sampleCount = 0;
        slot.sequence = 0;
    }

    reset();
}

void AudioBlockQueue::reset() noexcept
{
    readIndex.store(0, std::memory_order_relaxed);
    writeIndex.store(0, std::memory_order_relaxed);
}

bool AudioBlockQueue::tryPush(const float* const* const channels,
                              const std::size_t channelCount,
                              const std::size_t sampleCount,
                              const std::uint64_t sequence) noexcept
{
    if (slots.size() < 2 || channels == nullptr || channelCount != configuredChannelCount
        || sampleCount > configuredMaximumBlockSize)
        return false;

    const auto write = writeIndex.load(std::memory_order_relaxed);
    const auto nextWrite = (write + 1) % slots.size();
    if (nextWrite == readIndex.load(std::memory_order_acquire))
        return false;

    auto& slot = slots[write];
    for (std::size_t channel = 0; channel < channelCount; ++channel)
    {
        auto* destination = slot.samples.data() + channel * configuredMaximumBlockSize;
        const auto* source = channels[channel];
        if (source != nullptr)
            std::copy_n(source, sampleCount, destination);
        else
            std::fill_n(destination, sampleCount, 0.0f);
    }
    slot.sampleCount = sampleCount;
    slot.sequence = sequence;

    writeIndex.store(nextWrite, std::memory_order_release);
    return true;
}

bool AudioBlockQueue::tryPop(float* const* const channels,
                             const std::size_t channelCount,
                             const std::size_t channelCapacityInSamples,
                             std::size_t& sampleCount,
                             std::uint64_t* const sequence) noexcept
{
    sampleCount = 0;
    if (sequence != nullptr)
        *sequence = 0;
    if (slots.size() < 2 || channels == nullptr || channelCount != configuredChannelCount)
        return false;

    const auto read = readIndex.load(std::memory_order_relaxed);
    if (read == writeIndex.load(std::memory_order_acquire))
        return false;

    const auto& slot = slots[read];
    if (channelCapacityInSamples < slot.sampleCount)
        return false;

    for (std::size_t channel = 0; channel < channelCount; ++channel)
    {
        auto* destination = channels[channel];
        if (destination == nullptr)
            continue;

        const auto* source = slot.samples.data() + channel * configuredMaximumBlockSize;
        std::copy_n(source, slot.sampleCount, destination);
    }
    sampleCount = slot.sampleCount;
    if (sequence != nullptr)
        *sequence = slot.sequence;

    readIndex.store((read + 1) % slots.size(), std::memory_order_release);
    return true;
}

std::size_t AudioBlockQueue::capacity() const noexcept
{
    return slots.empty() ? 0 : slots.size() - 1;
}
} // namespace rave
