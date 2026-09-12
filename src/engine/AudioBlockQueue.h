#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rave
{
// A bounded single-producer/single-consumer queue for planar audio blocks.
// Call prepare only while the producer and consumer are stopped. tryPush and
// tryPop are allocation-free and non-blocking after preparation.
class AudioBlockQueue
{
public:
    void prepare(std::size_t channelCount,
                 std::size_t maximumSamplesPerBlock,
                 std::size_t blockCapacity);
    void reset() noexcept;

    [[nodiscard]] bool tryPush(const float* const* channels,
                               std::size_t channelCount,
                               std::size_t sampleCount,
                               std::uint64_t sequence = 0) noexcept;
    [[nodiscard]] bool tryPop(float* const* channels,
                              std::size_t channelCount,
                              std::size_t channelCapacityInSamples,
                              std::size_t& sampleCount,
                              std::uint64_t* sequence = nullptr) noexcept;

    [[nodiscard]] std::size_t channels() const noexcept { return configuredChannelCount; }
    [[nodiscard]] std::size_t maximumBlockSize() const noexcept { return configuredMaximumBlockSize; }
    [[nodiscard]] std::size_t capacity() const noexcept;

private:
    struct Slot
    {
        std::vector<float> samples;
        std::size_t sampleCount = 0;
        std::uint64_t sequence = 0;
    };

    std::vector<Slot> slots;
    std::size_t configuredChannelCount = 0;
    std::size_t configuredMaximumBlockSize = 0;
    alignas(64) std::atomic<std::size_t> readIndex { 0 };
    alignas(64) std::atomic<std::size_t> writeIndex { 0 };
};
} // namespace rave
