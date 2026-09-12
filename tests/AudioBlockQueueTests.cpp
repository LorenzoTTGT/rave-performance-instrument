#include "engine/AudioBlockQueue.h"

#include <array>
#include <cstdlib>
#include <iostream>

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

void testConfigurationAndEmptyQueue()
{
    rave::AudioBlockQueue queue;
    queue.prepare(2, 4, 2);

    require(queue.channels() == 2, "channel count");
    require(queue.maximumBlockSize() == 4, "maximum block size");
    require(queue.capacity() == 2, "queue capacity");

    std::array<float, 4> left {};
    std::array<float, 4> right {};
    float* output[] { left.data(), right.data() };
    std::size_t sampleCount = 99;
    require(!queue.tryPop(output, 2, 4, sampleCount), "empty queue must not pop");
    require(sampleCount == 0, "failed pop clears sample count");
}

void testRoundTripAndMissingInputChannel()
{
    rave::AudioBlockQueue queue;
    queue.prepare(2, 4, 2);

    const std::array<float, 4> left { 1.0f, 2.0f, 3.0f, 4.0f };
    const float* input[] { left.data(), nullptr };
    require(queue.tryPush(input, 2, left.size(), 42), "valid block must push");

    std::array<float, 4> outputLeft {};
    std::array<float, 4> outputRight { 9.0f, 9.0f, 9.0f, 9.0f };
    float* output[] { outputLeft.data(), outputRight.data() };
    std::size_t sampleCount = 0;
    std::uint64_t sequence = 0;
    require(queue.tryPop(output, 2, 4, sampleCount, &sequence), "queued block must pop");
    require(sampleCount == 4, "sample count round trip");
    require(sequence == 42, "sequence round trip");
    require(outputLeft == left, "audio samples round trip");
    require(outputRight == std::array<float, 4> {}, "missing channel becomes silence");
}

void testCapacityAndWrapAround()
{
    rave::AudioBlockQueue queue;
    queue.prepare(1, 2, 2);

    const std::array<float, 2> first { 1.0f, 2.0f };
    const std::array<float, 2> second { 3.0f, 4.0f };
    const std::array<float, 2> third { 5.0f, 6.0f };
    const float* firstInput[] { first.data() };
    const float* secondInput[] { second.data() };
    const float* thirdInput[] { third.data() };

    require(queue.tryPush(firstInput, 1, 2), "first block push");
    require(queue.tryPush(secondInput, 1, 2), "second block push");
    require(!queue.tryPush(thirdInput, 1, 2), "full queue rejects block");

    std::array<float, 2> output {};
    float* outputChannels[] { output.data() };
    std::size_t sampleCount = 0;
    require(queue.tryPop(outputChannels, 1, 2, sampleCount), "first block pop");
    require(output == first, "FIFO first block");
    require(queue.tryPush(thirdInput, 1, 2), "push after pop wraps");
    require(queue.tryPop(outputChannels, 1, 2, sampleCount), "second block pop");
    require(output == second, "FIFO second block");
    require(queue.tryPop(outputChannels, 1, 2, sampleCount), "wrapped block pop");
    require(output == third, "FIFO wrapped block");
}

void testInvalidShapesDoNotConsumeData()
{
    rave::AudioBlockQueue queue;
    queue.prepare(1, 2, 1);

    const std::array<float, 3> oversized { 1.0f, 2.0f, 3.0f };
    const float* oversizedInput[] { oversized.data() };
    require(!queue.tryPush(oversizedInput, 1, 3), "oversized block rejected");
    require(!queue.tryPush(oversizedInput, 0, 2), "channel mismatch rejected");

    const std::array<float, 2> valid { 7.0f, 8.0f };
    const float* validInput[] { valid.data() };
    require(queue.tryPush(validInput, 1, 2), "valid block push");

    std::array<float, 2> output {};
    float* outputChannels[] { output.data() };
    std::size_t sampleCount = 0;
    require(!queue.tryPop(outputChannels, 1, 1, sampleCount), "small destination rejected");
    require(queue.tryPop(outputChannels, 1, 2, sampleCount), "rejected pop preserves block");
    require(output == valid, "preserved block contents");
}
} // namespace

int main()
{
    testConfigurationAndEmptyQueue();
    testRoundTripAndMissingInputChannel();
    testCapacityAndWrapAround();
    testInvalidShapesDoNotConsumeData();
    std::cout << "AudioBlockQueue tests passed\n";
}
