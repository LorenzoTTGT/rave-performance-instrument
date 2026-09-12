#include "engine/InferenceWorker.h"

#include <array>
#include <atomic>
#include <chrono>
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

class GainBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }

    void prepare(const double newSampleRate, const std::size_t newMaximumBlockSize) override
    {
        sampleRate = newSampleRate;
        maximumBlockSize = newMaximumBlockSize;
    }

    void reset() noexcept override { resetCalled.store(true); }
    std::size_t latentDimensionCount() const noexcept override { return 2; }

    bool process(const std::span<const float> input,
                 const std::span<const float> latentControls,
                 const std::span<float> output) override
    {
        processThread = std::this_thread::get_id();
        if (failProcessing.load() || latentControls.size() != 2 || output.size() < input.size())
            return false;

        const auto latentOffset = latentControls[0] + latentControls[1];
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 2.0f + latentOffset;
        return true;
    }

    double sampleRate = 0.0;
    std::size_t maximumBlockSize = 0;
    std::atomic<bool> resetCalled { false };
    std::atomic<bool> failProcessing { false };
    std::thread::id processThread;
};

class ThrowingBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    void reset() noexcept override {}
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float>,
                 const std::span<const float>,
                 const std::span<float>) override
    {
        throw std::runtime_error("inference exploded");
    }
};

class NonFiniteBackend final : public rave::ModelBackend
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
            output[index] = input[index];
        output[0] = std::numeric_limits<float>::quiet_NaN();
        return true;
    }
};

bool receiveWithin(rave::InferenceWorker& worker,
                   float* output,
                   const std::size_t capacity,
                   std::size_t& sampleCount,
                   std::uint64_t* const sequence = nullptr)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (worker.tryReceive(output, capacity, sampleCount, sequence))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

void testLifecycleAndProcessing()
{
    const auto callerThread = std::this_thread::get_id();
    auto backend = std::make_shared<GainBackend>();
    rave::InferenceWorker worker;

    require(!worker.start(), "worker cannot start before configuration");
    worker.setBackend(backend);
    worker.prepare(48000.0, 8, 2);
    require(backend->sampleRate == 48000.0, "backend receives sample rate");
    require(backend->maximumBlockSize == 8, "backend receives block size");
    require(worker.start(), "configured worker starts");
    require(worker.isRunning(), "running state reported");
    require(backend->resetCalled.load(), "backend reset before processing");
    require(worker.latentDimensionCount() == 2, "worker exposes backend latent count");

    const std::array<float, 4> input { -1.0f, -0.25f, 0.5f, 1.0f };
    require(worker.trySubmit(input.data(), input.size(), 73), "audio block submitted");

    std::array<float, 8> output {};
    std::size_t sampleCount = 0;
    std::uint64_t receivedSequence = 0;
    require(receiveWithin(worker, output.data(), output.size(), sampleCount, &receivedSequence),
            "processed block received");
    require(sampleCount == input.size(), "processed sample count");
    require(receivedSequence == 73, "block sequence preserved");
    require(output[0] == -2.0f && output[1] == -0.5f && output[2] == 1.0f && output[3] == 2.0f,
            "backend output returned");
    require(backend->processThread != callerThread, "backend runs on worker thread");

    require(worker.setLatentControl(0, 1.0f) && worker.setLatentControl(1, 2.0f),
            "latent controls accepted");
    require(!worker.setLatentControl(2, 3.0f), "out-of-range latent rejected");
    require(worker.trySubmit(input.data(), input.size(), 74), "modulated block submitted");
    require(receiveWithin(worker, output.data(), output.size(), sampleCount, &receivedSequence),
            "modulated block received");
    require(receivedSequence == 74, "modulated block sequence preserved");
    require(output[0] == 1.0f && output[1] == 2.5f && output[2] == 4.0f && output[3] == 5.0f,
            "atomic latent snapshot reaches backend");

    worker.stop();
    require(!worker.isRunning(), "worker stops");
    require(!worker.trySubmit(input.data(), input.size()), "stopped worker rejects input");
}

void testProcessingErrorsAreCounted()
{
    auto backend = std::make_shared<GainBackend>();
    backend->failProcessing.store(true);

    rave::InferenceWorker worker;
    worker.setBackend(backend);
    worker.prepare(44100.0, 4, 1);
    require(worker.start(), "error-test worker starts");

    const std::array<float, 2> input { 1.0f, 2.0f };
    require(worker.trySubmit(input.data(), input.size()), "failing block submitted");

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.processingErrorCount() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

    require(worker.processingErrorCount() == 1, "processing failure counted");
    std::array<float, 4> output {};
    std::size_t sampleCount = 0;
    require(!worker.tryReceive(output.data(), output.size(), sampleCount), "failed processing emits no block");
    worker.stop();
}
void waitForProcessingError(rave::InferenceWorker& worker)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (worker.processingErrorCount() == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void testThrowingBackendIsCountedSafely()
{
    rave::InferenceWorker worker;
    worker.setBackend(std::make_shared<ThrowingBackend>());
    worker.prepare(48000.0, 4, 1);
    require(worker.start(), "throwing-backend worker starts");

    const std::array<float, 2> input { 1.0f, 2.0f };
    require(worker.trySubmit(input.data(), input.size()), "throwing block submitted");
    waitForProcessingError(worker);
    require(worker.processingErrorCount() >= 1, "thrown backend exception counted");

    std::array<float, 4> output {};
    std::size_t sampleCount = 99;
    require(!worker.tryReceive(output.data(), output.size(), sampleCount),
            "thrown backend exception emits no block");
    worker.stop();
    require(!worker.isRunning(), "throwing-backend worker stops cleanly");
}

void testNonFiniteOutputIsDropped()
{
    rave::InferenceWorker worker;
    worker.setBackend(std::make_shared<NonFiniteBackend>());
    worker.prepare(48000.0, 4, 1);
    require(worker.start(), "non-finite worker starts");

    const std::array<float, 2> input { 1.0f, 2.0f };
    require(worker.trySubmit(input.data(), input.size()), "non-finite block submitted");
    waitForProcessingError(worker);
    require(worker.processingErrorCount() >= 1, "non-finite output counted as processing error");

    std::array<float, 4> output {};
    std::size_t sampleCount = 99;
    require(!worker.tryReceive(output.data(), output.size(), sampleCount),
            "non-finite output is never emitted");
    worker.stop();
}

void testRepeatedPrepareStartStopAndReplacementRemainsBounded()
{
    rave::InferenceWorker worker;
    for (int cycle = 0; cycle < 3; ++cycle)
    {
        auto backend = std::make_shared<GainBackend>();
        worker.setBackend(backend);
        worker.prepare(48000.0, 4, 2);
        require(worker.start(), "cycle worker starts");
        require(worker.currentBackend() == backend, "active backend exposed while stopped");

        const std::array<float, 2> input { 0.5f, 1.0f };
        require(worker.trySubmit(input.data(), input.size(),
                                 static_cast<std::uint64_t>(cycle + 1)),
                "cycle block submitted");

        std::array<float, 4> output {};
        std::size_t sampleCount = 0;
        std::uint64_t sequence = 0;
        require(receiveWithin(worker, output.data(), output.size(), sampleCount, &sequence),
                "cycle block processed");
        require(sampleCount == input.size(), "cycle sample count");
        require(sequence == static_cast<std::uint64_t>(cycle + 1), "cycle sequence");
        require(output[0] == 1.0f && output[1] == 2.0f, "cycle backend output");

        worker.stop();
        require(!worker.isRunning(), "cycle worker stops");
    }

    auto runningBackend = std::make_shared<GainBackend>();
    worker.setBackend(runningBackend);
    worker.prepare(48000.0, 4, 2);
    require(worker.start(), "replacement-refusal worker starts");
    bool refused = false;
    try
    {
        worker.setBackend(std::make_shared<GainBackend>());
    }
    catch (const std::logic_error&)
    {
        refused = true;
    }
    require(refused, "replacement refused while the worker is running");
    require(worker.currentBackend() == runningBackend, "running backend unchanged after refusal");
    worker.stop();
}
} // namespace

int main()
{
    testLifecycleAndProcessing();
    testProcessingErrorsAreCounted();
    testThrowingBackendIsCountedSafely();
    testNonFiniteOutputIsDropped();
    testRepeatedPrepareStartStopAndReplacementRemainsBounded();
    std::cout << "InferenceWorker tests passed\n";
}
