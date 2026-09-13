#include "engine/InferenceWorker.h"

#include <chrono>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace rave
{
InferenceWorker::~InferenceWorker()
{
    stop();
}

void InferenceWorker::setBackend(ModelBackendPtr newBackend)
{
    if (isRunning())
        throw std::logic_error("Cannot replace an inference backend while the worker is running");

    backend = std::move(newBackend);

    // Keep latent storage in sync with the installed backend. Abandoning a
    // backend (nullptr) must clear the count, atomics, and snapshot so no
    // stale latent values survive a rollback failure or explicit clear.
    latentControlCount = backend != nullptr ? backend->latentDimensionCount() : 0;
    latentControlValues = latentControlCount > 0
        ? std::make_unique<std::atomic<float>[]>(latentControlCount)
        : nullptr;
    latentControlSnapshot.assign(latentControlCount, 0.0f);
    for (std::size_t index = 0; index < latentControlCount; ++index)
        latentControlValues[index].store(0.0f, std::memory_order_relaxed);
}

void InferenceWorker::prepare(const double sampleRate,
                              const std::size_t maximumSamplesPerBlock,
                              const std::size_t queueCapacity)
{
    if (isRunning())
        throw std::logic_error("Cannot prepare an inference worker while it is running");
    if (sampleRate <= 0.0 || maximumSamplesPerBlock == 0 || queueCapacity == 0)
        throw std::invalid_argument("Invalid inference worker configuration");

    inputQueue.prepare(1, maximumSamplesPerBlock, queueCapacity);
    outputQueue.prepare(1, maximumSamplesPerBlock, queueCapacity);
    inputBuffer.assign(maximumSamplesPerBlock, 0.0f);
    outputBuffer.assign(maximumSamplesPerBlock, 0.0f);
    // Latent storage/count is kept in sync by setBackend().

    droppedInputs.store(0, std::memory_order_relaxed);
    droppedOutputs.store(0, std::memory_order_relaxed);
    processingErrors.store(0, std::memory_order_relaxed);
    wakeSequence.store(0, std::memory_order_relaxed);

    if (backend != nullptr)
        backend->prepare(sampleRate, maximumSamplesPerBlock);
}

bool InferenceWorker::start(std::string* const startError)
{
    if (startError != nullptr)
        startError->clear();

    if (isRunning() || backend == nullptr || inputBuffer.empty())
    {
        if (startError != nullptr)
            *startError = "inference worker is not ready to start";
        return false;
    }

    inputQueue.reset();
    outputQueue.reset();
    // A misbehaving backend whose reset throws (violating the checked-reset
    // contract) must be contained here just like a false return.
    std::string resetError;
    auto resetSucceeded = false;
    try
    {
        resetSucceeded = backend->reset(resetError);
    }
    catch (const std::exception& exception)
    {
        resetError = exception.what();
    }
    catch (...)
    {
        resetError = "unknown reset exception";
    }
    if (!resetSucceeded)
    {
        // A backend whose reset fails must never run with unknown state; the
        // caller decides between verified rollback and bounded dry audio.
        if (startError != nullptr)
            *startError = resetError.empty()
                ? std::string("model reset failed")
                : "model reset failed: " + resetError;
        return false;
    }
    stopRequested.store(false, std::memory_order_release);
    running.store(true, std::memory_order_release);
    try
    {
        // Test-only seam fired just before the thread is constructed so tests
        // can exercise thread-start failure containment deterministically.
        if (threadStartHookForTesting != nullptr)
            threadStartHookForTesting();
        thread = std::thread([this] { run(); });
    }
    catch (const std::exception& exception)
    {
        // std::thread construction and any injected hook failure must never
        // escape the start path; the caller rolls back or falls back to dry.
        running.store(false, std::memory_order_release);
        stopRequested.store(true, std::memory_order_release);
        if (startError != nullptr)
            *startError = std::string("worker thread could not start: ") + exception.what();
        return false;
    }
    catch (...)
    {
        running.store(false, std::memory_order_release);
        stopRequested.store(true, std::memory_order_release);
        if (startError != nullptr)
            *startError = "worker thread could not start";
        return false;
    }
    return true;
}

void InferenceWorker::stop() noexcept
{
    if (thread.joinable())
    {
        stopRequested.store(true, std::memory_order_release);
        wakeCondition.notify_one();
        thread.join();
    }
    running.store(false, std::memory_order_release);
}

bool InferenceWorker::isRunning() const noexcept
{
    return running.load(std::memory_order_acquire);
}

bool InferenceWorker::trySubmit(const float* const samples,
                                const std::size_t sampleCount,
                                const std::uint64_t sequence) noexcept
{
    if (!isRunning() || samples == nullptr)
        return false;

    const float* channels[] { samples };
    if (!inputQueue.tryPush(channels, 1, sampleCount, sequence))
    {
        droppedInputs.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    wakeSequence.fetch_add(1, std::memory_order_release);
    wakeCondition.notify_one();
    return true;
}

bool InferenceWorker::tryReceive(float* const samples,
                                 const std::size_t sampleCapacity,
                                 std::size_t& sampleCount,
                                 std::uint64_t* const sequence) noexcept
{
    sampleCount = 0;
    if (samples == nullptr)
        return false;

    float* channels[] { samples };
    return outputQueue.tryPop(channels, 1, sampleCapacity, sampleCount, sequence);
}

std::size_t InferenceWorker::latentDimensionCount() const noexcept
{
    return latentControlCount;
}

float InferenceWorker::latentControl(const std::size_t index) const noexcept
{
    return index < latentControlCount && latentControlValues != nullptr
        ? latentControlValues[index].load(std::memory_order_relaxed)
        : 0.0f;
}

bool InferenceWorker::setLatentControl(const std::size_t index, const float value) noexcept
{
    if (index >= latentControlCount || latentControlValues == nullptr)
        return false;
    latentControlValues[index].store(value, std::memory_order_relaxed);
    return true;
}

std::uint64_t InferenceWorker::droppedInputBlockCount() const noexcept
{
    return droppedInputs.load(std::memory_order_relaxed);
}

std::uint64_t InferenceWorker::droppedOutputBlockCount() const noexcept
{
    return droppedOutputs.load(std::memory_order_relaxed);
}

std::uint64_t InferenceWorker::processingErrorCount() const noexcept
{
    return processingErrors.load(std::memory_order_relaxed);
}

void InferenceWorker::run()
{
    auto observedWakeSequence = wakeSequence.load(std::memory_order_acquire);

    while (!stopRequested.load(std::memory_order_acquire))
    {
        std::size_t sampleCount = 0;
        float* inputChannels[] { inputBuffer.data() };
        if (!inputQueue.tryPop(inputChannels, 1, inputBuffer.size(), sampleCount, &processingSequence))
        {
            std::unique_lock lock(wakeMutex);
            wakeCondition.wait_for(lock, std::chrono::milliseconds(2), [this, &observedWakeSequence] {
                return stopRequested.load(std::memory_order_acquire)
                    || wakeSequence.load(std::memory_order_acquire) != observedWakeSequence;
            });
            observedWakeSequence = wakeSequence.load(std::memory_order_acquire);
            continue;
        }

        for (std::size_t index = 0; index < latentControlCount; ++index)
            latentControlSnapshot[index] = latentControlValues[index].load(std::memory_order_relaxed);

        // A throwing backend must fail this block, never the worker thread.
        bool processed = false;
        try
        {
            processed = backend->process(
                std::span<const float>(inputBuffer.data(), sampleCount),
                std::span<const float>(latentControlSnapshot.data(), latentControlSnapshot.size()),
                std::span<float>(outputBuffer.data(), sampleCount));
        }
        catch (...)
        {
            processed = false;
        }

        if (!processed)
        {
            processingErrors.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // A stateful model can diverge at runtime; never emit non-finite audio.
        // The engine falls back to dry output when no valid block arrives.
        bool outputFinite = true;
        for (std::size_t index = 0; index < sampleCount; ++index)
        {
            if (!std::isfinite(outputBuffer[index]))
            {
                outputFinite = false;
                break;
            }
        }
        if (!outputFinite)
        {
            processingErrors.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const float* outputChannels[] { outputBuffer.data() };
        if (!outputQueue.tryPush(outputChannels, 1, sampleCount, processingSequence))
            droppedOutputs.fetch_add(1, std::memory_order_relaxed);
    }

    running.store(false, std::memory_order_release);
}
} // namespace rave
