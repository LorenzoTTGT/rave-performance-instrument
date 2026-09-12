#pragma once

#include "engine/AudioBlockQueue.h"
#include "model/ModelBackend.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rave
{
// Runs one mono ModelBackend on a dedicated thread. Configuration and backend
// replacement are only permitted while stopped. Audio submission and retrieval
// are bounded, allocation-free, and non-blocking after prepare().
class InferenceWorker
{
public:
    InferenceWorker() = default;
    ~InferenceWorker();

    InferenceWorker(const InferenceWorker&) = delete;
    InferenceWorker& operator=(const InferenceWorker&) = delete;

    void setBackend(ModelBackendPtr newBackend);
    void prepare(double sampleRate,
                 std::size_t maximumSamplesPerBlock,
                 std::size_t queueCapacity = 4);

    [[nodiscard]] bool start();
    void stop() noexcept;
    [[nodiscard]] bool isRunning() const noexcept;

    [[nodiscard]] bool trySubmit(const float* samples,
                                 std::size_t sampleCount,
                                 std::uint64_t sequence = 0) noexcept;
    [[nodiscard]] bool tryReceive(float* samples,
                                  std::size_t sampleCapacity,
                                  std::size_t& sampleCount,
                                  std::uint64_t* sequence = nullptr) noexcept;

    [[nodiscard]] std::size_t latentDimensionCount() const noexcept;
    [[nodiscard]] float latentControl(std::size_t index) const noexcept;
    [[nodiscard]] bool setLatentControl(std::size_t index, float value) noexcept;

    [[nodiscard]] std::uint64_t droppedInputBlockCount() const noexcept;
    [[nodiscard]] std::uint64_t droppedOutputBlockCount() const noexcept;
    [[nodiscard]] std::uint64_t processingErrorCount() const noexcept;

private:
    void run();

    ModelBackendPtr backend;
    AudioBlockQueue inputQueue;
    AudioBlockQueue outputQueue;
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    std::unique_ptr<std::atomic<float>[]> latentControlValues;
    std::vector<float> latentControlSnapshot;
    std::size_t latentControlCount = 0;
    std::uint64_t processingSequence = 0;

    std::thread thread;
    std::mutex wakeMutex;
    std::condition_variable wakeCondition;
    std::atomic<bool> running { false };
    std::atomic<bool> stopRequested { false };
    std::atomic<std::uint64_t> wakeSequence { 0 };
    std::atomic<std::uint64_t> droppedInputs { 0 };
    std::atomic<std::uint64_t> droppedOutputs { 0 };
    std::atomic<std::uint64_t> processingErrors { 0 };
};
} // namespace rave
