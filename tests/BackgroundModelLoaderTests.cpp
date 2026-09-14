#include "model/BackgroundModelLoader.h"
#include "model/ModelQualification.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

struct BackendLog
{
    std::thread::id loadThread;
    std::thread::id processThread;
    int loadCalls = 0;
    int prepareCalls = 0;
    int resetCalls = 0;
    int processCalls = 0;
    int resetOrder = 0;
    int firstProcessOrder = 0;
    double preparedSampleRate = 0.0;
    std::size_t preparedBlockSize = 0;
    std::vector<std::size_t> warmUpBlockSizes;
};

class QualifiedTestBackend final : public rave::ModelBackend
{
public:
    enum class ProcessBehavior
    {
        succeed,
        fail,
        produceNonFinite,
        throwStd,
        throwUnknown
    };

    enum class ResetBehavior
    {
        succeed,
        fail,
        throwStd
    };

    QualifiedTestBackend(std::shared_ptr<BackendLog> newLog,
                         const int newDeclaredSampleRate = -1,
                         const ProcessBehavior newBehavior = ProcessBehavior::succeed)
        : log(std::move(newLog)), declaredSampleRate(newDeclaredSampleRate), behavior(newBehavior)
    {
    }

    bool load(const std::string& path, std::string& errorMessage) override
    {
        log->loadThread = std::this_thread::get_id();
        ++log->loadCalls;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (path == "load-fails")
        {
            errorMessage = "invalid test model";
            return false;
        }
        return true;
    }

    void prepare(const double sampleRate, const std::size_t maximumBlockSize) override
    {
        ++log->prepareCalls;
        log->preparedSampleRate = sampleRate;
        log->preparedBlockSize = maximumBlockSize;
        if (throwOnPrepare)
            throw std::runtime_error("prepare exploded");
    }

    bool reset(std::string& errorMessage) override
    {
        ++log->resetCalls;
        log->resetOrder = ++orderCounter;
        switch (resetBehavior)
        {
        case ResetBehavior::fail:
            errorMessage = "reset refused";
            return false;
        case ResetBehavior::throwStd:
            throw std::runtime_error("reset exploded");
        case ResetBehavior::succeed:
            return true;
        }
        return true;
    }

    [[nodiscard]] std::size_t latentDimensionCount() const noexcept override { return latentCount; }
    [[nodiscard]] int modelSampleRate() const noexcept override { return declaredSampleRate; }

    // Mirrors the real backend contract: a declared sample rate must match the
    // runtime configuration; models without a declared rate accept any rate.
    [[nodiscard]] bool supportsConfiguration(const double sampleRate,
                                             const std::size_t /*maximumBlockSize*/)
        const noexcept override
    {
        if (sampleRate <= 0.0)
            return false;
        return declaredSampleRate <= 0
            || std::abs(sampleRate - static_cast<double>(declaredSampleRate)) < 0.5;
    }

    bool process(const std::span<const float> input,
                 const std::span<const float> /*latentControls*/,
                 const std::span<float> output) override
    {
        log->processThread = std::this_thread::get_id();
        ++log->processCalls;
        if (log->firstProcessOrder == 0)
            log->firstProcessOrder = ++orderCounter;
        log->warmUpBlockSizes.push_back(input.size());

        switch (behavior)
        {
        case ProcessBehavior::fail:
            return false;
        case ProcessBehavior::produceNonFinite:
            output[0] = std::numeric_limits<float>::quiet_NaN();
            return true;
        case ProcessBehavior::throwStd:
            throw std::runtime_error("process exploded");
        case ProcessBehavior::throwUnknown:
            throw 42;
        case ProcessBehavior::succeed:
            break;
        }

        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index];
        return true;
    }

    std::shared_ptr<BackendLog> log;
    int declaredSampleRate = -1;
    ProcessBehavior behavior = ProcessBehavior::succeed;
    std::size_t latentCount = 0;
    ResetBehavior resetBehavior = ResetBehavior::succeed;
    bool throwOnPrepare = false;
    int orderCounter = 0;
};

bool waitForTerminalState(rave::BackgroundModelLoader& loader)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline)
    {
        const auto state = loader.state();
        if (state == rave::BackgroundModelLoader::State::succeeded
            || state == rave::BackgroundModelLoader::State::failed)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
}

void testSuccessfulLoadQualifiesAtIntendedConfiguration()
{
    const auto callerThread = std::this_thread::get_id();
    auto log = std::make_shared<BackendLog>();
    rave::BackgroundModelLoader loader([log] {
        return std::make_shared<QualifiedTestBackend>(log);
    });
    const rave::ModelRuntimeConfiguration configuration { 48000.0, 8 };

    require(loader.start("valid-model", configuration), "first load starts");
    require(loader.state() == rave::BackgroundModelLoader::State::loading,
            "loading state reported");
    require(!loader.start("another-model", configuration), "concurrent load rejected");
    require(waitForTerminalState(loader), "load reaches terminal state");

    auto result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::succeeded,
            "qualified candidate succeeds");
    require(result.backend != nullptr, "qualified backend returned");
    require(result.errorMessage.empty(), "successful qualification has no error");
    require(log->loadThread != callerThread, "load runs on background thread");
    require(log->loadCalls == 1, "only accepted load executed");
    require(log->prepareCalls == 1, "candidate prepared once during qualification");
    require(log->preparedSampleRate == 48000.0 && log->preparedBlockSize == 8,
            "qualification uses the intended runtime configuration");
    require(log->resetCalls == 1, "candidate reset before warm-up");
    require(log->resetOrder != 0 && log->resetOrder < log->firstProcessOrder,
            "reset precedes warm-up processing");
    require(log->processCalls == 2, "warm-up runs two streaming blocks");
    require(log->warmUpBlockSizes == std::vector<std::size_t> { 8, 8 },
            "warm-up uses the intended block size");
    require(log->processThread != callerThread, "warm-up runs off the caller thread");
    require(loader.state() == rave::BackgroundModelLoader::State::idle,
            "result consumption resets state");
}

void testFailedLoadCanBeFollowedBySuccess()
{
    auto log = std::make_shared<BackendLog>();
    rave::BackgroundModelLoader loader([log] {
        return std::make_shared<QualifiedTestBackend>(log);
    });

    require(loader.start("load-fails", rave::ModelRuntimeConfiguration { 48000.0, 8 }),
            "failing load starts");
    require(waitForTerminalState(loader), "failing load completes");
    auto failed = loader.takeResult();
    require(failed.state == rave::BackgroundModelLoader::State::failed, "failure state returned");
    require(failed.backend == nullptr, "failed backend not returned");
    require(failed.errorMessage == "invalid test model", "backend error preserved");

    require(loader.start("valid-model", rave::ModelRuntimeConfiguration { 48000.0, 8 }),
            "loader reusable after failure");
    require(waitForTerminalState(loader), "second load completes");
    auto succeeded = loader.takeResult();
    require(succeeded.state == rave::BackgroundModelLoader::State::succeeded,
            "second load succeeds");
    require(succeeded.backend != nullptr, "second backend returned");
}

void testSampleRateMismatchFailsQualification()
{
    auto log = std::make_shared<BackendLog>();
    rave::BackgroundModelLoader loader([log] {
        return std::make_shared<QualifiedTestBackend>(log, 44100);
    });

    require(loader.start("valid-model", rave::ModelRuntimeConfiguration { 48000.0, 8 }),
            "mismatched load starts");
    require(waitForTerminalState(loader), "mismatched load completes");
    auto result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::failed,
            "sample-rate mismatch fails qualification");
    require(result.backend == nullptr, "mismatched candidate is not returned");
    require(result.errorMessage.find("44100") != std::string::npos
                && result.errorMessage.find("48000") != std::string::npos,
            "mismatch message names both sample rates");
    require(log->prepareCalls == 0 && log->processCalls == 0,
            "incompatible candidate is never prepared or warmed up");

    require(loader.start("valid-model", rave::ModelRuntimeConfiguration { 44100.0, 8 }),
            "loader reusable after mismatch");
    require(waitForTerminalState(loader), "matching load completes");
    result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::succeeded,
            "matching sample rate qualifies");
    require(log->preparedSampleRate == 44100.0, "matching configuration prepared");
}

void testWarmUpFailuresProduceDiagnostics()
{
    struct Case
    {
        const char* name;
        QualifiedTestBackend::ProcessBehavior behavior;
        bool throwOnPrepare;
        QualifiedTestBackend::ResetBehavior resetBehavior;
        const char* expectedMessagePart;
    };

    const Case cases[] {
        { "warm-up processing failure",
          QualifiedTestBackend::ProcessBehavior::fail,
          false,
          QualifiedTestBackend::ResetBehavior::succeed,
          "warm-up" },
        { "non-finite warm-up output",
          QualifiedTestBackend::ProcessBehavior::produceNonFinite,
          false,
          QualifiedTestBackend::ResetBehavior::succeed,
          "non-finite" },
        { "throwing warm-up",
          QualifiedTestBackend::ProcessBehavior::throwStd,
          false,
          QualifiedTestBackend::ResetBehavior::succeed,
          "exception" },
        { "unknown thrown value",
          QualifiedTestBackend::ProcessBehavior::throwUnknown,
          false,
          QualifiedTestBackend::ResetBehavior::succeed,
          "unknown exception" },
        { "throwing prepare",
          QualifiedTestBackend::ProcessBehavior::succeed,
          true,
          QualifiedTestBackend::ResetBehavior::succeed,
          "exception" },
        { "failing reset",
          QualifiedTestBackend::ProcessBehavior::succeed,
          false,
          QualifiedTestBackend::ResetBehavior::fail,
          "Model reset failed" },
        { "throwing reset",
          QualifiedTestBackend::ProcessBehavior::succeed,
          false,
          QualifiedTestBackend::ResetBehavior::throwStd,
          "exception" },
    };

    for (const auto& testCase : cases)
    {
        auto log = std::make_shared<BackendLog>();
        rave::BackgroundModelLoader loader([log, &testCase] {
            auto backend = std::make_shared<QualifiedTestBackend>(log, -1, testCase.behavior);
            backend->throwOnPrepare = testCase.throwOnPrepare;
            backend->resetBehavior = testCase.resetBehavior;
            return backend;
        });

        require(loader.start("valid-model", rave::ModelRuntimeConfiguration { 48000.0, 8 }),
                testCase.name);
        require(waitForTerminalState(loader), testCase.name);
        auto result = loader.takeResult();
        require(result.state == rave::BackgroundModelLoader::State::failed, testCase.name);
        require(result.backend == nullptr, testCase.name);
        require(result.errorMessage.find(testCase.expectedMessagePart) != std::string::npos,
                testCase.name);
    }
}

void testInvalidRuntimeConfigurationFailsQualification()
{
    auto log = std::make_shared<BackendLog>();
    rave::BackgroundModelLoader loader([log] {
        return std::make_shared<QualifiedTestBackend>(log);
    });

    require(loader.start("valid-model", rave::ModelRuntimeConfiguration { 0.0, 8 }),
            "invalid sample rate load starts");
    require(waitForTerminalState(loader), "invalid sample rate load completes");
    auto result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::failed,
            "invalid sample rate fails qualification");
    require(result.errorMessage.find("No audio runtime configuration") != std::string::npos,
            "unset configuration message");
    require(result.backend == nullptr, "no backend for invalid configuration");

    require(loader.start("valid-model", rave::ModelRuntimeConfiguration { 48000.0, 0 }),
            "invalid block size load starts");
    require(waitForTerminalState(loader), "invalid block size load completes");
    result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::failed,
            "invalid block size fails qualification");
    require(result.errorMessage.find("No audio runtime configuration") != std::string::npos,
            "unset block size message");
}

void testRepeatedReplacementCyclesRemainBounded()
{
    auto log = std::make_shared<BackendLog>();
    auto behavior = std::make_shared<std::atomic<int>>(
        static_cast<int>(QualifiedTestBackend::ProcessBehavior::succeed));
    auto declaredRate = std::make_shared<std::atomic<int>>(-1);
    auto throwOnPrepare = std::make_shared<std::atomic<bool>>(false);

    rave::BackgroundModelLoader loader([log, behavior, declaredRate, throwOnPrepare] {
        auto backend = std::make_shared<QualifiedTestBackend>(
            log, declaredRate->load(), static_cast<QualifiedTestBackend::ProcessBehavior>(
                                            behavior->load()));
        backend->throwOnPrepare = throwOnPrepare->load();
        return backend;
    });

    const rave::ModelRuntimeConfiguration rate48k { 48000.0, 8 };
    const rave::ModelRuntimeConfiguration rate44k { 44100.0, 8 };

    // success → failed warm-up → success → rate mismatch → success at 44.1 kHz.
    require(loader.start("valid-model", rate48k), "cycle 1 starts");
    require(waitForTerminalState(loader), "cycle 1 completes");
    require(loader.takeResult().state == rave::BackgroundModelLoader::State::succeeded,
            "cycle 1 qualifies");

    behavior->store(static_cast<int>(QualifiedTestBackend::ProcessBehavior::fail));
    require(loader.start("valid-model", rate48k), "cycle 2 starts");
    require(waitForTerminalState(loader), "cycle 2 completes");
    require(loader.takeResult().state == rave::BackgroundModelLoader::State::failed,
            "cycle 2 fails warm-up");

    behavior->store(static_cast<int>(QualifiedTestBackend::ProcessBehavior::succeed));
    require(loader.start("valid-model", rate48k), "cycle 3 starts");
    require(waitForTerminalState(loader), "cycle 3 completes");
    require(loader.takeResult().state == rave::BackgroundModelLoader::State::succeeded,
            "cycle 3 qualifies again");

    declaredRate->store(44100);
    require(loader.start("valid-model", rate48k), "cycle 4 starts");
    require(waitForTerminalState(loader), "cycle 4 completes");
    require(loader.takeResult().state == rave::BackgroundModelLoader::State::failed,
            "cycle 4 rejects rate mismatch");

    require(loader.start("valid-model", rate44k), "cycle 5 starts");
    require(waitForTerminalState(loader), "cycle 5 completes");
    auto result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::succeeded,
            "cycle 5 qualifies at 44.1 kHz");
    require(log->preparedSampleRate == 44100.0 && log->preparedBlockSize == 8,
            "latest configuration reached the backend");
    require(loader.state() == rave::BackgroundModelLoader::State::idle,
            "loader idle after repeated replacement");
}

void testLatentDimensionBoundaryBeforeQualificationAllocation()
{
    auto log = std::make_shared<BackendLog>();
    QualifiedTestBackend backend(log);
    backend.latentCount = rave::maximumLatentDimensions;
    require(rave::qualifyModelBackend(backend, {48000.0, 8}).empty(),
            "exactly 4096 latent dimensions qualify");
    require(log->prepareCalls == 1 && log->processCalls == 2,
            "4096 boundary runs normal qualification");

    auto oversizedLog = std::make_shared<BackendLog>();
    QualifiedTestBackend oversized(oversizedLog);
    oversized.latentCount = rave::maximumLatentDimensions + 1;
    const auto error = rave::qualifyModelBackend(oversized, {48000.0, 8});
    require(error.find("4097") != std::string::npos
                && error.find("4096") != std::string::npos,
            "4097 latent dimensions fail with actionable boundary diagnostic");
    require(oversizedLog->prepareCalls == 0 && oversizedLog->processCalls == 0,
            "oversized model is rejected before worker vectors are populated");
}
} // namespace

int main()
{
    testSuccessfulLoadQualifiesAtIntendedConfiguration();
    testFailedLoadCanBeFollowedBySuccess();
    testSampleRateMismatchFailsQualification();
    testWarmUpFailuresProduceDiagnostics();
    testInvalidRuntimeConfigurationFailsQualification();
    testRepeatedReplacementCyclesRemainBounded();
    testLatentDimensionBoundaryBeforeQualificationAllocation();
    std::cout << "BackgroundModelLoader tests passed\n";
}
