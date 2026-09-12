#include "model/BackgroundModelLoader.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <span>
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

struct LoadObservation
{
    std::thread::id loadThread;
    std::atomic<int> loadCalls { 0 };
};

class TestBackend final : public rave::ModelBackend
{
public:
    explicit TestBackend(std::shared_ptr<LoadObservation> newObservation)
        : observation(std::move(newObservation))
    {
    }

    bool load(const std::string& path, std::string& errorMessage) override
    {
        observation->loadThread = std::this_thread::get_id();
        observation->loadCalls.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (path == "valid-model")
            return true;
        errorMessage = "invalid test model";
        return false;
    }

    void prepare(double, std::size_t) override {}
    void reset() noexcept override {}
    std::size_t latentDimensionCount() const noexcept override { return 0; }
    bool process(std::span<const float>, std::span<const float>, std::span<float>) override
    {
        return false;
    }

private:
    std::shared_ptr<LoadObservation> observation;
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

void testSuccessfulBackgroundLoad()
{
    const auto callerThread = std::this_thread::get_id();
    auto observation = std::make_shared<LoadObservation>();
    rave::BackgroundModelLoader loader([observation] {
        return std::make_shared<TestBackend>(observation);
    });

    require(loader.start("valid-model"), "first load starts");
    require(loader.state() == rave::BackgroundModelLoader::State::loading, "loading state reported");
    require(!loader.start("another-model"), "concurrent load rejected");
    require(waitForTerminalState(loader), "load reaches terminal state");

    auto result = loader.takeResult();
    require(result.state == rave::BackgroundModelLoader::State::succeeded, "success state returned");
    require(result.backend != nullptr, "loaded backend returned");
    require(result.errorMessage.empty(), "successful load has no error");
    require(observation->loadThread != callerThread, "load runs on background thread");
    require(observation->loadCalls.load() == 1, "only accepted load executed");
    require(loader.state() == rave::BackgroundModelLoader::State::idle, "result consumption resets state");
}

void testFailedLoadCanBeFollowedBySuccess()
{
    auto observation = std::make_shared<LoadObservation>();
    rave::BackgroundModelLoader loader([observation] {
        return std::make_shared<TestBackend>(observation);
    });

    require(loader.start("bad-model"), "failing load starts");
    require(waitForTerminalState(loader), "failing load completes");
    auto failed = loader.takeResult();
    require(failed.state == rave::BackgroundModelLoader::State::failed, "failure state returned");
    require(failed.backend == nullptr, "failed backend not returned");
    require(failed.errorMessage == "invalid test model", "backend error preserved");

    require(loader.start("valid-model"), "loader reusable after failure");
    require(waitForTerminalState(loader), "second load completes");
    auto succeeded = loader.takeResult();
    require(succeeded.state == rave::BackgroundModelLoader::State::succeeded, "second load succeeds");
    require(succeeded.backend != nullptr, "second backend returned");
}
} // namespace

int main()
{
    testSuccessfulBackgroundLoad();
    testFailedLoadCanBeFollowedBySuccess();
    std::cout << "BackgroundModelLoader tests passed\n";
}
