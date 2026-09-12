#include "model/BackgroundModelLoader.h"

#include <exception>
#include <stdexcept>
#include <utility>

namespace rave
{
BackgroundModelLoader::BackgroundModelLoader(BackendFactory backendFactory)
    : factory(std::move(backendFactory))
{
    if (!factory)
        throw std::invalid_argument("Background model loader requires a backend factory");
}

BackgroundModelLoader::~BackgroundModelLoader()
{
    if (loaderThread.joinable())
        loaderThread.join();
}

bool BackgroundModelLoader::start(std::string modelPath)
{
    if (currentState.load(std::memory_order_acquire) == State::loading)
        return false;

    joinCompletedThread();
    {
        std::scoped_lock lock(resultMutex);
        loadedBackend.reset();
        loadError.clear();
    }
    currentState.store(State::loading, std::memory_order_release);

    loaderThread = std::thread([this, path = std::move(modelPath)] {
        ModelBackendPtr candidate;
        std::string error;
        State completedState = State::failed;

        try
        {
            candidate = factory();
            if (candidate == nullptr)
            {
                error = "Backend factory returned no backend";
            }
            else if (candidate->load(path, error))
            {
                completedState = State::succeeded;
            }
            else if (error.empty())
            {
                error = "Model backend rejected the model without an error message";
            }
        }
        catch (const std::exception& exception)
        {
            candidate.reset();
            error = exception.what();
        }
        catch (...)
        {
            candidate.reset();
            error = "Unknown exception while loading model";
        }

        {
            std::scoped_lock lock(resultMutex);
            loadedBackend = completedState == State::succeeded ? std::move(candidate) : nullptr;
            loadError = std::move(error);
        }
        currentState.store(completedState, std::memory_order_release);
    });

    return true;
}

BackgroundModelLoader::State BackgroundModelLoader::state() const noexcept
{
    return currentState.load(std::memory_order_acquire);
}

BackgroundModelLoader::Result BackgroundModelLoader::takeResult()
{
    const auto observedState = state();
    if (observedState == State::idle || observedState == State::loading)
        return { observedState, nullptr, {} };

    joinCompletedThread();
    std::scoped_lock lock(resultMutex);
    Result result { observedState, std::move(loadedBackend), std::move(loadError) };
    currentState.store(State::idle, std::memory_order_release);
    return result;
}

void BackgroundModelLoader::joinCompletedThread()
{
    if (loaderThread.joinable())
        loaderThread.join();
}
} // namespace rave
