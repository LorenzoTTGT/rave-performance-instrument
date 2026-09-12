#pragma once

#include "model/ModelBackend.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

namespace rave
{
class BackgroundModelLoader
{
public:
    enum class State
    {
        idle,
        loading,
        succeeded,
        failed
    };

    struct Result
    {
        State state = State::idle;
        ModelBackendPtr backend;
        std::string errorMessage;
    };

    using BackendFactory = std::function<ModelBackendPtr()>;

    explicit BackgroundModelLoader(BackendFactory backendFactory);
    ~BackgroundModelLoader();

    BackgroundModelLoader(const BackgroundModelLoader&) = delete;
    BackgroundModelLoader& operator=(const BackgroundModelLoader&) = delete;

    // Loads and qualifies a candidate on the loader thread at the intended
    // runtime configuration. The candidate only reaches State::succeeded after
    // it loaded, prepared, reset, and warmed up successfully; any mismatch,
    // malformed metadata, bad shape, non-finite output, processing failure, or
    // thrown backend exception produces State::failed with a diagnostic and no
    // backend. Returns false if another load is already in progress.
    [[nodiscard]] bool start(std::string modelPath, const ModelRuntimeConfiguration& configuration);
    [[nodiscard]] State state() const noexcept;

    // Returns the latest terminal result and resets the loader to idle. Calling
    // this while idle/loading returns that state without consuming anything.
    [[nodiscard]] Result takeResult();

private:
    void joinCompletedThread();

    BackendFactory factory;
    std::thread loaderThread;
    std::atomic<State> currentState { State::idle };
    std::mutex resultMutex;
    ModelBackendPtr loadedBackend;
    std::string loadError;
};
} // namespace rave
