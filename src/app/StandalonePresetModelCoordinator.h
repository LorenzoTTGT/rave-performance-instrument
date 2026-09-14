#pragma once

#include "engine/LifecycleStatusText.h"
#include "state/StandaloneSessionState.h"

#include <cstddef>
#include <cstdint>

namespace rave
{
// Keeps preset controls authoritative until the preset's model identity has been
// resolved. This prevents an old (or empty) engine from resizing a newly restored
// standalone session while loading is deferred, pending, or relink-required.
class StandalonePresetModelCoordinator final
{
public:
    enum class ModelState
    {
        engineAuthoritative,
        presetPending,
        relinkRequired,
        noModel
    };

    explicit StandalonePresetModelCoordinator(StandaloneSessionState& session) noexcept;

    [[nodiscard]] bool applyPreset(const StandaloneSessionState::Snapshot& preset,
                                   std::uint64_t generation);
    // Returns whether the request replaces a preset-owned/no-model state and
    // must keep wet output suppressed until activation succeeds. This is also
    // the classification used by MainComponent for ordinary Load Model clicks.
    [[nodiscard]] bool beginModelRequest(std::uint64_t generation,
                                         bool explicitlyReplacesIdentity) noexcept;
    void markPresetModelMissing(std::uint64_t generation) noexcept;
    void markReplacementFailed(std::uint64_t generation) noexcept;
    [[nodiscard]] bool reconcileActivatedModel(std::uint64_t generation,
                                               std::size_t latentCount) noexcept;
    [[nodiscard]] bool completeActivation(std::uint64_t generation) noexcept;

    [[nodiscard]] bool presetControlsAreAuthoritative() const noexcept;
    [[nodiscard]] bool wetMustBeSuppressed() const noexcept;
    [[nodiscard]] bool relinkRequired() const noexcept;
    [[nodiscard]] bool isCurrent(std::uint64_t generation) const noexcept;
    [[nodiscard]] const juce::String& savedModelPath() const noexcept;
    [[nodiscard]] bool shouldPublishLifecycleRevision(
        std::uint64_t revision, LifecycleStatusRevisionGate& gate) const noexcept;

private:
    StandaloneSessionState& session;
    ModelState state = ModelState::engineAuthoritative;
    juce::String modelPath;
    std::uint64_t currentGeneration = 0;
};
}
