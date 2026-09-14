#include "app/StandalonePresetModelCoordinator.h"

namespace rave
{
StandalonePresetModelCoordinator::StandalonePresetModelCoordinator(
    StandaloneSessionState& sessionState) noexcept
    : session(sessionState)
{
}

bool StandalonePresetModelCoordinator::applyPreset(
    const StandaloneSessionState::Snapshot& preset,
    const std::uint64_t generation)
{
    if (!session.restore(preset))
        return false;

    currentGeneration = generation;
    modelPath = preset.modelPath;
    state = modelPath.isEmpty() ? ModelState::noModel : ModelState::presetPending;
    return true;
}

void StandalonePresetModelCoordinator::beginOrdinaryRequest(const std::uint64_t generation) noexcept
{
    currentGeneration = generation;
    state = ModelState::engineAuthoritative;
}

void StandalonePresetModelCoordinator::beginReplacementRequest(
    const std::uint64_t generation) noexcept
{
    currentGeneration = generation;
    if (state != ModelState::noModel)
        state = ModelState::presetPending;
}

void StandalonePresetModelCoordinator::markPresetModelMissing(
    const std::uint64_t generation) noexcept
{
    if (isCurrent(generation) && !modelPath.isEmpty())
        state = ModelState::relinkRequired;
}

void StandalonePresetModelCoordinator::markReplacementFailed(
    const std::uint64_t generation) noexcept
{
    if (isCurrent(generation) && !modelPath.isEmpty())
        state = ModelState::relinkRequired;
}

bool StandalonePresetModelCoordinator::reconcileActivatedModel(
    const std::uint64_t generation,
    const std::size_t latentCount) noexcept
{
    if (!isCurrent(generation))
        return false;

    // StandaloneSessionState only resets slots which cannot belong to the
    // activated model. Values and MIDI mappings with matching indexes survive.
    session.setLatentCount(latentCount);
    return true;
}

bool StandalonePresetModelCoordinator::completeActivation(const std::uint64_t generation) noexcept
{
    if (!isCurrent(generation))
        return false;

    state = ModelState::engineAuthoritative;
    return true;
}

bool StandalonePresetModelCoordinator::presetControlsAreAuthoritative() const noexcept
{
    return state != ModelState::engineAuthoritative;
}

bool StandalonePresetModelCoordinator::wetMustBeSuppressed() const noexcept
{
    return state != ModelState::engineAuthoritative;
}

bool StandalonePresetModelCoordinator::relinkRequired() const noexcept
{
    return state == ModelState::relinkRequired;
}

bool StandalonePresetModelCoordinator::isCurrent(const std::uint64_t generation) const noexcept
{
    return currentGeneration == generation;
}

const juce::String& StandalonePresetModelCoordinator::savedModelPath() const noexcept
{
    return modelPath;
}
}
