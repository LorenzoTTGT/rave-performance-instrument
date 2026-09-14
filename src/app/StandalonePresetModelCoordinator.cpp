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

bool StandalonePresetModelCoordinator::beginModelRequest(
    const std::uint64_t generation,
    const bool explicitlyReplacesIdentity) noexcept
{
    // An ordinary user load after an empty or missing preset is still an
    // authoritative replacement: a prior model must remain muted and the
    // preset-shaped controls must not be resized by the old engine. In the
    // normal active state it remains a RAVE-02 candidate load.
    const auto replacesSuppressedState = wetMustBeSuppressed();
    const auto replacement = explicitlyReplacesIdentity || replacesSuppressedState;
    currentGeneration = generation;
    if (replacement && state != ModelState::noModel)
        state = ModelState::presetPending;
    else if (!replacement)
        state = ModelState::engineAuthoritative;
    return replacement;
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
