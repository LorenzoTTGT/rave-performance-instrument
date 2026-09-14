#include "app/StandalonePresetModelCoordinator.h"
#include "engine/LifecycleStatusText.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

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

rave::StandaloneSessionState::Snapshot twelveLatentPreset(const juce::String& modelPath)
{
    rave::StandaloneSessionState::Snapshot preset;
    preset.modelPath = modelPath;
    preset.midiInputId = "stable-midi-device";
    preset.audioDeviceType = "CoreAudio";
    preset.audioInputId = "input-device";
    preset.audioOutputId = "output-device";
    preset.dryWet = 0.6f;
    preset.latents.resize(12);
    preset.midiControllers.resize(13);
    preset.midiControllers[0] = 7;
    for (std::size_t index = 0; index < preset.latents.size(); ++index)
    {
        preset.latents[index] = -3.5f + static_cast<float>(index) * 0.5f;
        preset.midiControllers[index + 1] = static_cast<int>(20 + index);
    }
    return preset;
}

void requireAllPresetControls(const rave::StandaloneSessionState& state,
                              const char* const phase)
{
    const auto snapshot = state.snapshot();
    require(snapshot.latents.size() == 12, phase);
    require(snapshot.midiControllers.size() == 13, phase);
    require(snapshot.midiInputId == "stable-midi-device", phase);
    require(snapshot.audioInputId == "input-device", phase);
    require(snapshot.audioOutputId == "output-device", phase);
    require(snapshot.midiControllers[0] == 7, phase);
    for (std::size_t index = 0; index < snapshot.latents.size(); ++index)
    {
        require(std::abs(snapshot.latents[index] - (-3.5f + static_cast<float>(index) * 0.5f))
                    < 0.001f,
                phase);
        require(snapshot.midiControllers[index + 1] == static_cast<int>(20 + index), phase);
    }
}

void testFreshMissingPresetRetainsControlsUntilRelinkActivation()
{
    rave::StandaloneSessionState state; // Fresh standalone: the engine would report zero latents.
    rave::StandalonePresetModelCoordinator coordinator(state);
    constexpr std::uint64_t presetGeneration = 10;

    require(coordinator.applyPreset(twelveLatentPreset("/missing/model-z12.ts"), presetGeneration),
            "apply twelve-latent preset");
    require(coordinator.presetControlsAreAuthoritative(),
            "fresh zero-latent engine cannot authoritatively resize restored preset");
    require(coordinator.wetMustBeSuppressed(), "preset replacement suppresses old wet path");
    requireAllPresetControls(state, "controls survive fresh engine timer synchronization");

    // A pending load still leaves restored controls authoritative.
    require(coordinator.beginModelRequest(presetGeneration, true),
            "explicit preset replacement is classified as replacement");
    require(coordinator.presetControlsAreAuthoritative(), "pending replacement retains preset shape");
    requireAllPresetControls(state, "controls survive pending model load");

    coordinator.markPresetModelMissing(presetGeneration);
    require(coordinator.relinkRequired(), "missing preset model enables relink");
    require(coordinator.savedModelPath() == "/missing/model-z12.ts",
            "missing preset identity remains exact relink target");
    require(coordinator.wetMustBeSuppressed(), "missing model remains aligned dry");
    requireAllPresetControls(state, "controls and mappings survive missing/relink state");

    require(coordinator.beginModelRequest(presetGeneration, true),
            "relink request remains a replacement");
    coordinator.markReplacementFailed(presetGeneration);
    require(coordinator.relinkRequired() && coordinator.wetMustBeSuppressed(),
            "failed relink cannot recover wet output");
    requireAllPresetControls(state, "failed relink preserves all preset controls");

    require(coordinator.reconcileActivatedModel(presetGeneration, 12),
            "activation reconciles retained shape to model dimensions");
    requireAllPresetControls(state, "all values and mappings survive twelve-latent activation");
    require(coordinator.completeActivation(presetGeneration), "complete successful activation");
    require(!coordinator.presetControlsAreAuthoritative() && !coordinator.wetMustBeSuppressed(),
            "engine authority and wet recovery occur only after control installation");
}

void testActivationReconcilesByIndex()
{
    rave::StandaloneSessionState state;
    rave::StandalonePresetModelCoordinator coordinator(state);
    require(coordinator.applyPreset(twelveLatentPreset("model-z8.ts"), 30),
            "smaller activated model preset applies");
    require(coordinator.reconcileActivatedModel(30, 8),
            "activation accepts current smaller model");
    const auto snapshot = state.snapshot();
    require(snapshot.latents.size() == 8 && snapshot.midiControllers.size() == 9,
            "activation trims only dimensions unavailable in model");
    for (std::size_t index = 0; index < snapshot.latents.size(); ++index)
    {
        require(std::abs(snapshot.latents[index] - (-3.5f + static_cast<float>(index) * 0.5f))
                    < 0.001f,
                "activation retains latent value by index");
        require(snapshot.midiControllers[index + 1] == static_cast<int>(20 + index),
                "activation retains MIDI mapping by index");
    }
}

void testOrdinaryLoadClassifiesSuppressedPresetStates()
{
    rave::StandaloneSessionState state;
    rave::StandalonePresetModelCoordinator coordinator(state);

    auto empty = twelveLatentPreset({});
    require(coordinator.applyPreset(empty, 40), "empty preset applies before ordinary load");
    require(coordinator.beginModelRequest(41, false),
            "ordinary load after empty preset is authoritative replacement");
    require(coordinator.wetMustBeSuppressed() && coordinator.presetControlsAreAuthoritative(),
            "empty preset remains protected while ordinary load is pending");
    requireAllPresetControls(state, "ordinary pending load retains empty preset controls");
    coordinator.markReplacementFailed(41);
    require(coordinator.wetMustBeSuppressed() && !coordinator.relinkRequired()
                && coordinator.savedModelPath().isEmpty(),
            "failed ordinary load keeps empty preset aligned dry without relink");

    require(coordinator.applyPreset(twelveLatentPreset("/missing/model-z12.ts"), 50),
            "missing preset applies before ordinary load");
    coordinator.markPresetModelMissing(50);
    require(coordinator.beginModelRequest(51, false),
            "ordinary load after missing preset is authoritative replacement");
    require(coordinator.wetMustBeSuppressed() && coordinator.presetControlsAreAuthoritative(),
            "missing preset remains protected while ordinary load is pending");
    coordinator.markReplacementFailed(51);
    require(coordinator.wetMustBeSuppressed() && coordinator.relinkRequired()
                && coordinator.savedModelPath() == "/missing/model-z12.ts",
            "failed ordinary load restores missing preset relink identity");

    require(coordinator.reconcileActivatedModel(51, 12),
            "ordinary replacement success reconciles controls before wet recovery");
    require(coordinator.completeActivation(51), "ordinary replacement completes activation");
    require(!coordinator.wetMustBeSuppressed(),
            "ordinary replacement success permits wet recovery after controls install");

    rave::StandaloneSessionState activeState;
    rave::StandalonePresetModelCoordinator activeCoordinator(activeState);
    require(!activeCoordinator.beginModelRequest(60, false),
            "ordinary active-model candidate load is not a replacement");
    require(!activeCoordinator.wetMustBeSuppressed()
                && !activeCoordinator.presetControlsAreAuthoritative(),
            "ordinary active-model candidate keeps RAVE-02 failure behavior");
}

void testCoordinatorOwnedStatusSurvivesDeviceLifecycle()
{
    rave::StandaloneSessionState state;
    rave::StandalonePresetModelCoordinator coordinator(state);
    rave::LifecycleStatusRevisionGate gate;

    auto pending = twelveLatentPreset("/missing/preset.ts");
    require(coordinator.applyPreset(pending, 70), "pending preset applies");
    require(!coordinator.shouldPublishLifecycleRevision(10, gate),
            "pending preset status survives old-engine release");
    require(!coordinator.shouldPublishLifecycleRevision(11, gate),
            "pending preset status survives old-engine reprepare");
    coordinator.markPresetModelMissing(70);
    require(!coordinator.shouldPublishLifecycleRevision(12, gate),
            "missing/relink status survives device lifecycle revision");

    require(coordinator.reconcileActivatedModel(70, 12)
                && coordinator.completeActivation(70),
            "coordinator returns authority after activation");
    require(!coordinator.shouldPublishLifecycleRevision(12, gate),
            "already-observed old-engine revision stays suppressed");
    require(coordinator.shouldPublishLifecycleRevision(13, gate),
            "genuinely newer lifecycle revision publishes after authority ends");

    auto noModel = twelveLatentPreset({});
    require(coordinator.applyPreset(noModel, 71), "explicit no-model preset applies");
    require(!coordinator.shouldPublishLifecycleRevision(14, gate),
            "explicit no-model status cannot be replaced by old active engine");
}

void testActivationFailureStatusRevisionInterleaving()
{
    rave::LifecycleStatusRevisionGate gate;
    require(gate.shouldPublish(10), "initial lifecycle event publishes");
    gate.markObserved(11); // explicit candidate-failure event rendered revision 11
    require(!gate.shouldPublish(11),
            "immediate lifecycle refresh preserves actionable activation failure");
    require(gate.shouldPublish(12),
            "genuinely newer lifecycle event supersedes activation failure");
    require(!gate.shouldPublish(11), "stale lifecycle event cannot overwrite newer status");
}

void testEmptyAndStalePresetTransitions()
{
    rave::StandaloneSessionState state;
    rave::StandalonePresetModelCoordinator coordinator(state);
    require(coordinator.applyPreset(twelveLatentPreset("existing.ts"), 20),
            "existing model preset applies");
    require(coordinator.wetMustBeSuppressed(), "existing preset identity suppresses old model wet");

    auto empty = twelveLatentPreset({});
    require(coordinator.applyPreset(empty, 21), "empty-model preset applies");
    require(coordinator.presetControlsAreAuthoritative() && coordinator.wetMustBeSuppressed(),
            "empty-model preset explicitly selects aligned dry no-model state");
    require(!coordinator.relinkRequired() && coordinator.savedModelPath().isEmpty(),
            "empty-model preset clears relink and saved model identity");
    requireAllPresetControls(state, "empty-model preset keeps controls and device identities");

    require(!coordinator.reconcileActivatedModel(20, 12),
            "stale successful result cannot overwrite newer empty preset");
    coordinator.markPresetModelMissing(20);
    require(!coordinator.relinkRequired(), "stale failure cannot enable relink for empty preset");
}
}

int main()
{
    testFreshMissingPresetRetainsControlsUntilRelinkActivation();
    testActivationReconcilesByIndex();
    testOrdinaryLoadClassifiesSuppressedPresetStates();
    testCoordinatorOwnedStatusSurvivesDeviceLifecycle();
    testActivationFailureStatusRevisionInterleaving();
    testEmptyAndStalePresetTransitions();
    std::cout << "Standalone preset coordinator tests passed\n";
}
