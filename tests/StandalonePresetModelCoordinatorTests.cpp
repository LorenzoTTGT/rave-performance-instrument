#include "app/StandalonePresetModelCoordinator.h"
#include "engine/LifecycleStatusText.h"
#include "engine/RaveAudioEngine.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

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

class ThreeLatentBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 3; }
    bool process(std::span<const float>, std::span<const float>,
                 std::span<float>) override { return true; }
};

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

void testStandaloneLoaderFailureStatusOutcomes()
{
    const rave::RaveAudioEngine::LifecycleStatusSnapshot usable {
        20, true, true, {}, 12
    };
    auto text = rave::standaloneModelLoadFailureStatus(
        "qualification rejected fixture", false, false, usable);
    require(text.contains("qualification rejected fixture")
                && text.containsIgnoreCase("previous model remains usable")
                && text.contains("12 latent dimensions"),
            "ordinary candidate failure reports concrete error and usable previous model");

    const rave::RaveAudioEngine::LifecycleStatusSnapshot unavailable {
        21, false, false, {}, 0
    };
    text = rave::standaloneModelLoadFailureStatus(
        "malformed model", false, false, unavailable);
    require(text.contains("malformed model")
                && text.containsIgnoreCase("bounded dry pass-through")
                && !text.containsIgnoreCase("active"),
            "ordinary failure without usable model reports truthful dry outcome");

    rave::StandaloneSessionState state;
    rave::StandalonePresetModelCoordinator coordinator(state);
    require(coordinator.applyPreset(twelveLatentPreset("/missing/model.ts"), 30),
            "replacement preset applies");
    require(coordinator.beginModelRequest(30, true), "replacement request classified");
    coordinator.markReplacementFailed(30);
    text = rave::standaloneModelLoadFailureStatus(
        "decode failed", true, coordinator.relinkRequired(), usable);
    require(text.contains("decode failed") && text.containsIgnoreCase("aligned dry")
                && text.containsIgnoreCase("relink required")
                && !text.containsIgnoreCase("active"),
            "replacement failure ignores old active engine and reports suppression/relink");
    text = rave::standaloneModelFailureStatus(
        "Model activation failed", "candidate reset refused", true,
        coordinator.relinkRequired(), usable);
    require(text.contains("candidate reset refused") && text.containsIgnoreCase("aligned dry")
                && text.containsIgnoreCase("relink required")
                && !text.containsIgnoreCase("previous model retained")
                && !text.containsIgnoreCase("active"),
            "replacement activation failure reports audible suppression, not engine rollback");
    text = rave::standaloneModelFailureStatus(
        "Model activation failed", "candidate reset refused", false, false, usable);
    require(text.contains("candidate reset refused")
                && text.containsIgnoreCase("previous model remains usable"),
            "ordinary activation failure retains truthful prior-model context");

    rave::LifecycleStatusRevisionGate gate;
    gate.markObserved(usable.revision);
    require(!gate.shouldPublish(usable.revision),
            "immediate lifecycle refresh preserves ordinary failure event");
    require(gate.shouldPublish(usable.revision + 1),
            "genuinely newer lifecycle event may supersede ordinary failure");
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

void testCallbackReattachmentGuard()
{
    int reattachments = 0;
    {
        rave::ScopedCallbackReattachment guard([&] { ++reattachments; });
    }
    require(reattachments == 1, "early exit reattaches callback exactly once");
    {
        rave::ScopedCallbackReattachment guard([&] { ++reattachments; });
        guard.reattach();
        guard.reattach();
    }
    require(reattachments == 2,
            "successful explicit reattachment and destructor remain exactly once");
    try
    {
        rave::ScopedCallbackReattachment guard([&] { ++reattachments; });
        throw std::runtime_error("activation commit seam");
    }
    catch (const std::runtime_error&)
    {
    }
    require(reattachments == 3, "exception exit reattaches callback exactly once");
}

void testEmptyPresetClearsInstalledBackendWhileDetached()
{
    rave::RaveAudioEngine engine;
    engine.setModelBackend(std::make_shared<ThreeLatentBackend>());
    require(engine.hasModelBackend() && engine.latentDimensionCount() == 3,
            "active standalone fixture exposes installed backend and latents");

    int detachments = 0;
    int reattachments = 0;
    rave::clearStandaloneModelBackend(
        engine,
        [&] { ++detachments; },
        [&] {
            ++reattachments;
            require(detachments == 1, "empty preset clears only after callback detaches");
            require(!engine.hasModelBackend() && engine.latentDimensionCount() == 0,
                    "empty preset reattaches only after backend and latents clear");
        });
    require(detachments == 1 && reattachments == 1,
            "empty preset detaches and reattaches callback exactly once");
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
    testStandaloneLoaderFailureStatusOutcomes();
    testCoordinatorOwnedStatusSurvivesDeviceLifecycle();
    testActivationFailureStatusRevisionInterleaving();
    testCallbackReattachmentGuard();
    testEmptyPresetClearsInstalledBackendWhileDetached();
    testEmptyAndStalePresetTransitions();
    std::cout << "Standalone preset coordinator tests passed\n";
}
