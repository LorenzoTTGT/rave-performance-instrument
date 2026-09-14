#include "plugin/PluginProcessor.h"

#include "plugin/PluginEditor.h"
#include "engine/LifecycleStatusText.h"

#if RAVE_HAS_LIBTORCH
#include "model/TorchScriptBackend.h"
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <cmath>

namespace
{
// juce::String(const char*) rejects non-ASCII literals, so every status string
// that contains typographic characters is built through an explicit UTF-8 pointer.
[[nodiscard]] juce::String utf8(const char* const text)
{
    return juce::String(juce::CharPointer_UTF8(text));
}
} // namespace

RavePluginProcessor::RavePluginProcessor(std::function<rave::ModelBackendPtr()> backendFactory)
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    auto parameter = std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "dryWet", 1 },
        "Dry / Wet",
        juce::NormalisableRange<float> { 0.0f, 1.0f, 0.01f },
        0.0f);
    dryWetParameter = parameter.get();
    addParameter(parameter.release());

    for (std::size_t index = 0; index < macroCount; ++index)
    {
        auto macro = std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID { "macro" + juce::String(index + 1), 1 },
            "Macro " + juce::String(index + 1) + " / Latent " + juce::String(index + 1),
            juce::NormalisableRange<float> { -4.0f, 4.0f, 0.01f },
            0.0f);
        macroParameters[index] = macro.get();
        addParameter(macro.release());
    }

    for (auto& controller : midiControllers)
        controller.store(-1, std::memory_order_relaxed);
    for (std::size_t target = 0; target < midiTargetCount; ++target)
    {
        midiLatestValues[target].store(0.0f, std::memory_order_relaxed);
        midiValueSequences[target].store(0, std::memory_order_relaxed);
        midiPublishedSequences[target].store(0, std::memory_order_relaxed);
    }

#if RAVE_HAS_LIBTORCH
    modelLoader = std::make_unique<rave::BackgroundModelLoader>(
        backendFactory != nullptr
            ? std::move(backendFactory)
            : [] { return std::make_shared<rave::TorchScriptBackend>(); });
#else
    static_cast<void>(backendFactory);
#endif
    startTimerHz(10);
}

RavePluginProcessor::~RavePluginProcessor()
{
    stopTimer();
    engine.release();
}

void RavePluginProcessor::prepareToPlay(const double sampleRate, const int samplesPerBlock)
{
    engine.prepare(sampleRate,
                   static_cast<std::size_t>(std::max(1, samplesPerBlock)),
                   getTotalNumOutputChannels());
    setLatencySamples(static_cast<int>(rave::RaveAudioEngine::transportLatencySamples));
    // Surface an incompatible reprepare or failing checked reset/start in the
    // visible status instead of leaving the previous active claim up.
    refreshLifecycleStatus();
    // A host may restore state before supplying its audio configuration.
    // Starting the retained newest request here avoids requiring a second
    // setStateInformation call.
    timerCallback();
}

void RavePluginProcessor::refreshLifecycleStatus()
{
    // Lock order for every status commit is modelStateLock, then the engine's
    // lifecycleMutex through lifecycleStatusSnapshot(). Activation releases
    // lifecycle ownership before taking modelStateLock, so this order cannot
    // form a cycle. The realtime process path takes neither lock.
    const juce::ScopedLock lock(modelStateLock);
    const auto snapshot = engine.lifecycleStatusSnapshot();
    if (snapshot.revision <= lastSeenLifecycleRevision)
        return;
    lastSeenLifecycleRevision = snapshot.revision;

    // Engine truth is rendered on demand by modelStatus(); refresh only
    // publishes that a newer lifecycle revision is observable.
    currentModelRevision.fetch_add(1, std::memory_order_release);
}

void RavePluginProcessor::publishModelLoadFailure(const std::string& errorMessage,
                                                  const bool relink)
{
    // Capture the would-be final state before the seam to prove event storage
    // is independent of it. modelStatus() deliberately ignores this snapshot
    // and derives engine truth anew for every read.
    static_cast<void>(engine.lifecycleStatusSnapshot());
    if (failedLoadStatusInterleaveForTesting)
        failedLoadStatusInterleaveForTesting();

    const juce::ScopedLock lock(modelStateLock);
    currentModelStatus = relink
        ? rave::suppressedReplacementFailureStatus("Model load failed", errorMessage, true)
        : "Model load failed: " + juce::String(errorMessage);
    currentStatusNeedsLifecycle = !relink;
    currentModelRevision.fetch_add(1, std::memory_order_release);
}

void RavePluginProcessor::releaseResources()
{
    engine.release();
    // Publish installed-but-not-running instead of leaving the active claim.
    refreshLifecycleStatus();
}

bool RavePluginProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto input = layouts.getMainInputChannelSet();
    const auto output = layouts.getMainOutputChannelSet();
    return input == output
        && (output == juce::AudioChannelSet::mono() || output == juce::AudioChannelSet::stereo());
}

void RavePluginProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                       juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    applyMidi(midiMessages);
    const auto realtimeParameterValue = [this](const std::size_t target,
                                                juce::AudioParameterFloat* const parameter) {
        const auto latestSequence = midiValueSequences[target].load(std::memory_order_acquire);
        const auto publishedSequence = midiPublishedSequences[target].load(std::memory_order_acquire);
        return latestSequence != publishedSequence
            ? parameter->convertFrom0to1(midiLatestValues[target].load(std::memory_order_relaxed))
            : parameter->get();
    };
    engine.setDryWet(dryWetParameter != nullptr
                         ? realtimeParameterValue(0, dryWetParameter)
                         : 0.0f);

    constexpr std::size_t maximumSupportedChannels = 2;
    const auto channelCount = static_cast<std::size_t>(buffer.getNumChannels());
    if (channelCount == 0 || channelCount > maximumSupportedChannels)
    {
        buffer.clear();
        return;
    }

    std::array<const float*, maximumSupportedChannels> inputs {};
    std::array<float*, maximumSupportedChannels> outputs {};
    for (std::size_t channel = 0; channel < channelCount; ++channel)
    {
        inputs[channel] = buffer.getReadPointer(static_cast<int>(channel));
        outputs[channel] = buffer.getWritePointer(static_cast<int>(channel));
    }

    const auto controlledLatents = std::min(macroCount, engine.latentDimensionCount());
    for (std::size_t index = 0; index < controlledLatents; ++index)
        static_cast<void>(engine.setLatentControl(
            index, realtimeParameterValue(index + 1, macroParameters[index])));

    engine.processAudio(inputs.data(),
                        static_cast<int>(channelCount),
                        outputs.data(),
                        static_cast<int>(channelCount),
                        buffer.getNumSamples());
}

juce::AudioProcessorEditor* RavePluginProcessor::createEditor()
{
    return new RavePluginEditor(*this);
}

void RavePluginProcessor::getStateInformation(juce::MemoryBlock& destinationData)
{
    juce::XmlElement state("RavePluginState");
    state.setAttribute("version", stateSchemaVersion);
    state.setAttribute("dryWet", dryWetParameter != nullptr ? dryWetParameter->get() : 0.0f);
    for (std::size_t index = 0; index < macroCount; ++index)
    {
        state.setAttribute("macro" + juce::String(index + 1), macroParameters[index]->get());
        state.setAttribute("midiCc" + juce::String(index + 1),
                           midiControllers[index + 1].load(std::memory_order_relaxed));
    }
    state.setAttribute("midiCcDryWet", midiControllers[0].load(std::memory_order_relaxed));

    bool savedRelinkRequired = false;
    juce::File savedModel;
    std::vector<float> savedRelinkLatents;
    {
        const juce::ScopedLock lock(modelStateLock);
        savedRelinkRequired = relinkRequired;
        if (savedRelinkRequired)
        {
            savedModel = requestedMissingModelFile;
            savedRelinkLatents = retainedMissingLatents;
        }
        else if (hasQueuedModelRestore.load(std::memory_order_acquire))
        {
            savedModel = queuedRestoreModelFile;
            savedRelinkLatents = queuedRestoreLatents;
        }
        else
        {
            savedModel = activeModelFile;
        }
    }
    if (savedModel != juce::File {})
        state.setAttribute("modelPath", savedModel.getFullPathName());
    state.setAttribute("relinkRequired", savedRelinkRequired);

    auto* const latents = state.createNewChildElement("Latents");
    const bool useSavedLatents = !savedRelinkLatents.empty();
    const auto latentCount = std::min(
        useSavedLatents ? savedRelinkLatents.size() : engine.latentDimensionCount(),
        maximumLatentCount);
    const auto dynamicCount = latentCount > macroCount ? latentCount - macroCount : 0;
    latents->setAttribute("count", static_cast<int>(dynamicCount));
    latents->setAttribute("firstIndex", static_cast<int>(macroCount));
    for (std::size_t index = macroCount; index < latentCount; ++index)
        latents->setAttribute("v" + juce::String(index), useSavedLatents ? savedRelinkLatents[index] : engine.latentControl(index));

    juce::MemoryBlock candidate;
    copyXmlToBinary(state, candidate);
    if (candidate.getSize() <= maximumStateBytes)
        destinationData = candidate;
}

void RavePluginProcessor::setStateInformation(const void* const data, const int sizeInBytes)
{
    if (data == nullptr || dryWetParameter == nullptr || sizeInBytes <= 0
        || static_cast<std::size_t>(sizeInBytes) > maximumStateBytes)
        return;

    // Preserve compatibility with the initial float-only state format.
    if (sizeInBytes == static_cast<int>(sizeof(float)))
    {
        float value = 0.0f;
        std::memcpy(&value, data, sizeof(value));
        if (std::isfinite(value))
            dryWetParameter->setValueNotifyingHost(
                dryWetParameter->convertTo0to1(juce::jlimit(0.0f, 1.0f, value)));
        return;
    }

    const auto state = getXmlFromBinary(data, sizeInBytes);
    if (state == nullptr || !state->hasTagName("RavePluginState"))
        return;
    const auto version = state->getIntAttribute("version", 1);
    if (version < 1 || version > stateSchemaVersion)
        return;
    const auto finiteAttribute = [](const juce::XmlElement& element, const juce::String& name,
                                    double fallback) { const auto value = element.getDoubleAttribute(name, fallback); return std::isfinite(value); };
    if (!finiteAttribute(*state, "dryWet", 0.0)) return;
    for (std::size_t i=0;i<macroCount;++i)
        if (!finiteAttribute(*state, "macro"+juce::String(i+1), 0.0)) return;

    const auto restoredDryWet = juce::jlimit(
        0.0f, 1.0f, static_cast<float>(state->getDoubleAttribute("dryWet", 0.0)));
    std::array<float, macroCount> restoredMacros {};
    std::array<int, midiTargetCount> restoredMidi {};
    restoredMidi[0] = juce::jlimit(-1, 127, state->getIntAttribute("midiCcDryWet", -1));
    for (std::size_t index = 0; index < macroCount; ++index)
    {
        const auto suffix = juce::String(index + 1);
        restoredMacros[index] = juce::jlimit(
            -4.0f, 4.0f,
            static_cast<float>(state->getDoubleAttribute("macro" + suffix, 0.0)));
        restoredMidi[index + 1] = juce::jlimit(
            -1, 127, state->getIntAttribute("midiCc" + suffix, -1));
    }

    std::vector<float> restoredLatents(restoredMacros.begin(), restoredMacros.end());
    if (const auto* const latents = state->getChildByName("Latents"))
    {
        const auto count = latents->getIntAttribute("count", 0);
        const auto first = version >= 2 ? latents->getIntAttribute("firstIndex", int(macroCount)) : 0;
        if (count < 0 || first < 0 || static_cast<std::size_t>(count) > maximumLatentCount
            || static_cast<std::size_t>(first) + static_cast<std::size_t>(count) > maximumLatentCount) return;
        restoredLatents.resize(std::max(restoredLatents.size(), static_cast<std::size_t>(first + count)), 0.0f);
        for (int index = 0; index < count; ++index)
        {
            const auto key = version >= 2 ? "v" + juce::String(first + index) : "v" + juce::String(index + 1);
            const auto value = latents->getDoubleAttribute(key, 0.0);
            if (!std::isfinite(value)) return;
            if (version == 1 && index < int(macroCount)) continue; // host macros are authoritative
            restoredLatents[static_cast<std::size_t>(first + index)] = juce::jlimit(-4.0f, 4.0f, static_cast<float>(value));
        }
    }

    const auto restoredModelPath = state->getStringAttribute("modelPath");
    if (restoredModelPath.length() > 4096)
        return;

    // Validation is now complete. Capture the exact callback generations that
    // this state commit supersedes; a callback arriving after this boundary
    // must remain pending and authoritative.
    std::array<std::uint64_t, midiTargetCount> capturedMidiSequences {};
    for (std::size_t target = 0; target < midiTargetCount; ++target)
        capturedMidiSequences[target]
            = midiValueSequences[target].load(std::memory_order_acquire);

    const auto restoreParameter = [](juce::AudioParameterFloat& parameter, const float value) {
        parameter.setValueNotifyingHost(parameter.convertTo0to1(value));
    };
    restoreParameter(*dryWetParameter, restoredDryWet);
    midiControllers[0].store(restoredMidi[0], std::memory_order_relaxed);
    for (std::size_t index = 0; index < macroCount; ++index)
    {
        restoreParameter(*macroParameters[index], restoredMacros[index]);
        midiControllers[index + 1].store(restoredMidi[index + 1], std::memory_order_relaxed);
    }

    if (stateRestoreMidiAcknowledgeInterleaveForTesting)
        stateRestoreMidiAcknowledgeInterleaveForTesting();
    for (std::size_t target = 0; target < midiTargetCount; ++target)
    {
        const auto captured = capturedMidiSequences[target];
        if (midiValueSequences[target].load(std::memory_order_acquire) == captured)
            midiPublishedSequences[target].store(captured, std::memory_order_release);
    }

    const juce::File restoredModel(restoredModelPath);
    requestGeneration.fetch_add(1, std::memory_order_acq_rel);
    if (restoredModel.existsAsFile())
    {
        {
            const juce::ScopedLock lock(modelStateLock);
            relinkRequired = false;
            requestedMissingModelFile = juce::File {};
            retainedMissingLatents.clear();
        }
        static_cast<void>(queueModelRequest(restoredModel, std::move(restoredLatents), false));
    }
    else
    {
        const bool isMissing = restoredModel != juce::File {};
        {
            const juce::ScopedLock lock(modelStateLock);
            queuedRestoreModelFile = juce::File {};
            queuedRestoreLatents.clear();
            queuedRelink = false;
            hasQueuedModelRestore.store(false, std::memory_order_release);
            requestedMissingModelFile = isMissing ? restoredModel : juce::File {};
            retainedMissingLatents = isMissing ? std::move(restoredLatents) : std::vector<float> {};
            relinkRequired = isMissing;
            activeModelFile = juce::File {};
            currentModelStatus = isMissing
                ? "Saved model is missing; relink required: " + restoredModel.getFileName()
                : "No model loaded";
            currentStatusNeedsLifecycle = false;
            currentModelRevision.fetch_add(1, std::memory_order_release);
        }
        engine.setWetSuppressed(true);
    }

    if (const auto* const messageManager = juce::MessageManager::getInstanceWithoutCreating();
        messageManager != nullptr && messageManager->isThisTheMessageThread())
        timerCallback();
}

juce::RangedAudioParameter& RavePluginProcessor::dryWetParameterReference() const
{
    jassert(dryWetParameter != nullptr);
    return *dryWetParameter;
}

juce::RangedAudioParameter& RavePluginProcessor::macroParameterReference(const std::size_t index) const
{
    jassert(index < macroCount && macroParameters[index] != nullptr);
    return *macroParameters[index];
}

void RavePluginProcessor::beginMidiLearn(const std::size_t targetIndex) noexcept
{
    if (targetIndex < midiTargetCount)
        learningMidiTarget.store(static_cast<int>(targetIndex), std::memory_order_release);
}

int RavePluginProcessor::midiControllerForTarget(const std::size_t targetIndex) const noexcept
{
    return targetIndex < midiTargetCount
        ? midiControllers[targetIndex].load(std::memory_order_relaxed)
        : -1;
}

bool RavePluginProcessor::isLearningMidiTarget(const std::size_t targetIndex) const noexcept
{
    return targetIndex < midiTargetCount
        && learningMidiTarget.load(std::memory_order_acquire) == static_cast<int>(targetIndex);
}

bool RavePluginProcessor::startModelLoad(const juce::File& modelFile)
{
    requestGeneration.fetch_add(1, std::memory_order_acq_rel);
    return queueModelRequest(modelFile, {}, false);
}

bool RavePluginProcessor::relinkMissingModel(const juce::File& modelFile)
{
    std::vector<float> values;
    { const juce::ScopedLock lock(modelStateLock); if (!relinkRequired) return false; values = retainedMissingLatents; }
    requestGeneration.fetch_add(1, std::memory_order_acq_rel);
    return queueModelRequest(modelFile, std::move(values), true);
}

bool RavePluginProcessor::isRelinkRequired() const noexcept
{ const juce::ScopedLock lock(modelStateLock); return relinkRequired; }
juce::String RavePluginProcessor::requestedModelPath() const
{ const juce::ScopedLock lock(modelStateLock); return requestedMissingModelFile.getFullPathName(); }

bool RavePluginProcessor::queueModelRequest(const juce::File& modelFile,
                                            std::vector<float> values,
                                            const bool relink)
{
#if RAVE_HAS_LIBTORCH
    const auto generation = requestGeneration.load(std::memory_order_acquire);
    // Retain the newest request until a real host/device configuration exists.
    const auto configuration = engine.runtimeConfiguration();
    if (configuration.sampleRate <= 0.0 || configuration.maximumBlockSize == 0)
    {
        const juce::ScopedLock lock(modelStateLock);
        queuedRestoreModelFile = modelFile;
        queuedRestoreLatents = std::move(values);
        queuedGeneration = generation;
        queuedRelink = relink;
        hasQueuedModelRestore.store(true, std::memory_order_release);
        currentModelStatus = "Waiting for the host audio configuration before loading a model";
        currentStatusNeedsLifecycle = false;
        return true;
    }

    if (!modelFile.existsAsFile() || modelLoader == nullptr)
        return false;
    if (modelLoader->state() == rave::BackgroundModelLoader::State::loading)
    {
        const juce::ScopedLock lock(modelStateLock);
        queuedRestoreModelFile=modelFile; queuedRestoreLatents=std::move(values); queuedGeneration=generation;
        queuedRelink = relink;
        hasQueuedModelRestore.store(true, std::memory_order_release);
        currentModelStatus="Queued " + modelFile.getFileName() + utf8("…"); currentStatusNeedsLifecycle=false;
        return true;
    }
    if (!modelLoader->start(modelFile.getFullPathName().toStdString(), configuration)) return false;

    const juce::ScopedLock lock(modelStateLock);
    pendingModelFile = modelFile;
    pendingLatentRestore = std::move(values);
    pendingGeneration = generation;
    pendingRelink = relink;
    currentModelStatus = "Loading " + modelFile.getFileName() + utf8("…");
    currentStatusNeedsLifecycle = false;
    return true;
#else
    static_cast<void>(modelFile); static_cast<void>(values); static_cast<void>(relink);
    const juce::ScopedLock lock(modelStateLock);
    currentModelStatus = "LibTorch backend unavailable";
    currentStatusNeedsLifecycle = false;
    return false;
#endif
}

bool RavePluginProcessor::finishModelLoadIfReady()
{
#if RAVE_HAS_LIBTORCH
    if (modelLoader == nullptr)
        return false;

    const auto state = modelLoader->state();
    if (state != rave::BackgroundModelLoader::State::succeeded
        && state != rave::BackgroundModelLoader::State::failed)
        return false;

    auto result = modelLoader->takeResult();
    juce::File loadedFile;
    std::vector<float> restoredLatents;
    std::uint64_t generation = 0;
    bool relink = false;
    {
        const juce::ScopedLock lock(modelStateLock);
        loadedFile = pendingModelFile;
        restoredLatents = std::move(pendingLatentRestore);
        pendingModelFile = juce::File {};
        generation = pendingGeneration;
        relink = pendingRelink;
        pendingRelink = false;
    }
    if (generation != requestGeneration.load(std::memory_order_acquire))
        return true;

    if (result.state == rave::BackgroundModelLoader::State::failed)
    {
        // Qualification failed, so the previous active model (if any) was never
        // replaced and remains playable.
        publishModelLoadFailure(result.errorMessage, relink);
        return true;
    }

    activateModel(std::move(result.backend), loadedFile, restoredLatents, generation, relink);
    return true;
#else
    return false;
#endif
}

juce::String RavePluginProcessor::modelStatus() const
{
    // Same lock order as refresh: modelStateLock, then engine lifecycleMutex.
    const juce::ScopedLock lock(modelStateLock);
    if (currentModelStatus.isNotEmpty() && !currentStatusNeedsLifecycle)
        return currentModelStatus;

    const auto snapshot = engine.lifecycleStatusSnapshot();
    const auto lifecycle = rave::lifecycleStatusText(snapshot.diagnostic,
                                                      snapshot.modelUsable,
                                                      snapshot.modelInstalled,
                                                      activeModelFile.getFileName(),
                                                      snapshot.latentDimensionCount,
                                                      "No model loaded");
    return currentModelStatus.isEmpty()
        ? lifecycle
        : currentModelStatus + utf8(" — current state: ") + lifecycle;
}

bool RavePluginProcessor::isModelLoading() const noexcept
{
#if RAVE_HAS_LIBTORCH
    return modelLoader != nullptr
        && modelLoader->state() == rave::BackgroundModelLoader::State::loading;
#else
    return false;
#endif
}

std::uint64_t RavePluginProcessor::modelRevision() const noexcept
{
    return currentModelRevision.load(std::memory_order_acquire);
}

std::size_t RavePluginProcessor::latentDimensionCount() const noexcept
{
    return engine.latentDimensionCount();
}

float RavePluginProcessor::latentControl(const std::size_t index) const noexcept
{
    return engine.latentControl(index);
}

bool RavePluginProcessor::setLatentControl(const std::size_t index, const float value) noexcept
{
    return engine.setLatentControl(index, value);
}

void RavePluginProcessor::timerCallback()
{
    publishPendingMidiParameterChanges();
    refreshLifecycleStatus();
    static_cast<void>(finishModelLoadIfReady());

#if RAVE_HAS_LIBTORCH
    if (!hasQueuedModelRestore.load(std::memory_order_acquire) || isModelLoading()
        || modelLoader == nullptr)
        return;

    juce::File modelFile;
    std::vector<float> restoredLatents;
    std::uint64_t generation = 0;
    bool relink = false;
    {
        const juce::ScopedLock lock(modelStateLock);
        modelFile = queuedRestoreModelFile;
        restoredLatents = queuedRestoreLatents;
        generation = queuedGeneration;
        relink = queuedRelink;
    }

    // Reconcile generation before interpreting disappearance. A stale queued
    // path must never create relink state over a newer request.
    if (generation != requestGeneration.load(std::memory_order_acquire))
    {
        const juce::ScopedLock lock(modelStateLock);
        if (queuedGeneration == generation)
        {
            hasQueuedModelRestore.store(false, std::memory_order_release);
            queuedRestoreModelFile = juce::File {};
            queuedRestoreLatents.clear();
            queuedRelink = false;
        }
        return;
    }

    if (!modelFile.existsAsFile())
    {
        {
            const juce::ScopedLock lock(modelStateLock);
            if (generation != requestGeneration.load(std::memory_order_acquire)
                || queuedGeneration != generation)
                return;
            hasQueuedModelRestore.store(false, std::memory_order_release);
            queuedRestoreModelFile = juce::File {};
            queuedRestoreLatents.clear();
            queuedRelink = false;
            requestedMissingModelFile = modelFile;
            retainedMissingLatents = std::move(restoredLatents);
            relinkRequired = true;
            activeModelFile = juce::File {};
            currentModelStatus = "Saved model is missing; relink required: "
                + modelFile.getFileName();
            currentStatusNeedsLifecycle = false;
            currentModelRevision.fetch_add(1, std::memory_order_release);
        }
        engine.setWetSuppressed(true);
        return;
    }

    // A restore must not qualify against invented defaults either; defer it
    // until a real host/device configuration exists.
    const auto configuration = engine.runtimeConfiguration();
    if (configuration.sampleRate <= 0.0 || configuration.maximumBlockSize == 0)
    {
        const juce::ScopedLock lock(modelStateLock);
        currentModelStatus = "Waiting for the host audio configuration before restoring "
            + modelFile.getFileName();
        currentStatusNeedsLifecycle = false;
        return;
    }

    if (modelLoader->start(modelFile.getFullPathName().toStdString(), configuration))
    {
        const juce::ScopedLock lock(modelStateLock);
        // Generation was checked immediately before start; activation checks
        // it again so a superseding request can never mutate visible state.
        pendingModelFile = modelFile;
        pendingLatentRestore = std::move(restoredLatents);
        pendingGeneration = generation;
        pendingRelink = relink;
        queuedRestoreModelFile = juce::File {};
        queuedRestoreLatents.clear();
        queuedRelink = false;
        hasQueuedModelRestore.store(false, std::memory_order_release);
        currentModelStatus = "Restoring " + modelFile.getFileName() + utf8("…");
        currentStatusNeedsLifecycle = false;
    }
#endif
}

void RavePluginProcessor::applyMidi(juce::MidiBuffer& midiMessages) noexcept
{
    for (const auto metadata : midiMessages)
    {
        // Inspect JUCE's non-owning event bytes directly. getMessage() may
        // allocate by constructing an owning MidiMessage for long SysEx data.
        applyRawMidiEvent(metadata.data, metadata.numBytes);
    }
}

void RavePluginProcessor::applyRawMidiEvent(const std::uint8_t* const data,
                                            const int byteCount) noexcept
{
    if (data == nullptr || byteCount != 3 || (data[0] & 0xf0u) != 0xb0u
        || data[1] >= 0x80u || data[2] >= 0x80u)
        return;

    const auto controller = static_cast<int>(data[1]);
    const auto learnedTarget = learningMidiTarget.exchange(-1, std::memory_order_acq_rel);
    if (learnedTarget >= 0 && learnedTarget < static_cast<int>(midiTargetCount))
        midiControllers[static_cast<std::size_t>(learnedTarget)].store(
            controller, std::memory_order_relaxed);

    const auto normalizedValue = static_cast<float>(data[2]) / 127.0f;
    for (std::size_t target = 0; target < midiTargetCount; ++target)
    {
        if (midiControllers[target].load(std::memory_order_relaxed) != controller)
            continue;
        midiLatestValues[target].store(normalizedValue, std::memory_order_relaxed);
        midiValueSequences[target].fetch_add(1, std::memory_order_release);
    }
}

void RavePluginProcessor::applyRawMidiEventForTesting(const std::uint8_t* const data,
                                                      const int byteCount) noexcept
{
    applyRawMidiEvent(data, byteCount);
}

void RavePluginProcessor::publishPendingMidiParameterChanges()
{
    for (std::size_t target = 0; target < midiTargetCount; ++target)
    {
        const auto sequence = midiValueSequences[target].load(std::memory_order_acquire);
        if (sequence == midiPublishedSequences[target].load(std::memory_order_relaxed))
            continue;
        const auto value = midiLatestValues[target].load(std::memory_order_relaxed);
        if (midiPublishInterleaveForTesting)
            midiPublishInterleaveForTesting();
        auto* const parameter = target == 0 ? dryWetParameter : macroParameters[target - 1];
        if (parameter != nullptr)
            parameter->setValueNotifyingHost(value);
        // A callback may have published a newer value while JUCE listeners
        // ran. Never mark that newer mailbox consumed by this stale notify.
        if (midiValueSequences[target].load(std::memory_order_acquire) == sequence)
            midiPublishedSequences[target].store(sequence, std::memory_order_release);
    }
}

void RavePluginProcessor::activateModel(rave::ModelBackendPtr backend,
                                        const juce::File& modelFile,
                                        const std::vector<float>& restoredLatents,
                                        const std::uint64_t generation,
                                        const bool relink)
{
    if (generation != requestGeneration.load(std::memory_order_acquire)) return;
    bool activated = false;
    std::string activationFailure;
    std::vector<float> prior(engine.latentDimensionCount());
    for (std::size_t i=0;i<prior.size();++i) prior[i]=engine.latentControl(i);

    suspendProcessing(true);
    {
        const juce::ScopedLock callbackGuard(getCallbackLock());
        try
        {
            activated = engine.activateModelBackend(std::move(backend), &activationFailure);
            if (activated)
            {
                const auto& values = restoredLatents.empty() ? prior : restoredLatents;
                const auto restoreCount = std::min(values.size(), engine.latentDimensionCount());
                for (std::size_t index = 0; index < restoreCount; ++index)
                    static_cast<void>(engine.setLatentControl(index, values[index]));
                for (std::size_t index=0; index<std::min(macroCount,engine.latentDimensionCount()); ++index)
                    static_cast<void>(engine.setLatentControl(index, macroParameters[index]->get()));
            }
        }
        catch (const std::exception& exception)
        {
            activationFailure = exception.what();
        }
        catch (...)
        {
            activationFailure = "unknown activation error";
        }
    }
    suspendProcessing(false);

    const juce::ScopedLock lock(modelStateLock);
    const auto lifecycleSnapshot = engine.lifecycleStatusSnapshot();
    // This explicit activation result already incorporates the engine's
    // current lifecycle outcome. Mark that revision observed so an immediate
    // timer refresh cannot erase candidate-failure context with the underlying
    // rollback diagnostic alone.
    lastSeenLifecycleRevision = lifecycleSnapshot.revision;
    if (activated)
    {
        activeModelFile = modelFile;
        relinkRequired = false; requestedMissingModelFile = juce::File {}; retainedMissingLatents.clear();
        engine.setWetSuppressed(false);
        currentModelStatus.clear();
        currentStatusNeedsLifecycle = false;
    }
    else
    {
        // A rollback that abandoned the previous model must also drop the
        // serialized identity: no model means zero latents and no dead path
        // for later state restores.
        if (!lifecycleSnapshot.modelInstalled)
            activeModelFile = juce::File {};

        // The engine diagnostic already states the truthful outcome (previous
        // model retained, no previous model, or rollback failure with bounded
        // dry pass-through) and may contain typographic characters, so build
        // it through the explicit UTF-8 pointer.
        const auto cause = rave::candidateFailureCause(
            activationFailure.empty() ? "candidate was not accepted" : activationFailure);
        currentModelStatus = relink
            ? rave::suppressedReplacementFailureStatus(
                  "Model activation failed", cause, true)
            : "Model activation failed: " + utf8(cause.c_str());
        currentStatusNeedsLifecycle = !relink;
    }
    currentModelRevision.fetch_add(1, std::memory_order_release);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RavePluginProcessor();
}
