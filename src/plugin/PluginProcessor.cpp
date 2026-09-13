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

void RavePluginProcessor::publishModelLoadFailure(const std::string& errorMessage)
{
    // Capture the would-be final state before the seam to prove event storage
    // is independent of it. modelStatus() deliberately ignores this snapshot
    // and derives engine truth anew for every read.
    static_cast<void>(engine.lifecycleStatusSnapshot());
    if (failedLoadStatusInterleaveForTesting)
        failedLoadStatusInterleaveForTesting();

    const juce::ScopedLock lock(modelStateLock);
    currentModelStatus = "Model load failed: " + juce::String(errorMessage);
    currentStatusNeedsLifecycle = true;
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
    engine.setDryWet(dryWetParameter != nullptr ? dryWetParameter->get() : 0.0f);

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
        static_cast<void>(engine.setLatentControl(index, macroParameters[index]->get()));

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
    state.setAttribute("version", 1);
    state.setAttribute("dryWet", dryWetParameter != nullptr ? dryWetParameter->get() : 0.0f);
    for (std::size_t index = 0; index < macroCount; ++index)
    {
        state.setAttribute("macro" + juce::String(index + 1), macroParameters[index]->get());
        state.setAttribute("midiCc" + juce::String(index + 1),
                           midiControllers[index + 1].load(std::memory_order_relaxed));
    }
    state.setAttribute("midiCcDryWet", midiControllers[0].load(std::memory_order_relaxed));

    {
        const juce::ScopedLock lock(modelStateLock);
        if (activeModelFile != juce::File {})
            state.setAttribute("modelPath", activeModelFile.getFullPathName());
    }

    auto* const latents = state.createNewChildElement("Latents");
    const auto latentCount = engine.latentDimensionCount();
    latents->setAttribute("count", static_cast<int>(latentCount));
    for (std::size_t index = 0; index < latentCount; ++index)
        latents->setAttribute("v" + juce::String(index + 1), engine.latentControl(index));

    copyXmlToBinary(state, destinationData);
}

void RavePluginProcessor::setStateInformation(const void* const data, const int sizeInBytes)
{
    if (data == nullptr || dryWetParameter == nullptr)
        return;

    // Preserve compatibility with the initial float-only state format.
    if (sizeInBytes == static_cast<int>(sizeof(float)))
    {
        float value = 0.0f;
        std::memcpy(&value, data, sizeof(value));
        dryWetParameter->setValueNotifyingHost(
            dryWetParameter->convertTo0to1(juce::jlimit(0.0f, 1.0f, value)));
        return;
    }

    const auto state = getXmlFromBinary(data, sizeInBytes);
    if (state == nullptr || !state->hasTagName("RavePluginState"))
        return;

    const auto restoreParameter = [](juce::AudioParameterFloat& parameter, const float value) {
        parameter.setValueNotifyingHost(parameter.convertTo0to1(value));
    };
    restoreParameter(*dryWetParameter,
                     juce::jlimit(0.0f, 1.0f,
                                  static_cast<float>(state->getDoubleAttribute("dryWet", 0.0))));
    midiControllers[0].store(juce::jlimit(-1, 127,
                                          state->getIntAttribute("midiCcDryWet", -1)),
                             std::memory_order_relaxed);
    for (std::size_t index = 0; index < macroCount; ++index)
    {
        const auto suffix = juce::String(index + 1);
        restoreParameter(*macroParameters[index],
                         juce::jlimit(-4.0f, 4.0f,
                                      static_cast<float>(state->getDoubleAttribute("macro" + suffix, 0.0))));
        midiControllers[index + 1].store(
            juce::jlimit(-1, 127, state->getIntAttribute("midiCc" + suffix, -1)),
            std::memory_order_relaxed);
    }

    std::vector<float> restoredLatents;
    if (const auto* const latents = state->getChildByName("Latents"))
    {
        const auto count = std::max(0, latents->getIntAttribute("count", 0));
        restoredLatents.resize(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            restoredLatents[static_cast<std::size_t>(index)] = juce::jlimit(
                -4.0f,
                4.0f,
                static_cast<float>(latents->getDoubleAttribute(
                    "v" + juce::String(index + 1), 0.0)));
        }
    }

    const juce::File restoredModel(state->getStringAttribute("modelPath"));
    if (restoredModel.existsAsFile())
    {
        const juce::ScopedLock lock(modelStateLock);
        queuedRestoreModelFile = restoredModel;
        queuedRestoreLatents = std::move(restoredLatents);
        hasQueuedModelRestore.store(true, std::memory_order_release);
        currentModelStatus = "Restoring " + restoredModel.getFileName() + utf8("…");
        currentStatusNeedsLifecycle = false;
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
#if RAVE_HAS_LIBTORCH
    // Never qualify a candidate against invented defaults: defer until a real
    // host/device configuration exists.
    const auto configuration = engine.runtimeConfiguration();
    if (configuration.sampleRate <= 0.0 || configuration.maximumBlockSize == 0)
    {
        const juce::ScopedLock lock(modelStateLock);
        currentModelStatus = "Waiting for the host audio configuration before loading a model";
        currentStatusNeedsLifecycle = false;
        return false;
    }

    if (!modelFile.existsAsFile() || modelLoader == nullptr
        || !modelLoader->start(modelFile.getFullPathName().toStdString(), configuration))
        return false;

    const juce::ScopedLock lock(modelStateLock);
    pendingModelFile = modelFile;
    pendingLatentRestore.clear();
    currentModelStatus = "Loading " + modelFile.getFileName() + utf8("…");
    currentStatusNeedsLifecycle = false;
    return true;
#else
    static_cast<void>(modelFile);
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
    {
        const juce::ScopedLock lock(modelStateLock);
        loadedFile = pendingModelFile;
        restoredLatents = std::move(pendingLatentRestore);
        pendingModelFile = juce::File {};
    }

    if (result.state == rave::BackgroundModelLoader::State::failed)
    {
        // Qualification failed, so the previous active model (if any) was never
        // replaced and remains playable.
        publishModelLoadFailure(result.errorMessage);
        return true;
    }

    activateModel(std::move(result.backend), loadedFile, restoredLatents);
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
    refreshLifecycleStatus();
    static_cast<void>(finishModelLoadIfReady());

#if RAVE_HAS_LIBTORCH
    if (!hasQueuedModelRestore.load(std::memory_order_acquire) || isModelLoading()
        || modelLoader == nullptr)
        return;

    juce::File modelFile;
    std::vector<float> restoredLatents;
    {
        const juce::ScopedLock lock(modelStateLock);
        modelFile = queuedRestoreModelFile;
        restoredLatents = queuedRestoreLatents;
    }

    if (!modelFile.existsAsFile())
    {
        const juce::ScopedLock lock(modelStateLock);
        hasQueuedModelRestore.store(false, std::memory_order_release);
        currentModelStatus = "Saved model file is unavailable";
        currentStatusNeedsLifecycle = false;
        currentModelRevision.fetch_add(1, std::memory_order_release);
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
        pendingModelFile = modelFile;
        pendingLatentRestore = std::move(restoredLatents);
        queuedRestoreModelFile = juce::File {};
        queuedRestoreLatents.clear();
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
        const auto message = metadata.getMessage();
        if (!message.isController())
            continue;

        const auto controller = message.getControllerNumber();
        const auto learnedTarget = learningMidiTarget.exchange(-1, std::memory_order_acq_rel);
        if (learnedTarget >= 0 && learnedTarget < static_cast<int>(midiTargetCount))
            midiControllers[static_cast<std::size_t>(learnedTarget)].store(
                controller, std::memory_order_relaxed);

        const auto normalizedValue = static_cast<float>(message.getControllerValue()) / 127.0f;
        for (std::size_t target = 0; target < midiTargetCount; ++target)
        {
            if (midiControllers[target].load(std::memory_order_relaxed) != controller)
                continue;

            auto* const parameter = target == 0 ? dryWetParameter : macroParameters[target - 1];
            if (parameter != nullptr)
                parameter->setValueNotifyingHost(normalizedValue);
        }
    }
}

void RavePluginProcessor::activateModel(rave::ModelBackendPtr backend,
                                        const juce::File& modelFile,
                                        const std::vector<float>& restoredLatents)
{
    bool activated = false;
    std::string activationFailure;

    suspendProcessing(true);
    {
        const juce::ScopedLock callbackGuard(getCallbackLock());
        try
        {
            activated = engine.activateModelBackend(std::move(backend), &activationFailure);
            if (activated)
            {
                const auto restoreCount = std::min(restoredLatents.size(), engine.latentDimensionCount());
                for (std::size_t index = 0; index < restoreCount; ++index)
                    static_cast<void>(engine.setLatentControl(index, restoredLatents[index]));
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
        currentModelStatus = "Model activation failed: " + utf8(cause.c_str());
        currentStatusNeedsLifecycle = true;
    }
    currentModelRevision.fetch_add(1, std::memory_order_release);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new RavePluginProcessor();
}
