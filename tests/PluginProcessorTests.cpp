#include "plugin/PluginProcessor.h"

#include "JuceTestAssertions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <thread>

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();

namespace allocationAudit
{
std::atomic<bool> enabled { false };
std::atomic<std::size_t> count { 0 };
}

void* operator new(const std::size_t size)
{
    if (allocationAudit::enabled.load(std::memory_order_relaxed))
        allocationAudit::count.fetch_add(1, std::memory_order_relaxed);
    if (auto* memory = std::malloc(size)) return memory;
    throw std::bad_alloc();
}
void* operator new[](const std::size_t size) { return ::operator new(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

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

bool waitForModelSettled(RavePluginProcessor& processor)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (processor.isModelLoading() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return !processor.isModelLoading();
}

bool loadModel(RavePluginProcessor& processor, const juce::File& modelFile)
{
    if (!processor.startModelLoad(modelFile))
        return false;
    return waitForModelSettled(processor) && processor.finishModelLoadIfReady();
}

void requireFiniteOutput(juce::AudioBuffer<float>& buffer)
{
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
    {
        const auto* samples = buffer.getReadPointer(channel);
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            require(std::isfinite(samples[sample]), "processed output stays finite");
    }
}

juce::RangedAudioParameter* findParameter(juce::AudioProcessor& processor,
                                          const juce::String& parameterId)
{
    for (auto* parameter : processor.getParameters())
    {
        auto* ranged = dynamic_cast<juce::RangedAudioParameter*>(parameter);
        if (ranged != nullptr && ranged->getParameterID() == parameterId)
            return ranged;
    }
    return nullptr;
}

class ParameterListenerCounter final : public juce::AudioProcessorParameter::Listener
{
public:
    void parameterValueChanged(int, float) override { ++valueChanges; }
    void parameterGestureChanged(int, bool) override {}
    std::atomic<int> valueChanges { 0 };
};

juce::MemoryBlock xmlState(juce::XmlElement& xml)
{
    juce::MemoryBlock result;
    juce::AudioProcessor::copyXmlToBinary(xml, result);
    return result;
}

void testStateSchemaMigrationsBoundsAndMacroAuthority()
{
    RavePluginProcessor processor;
    float legacy = 0.42f;
    processor.setStateInformation(&legacy, sizeof(legacy));
    require(std::abs(processor.dryWetParameterReference().convertFrom0to1(
        processor.dryWetParameterReference().getValue()) - legacy) < 0.001f, "float state migrates");

    juce::XmlElement v1("RavePluginState"); v1.setAttribute("version",1); v1.setAttribute("dryWet",.6);
    v1.setAttribute("macro1",2.0); auto* old=v1.createNewChildElement("Latents"); old->setAttribute("count",12);
    for(int i=0;i<12;++i) old->setAttribute("v"+juce::String(i+1),-3.0+i*.25);
    auto block=xmlState(v1); processor.setStateInformation(block.getData(),int(block.getSize()));
    require(std::abs(processor.macroParameterReference(0).convertFrom0to1(processor.macroParameterReference(0).getValue())-2.f)<.001f,
            "v1 latent payload cannot override macro");

    juce::MemoryBlock saved; processor.getStateInformation(saved);
    auto parsed=juce::AudioProcessor::getXmlFromBinary(saved.getData(),int(saved.getSize()));
    require(parsed && parsed->getIntAttribute("version")==2,"writes schema v2");
    auto* dynamic=parsed->getChildByName("Latents");
    require(dynamic && dynamic->getIntAttribute("firstIndex")==8,"v2 dynamic latents start after macros");

    const auto before=processor.dryWetParameterReference().getValue();
    juce::XmlElement unsupported("RavePluginState"); unsupported.setAttribute("version",99);unsupported.setAttribute("dryWet",.1);
    block=xmlState(unsupported);processor.setStateInformation(block.getData(),int(block.getSize()));
    require(std::abs(processor.dryWetParameterReference().getValue() - before) < 0.000001f,
            "unsupported version fails closed");
    juce::XmlElement nonfinite("RavePluginState");nonfinite.setAttribute("version",2);nonfinite.setAttribute("dryWet","nan");
    block=xmlState(nonfinite);processor.setStateInformation(block.getData(),int(block.getSize()));
    require(std::abs(processor.dryWetParameterReference().getValue() - before) < 0.000001f,
            "non-finite state fails closed");
    juce::XmlElement badLatent("RavePluginState"); badLatent.setAttribute("version",2); badLatent.setAttribute("dryWet",.1);
    auto* badLatents=badLatent.createNewChildElement("Latents"); badLatents->setAttribute("count",1); badLatents->setAttribute("firstIndex",8); badLatents->setAttribute("v8","nan");
    block=xmlState(badLatent);processor.setStateInformation(block.getData(),int(block.getSize()));
    require(std::abs(processor.dryWetParameterReference().getValue() - before) < 0.000001f,
            "invalid latent fails closed transactionally");
    juce::XmlElement longPath("RavePluginState"); longPath.setAttribute("version",2); longPath.setAttribute("dryWet",.1); longPath.setAttribute("modelPath",juce::String::repeatedString("x",4097));
    block=xmlState(longPath);processor.setStateInformation(block.getData(),int(block.getSize()));
    require(std::abs(processor.dryWetParameterReference().getValue() - before) < 0.000001f,
            "overlong model path fails closed transactionally");
    std::vector<char> oversized(RavePluginProcessor::maximumStateBytes+1);
    processor.setStateInformation(oversized.data(),int(oversized.size()));
    require(std::abs(processor.dryWetParameterReference().getValue() - before) < 0.000001f,
            "oversized state fails closed");

    juce::XmlElement missing("RavePluginState");missing.setAttribute("version",2);missing.setAttribute("modelPath","/definitely/missing/rave-model.ts");
    missing.setAttribute("dryWet",.75);missing.setAttribute("macro1",1.5);auto* lat=missing.createNewChildElement("Latents");lat->setAttribute("count",4);lat->setAttribute("firstIndex",8);
    for(int i=8;i<12;++i)lat->setAttribute("v"+juce::String(i),float(i)/10.f);
    block=xmlState(missing);processor.setStateInformation(block.getData(),int(block.getSize()));
    require(processor.isRelinkRequired()&&processor.requestedModelPath().contains("rave-model.ts"),"missing identity retained for relink");
    require(processor.modelStatus().containsIgnoreCase("relink required"),"missing model status visible");
    require(std::abs(processor.dryWetParameterReference().convertFrom0to1(processor.dryWetParameterReference().getValue())-.75f)<.001f,"missing restore retains dry/wet");
    require(std::abs(processor.macroParameterReference(0).convertFrom0to1(processor.macroParameterReference(0).getValue())-1.5f)<.001f,"missing restore retains macros");

    juce::XmlElement empty("RavePluginState");
    empty.setAttribute("version", 2);
    empty.setAttribute("dryWet", .75);
    empty.setAttribute("macro1", 1.5);
    block = xmlState(empty);
    processor.setStateInformation(block.getData(), int(block.getSize()));
    require(!processor.isRelinkRequired() && processor.requestedModelPath().isEmpty(),
            "empty restore clears missing and relink identity");
    juce::MemoryBlock emptySaved;
    processor.getStateInformation(emptySaved);
    parsed = juce::AudioProcessor::getXmlFromBinary(emptySaved.getData(), int(emptySaved.getSize()));
    require(parsed && parsed->getStringAttribute("modelPath").isEmpty()
                && !parsed->getBoolAttribute("relinkRequired"),
            "empty identity serializes after missing identity");
    require(std::abs(processor.dryWetParameterReference().convertFrom0to1(
                processor.dryWetParameterReference().getValue()) - .75f) < .001f,
            "empty restore preserves host dry/wet");

#if !RAVE_HAS_LIBTORCH
    auto retainedFile = juce::File::createTempFile("rave-no-torch-retained.ts");
    require(retainedFile.replaceWithText("identity only"),
            "create unsupported-backend retained identity");
    processor.prepareToPlay(48000.0, 8);
    juce::XmlElement retained("RavePluginState");
    retained.setAttribute("version", 2);
    retained.setAttribute("modelPath", retainedFile.getFullPathName());
    auto* retainedLatents = retained.createNewChildElement("Latents");
    retainedLatents->setAttribute("count", 1);
    retainedLatents->setAttribute("firstIndex", 8);
    retainedLatents->setAttribute("v8", 0.625);
    block = xmlState(retained);
    processor.setStateInformation(block.getData(), int(block.getSize()));
    juce::MemoryBlock retainedSaved;
    processor.getStateInformation(retainedSaved);
    parsed = juce::AudioProcessor::getXmlFromBinary(
        retainedSaved.getData(), int(retainedSaved.getSize()));
    require(parsed && parsed->getStringAttribute("modelPath") == retainedFile.getFullPathName(),
            "no-LibTorch build retains desired model identity");
    dynamic = parsed->getChildByName("Latents");
    require(dynamic && dynamic->getIntAttribute("count") == 1
                && std::abs(dynamic->getDoubleAttribute("v8") - 0.625) < 0.001,
            "no-LibTorch build retains desired dynamic latents");
    processor.releaseResources();
    retainedFile.deleteFile();
#endif
}

void testFactoryAudioAndState()
{
    std::unique_ptr<juce::AudioProcessor> processor(createPluginFilter());
    require(processor != nullptr, "plugin factory returns processor");
    require(processor->getName() == "RAVE Performance Instrument", "plugin name");
    require(processor->getTotalNumInputChannels() == 2, "default stereo input");
    require(processor->getTotalNumOutputChannels() == 2, "default stereo output");

    processor->prepareToPlay(48000.0, 4);
    juce::AudioBuffer<float> buffer(2, 4);
    const std::array<float, 4> left { 1.0f, 2.0f, 3.0f, 4.0f };
    const std::array<float, 4> right { 5.0f, 6.0f, 7.0f, 8.0f };
    buffer.copyFrom(0, 0, left.data(), 4);
    buffer.copyFrom(1, 0, right.data(), 4);
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    require(processor->getLatencySamples() == 4096, "plugin reports fixed transport latency");
    for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
        require(buffer.getSample(0, sample) == 0.0f && buffer.getSample(1, sample) == 0.0f,
                "prepared plugin observes the fixed delayed transport");

    auto* dryWet = findParameter(*processor, "dryWet");
    require(dryWet != nullptr, "dry/wet parameter exposed");
    require(processor->getParameters().size() == 9, "stable dry/wet plus eight macro parameters");
    for (std::size_t index = 0; index < RavePluginProcessor::macroCount; ++index)
        require(findParameter(*processor, "macro" + juce::String(index + 1)) != nullptr,
                "stable macro parameter ID exposed");
    dryWet->setValueNotifyingHost(dryWet->convertTo0to1(0.73f));
    juce::MemoryBlock state;
    processor->getStateInformation(state);

    std::unique_ptr<juce::AudioProcessor> restored(createPluginFilter());
    restored->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    auto* restoredDryWet = findParameter(*restored, "dryWet");
    require(restoredDryWet != nullptr, "restored dry/wet parameter exposed");
    require(std::abs(restoredDryWet->convertFrom0to1(restoredDryWet->getValue()) - 0.73f) < 0.001f,
            "dry/wet state restored");

    auto* concrete = dynamic_cast<RavePluginProcessor*>(processor.get());
    require(concrete != nullptr, "factory returns concrete plugin processor");
    concrete->beginMidiLearn(1);
    auto* macro1 = findParameter(*processor, "macro1");
    require(macro1 != nullptr, "first macro parameter exposed");
    ParameterListenerCounter listener;
    macro1->addListener(&listener);
    midi.addEvent(juce::MidiMessage::controllerEvent(1, 74, 127), 0);
    processor->processBlock(buffer, midi);
    require(listener.valueChanges.load() == 0,
            "MIDI callback does not enter the parameter listener path");
    concrete->publishPendingMidiParameterChanges();
    require(listener.valueChanges.load() == 1,
            "message-thread handoff publishes one coalesced parameter change");
    macro1->removeListener(&listener);
    require(std::abs(macro1->convertFrom0to1(macro1->getValue()) - 4.0f) < 0.001f,
            "learned MIDI CC controls macro");

    state.reset();
    processor->getStateInformation(state);
    restored->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    restored->prepareToPlay(48000.0, 4);
    juce::MidiBuffer restoredMidi;
    restoredMidi.addEvent(juce::MidiMessage::controllerEvent(1, 74, 0), 0);
    restored->processBlock(buffer, restoredMidi);
    dynamic_cast<RavePluginProcessor&>(*restored).publishPendingMidiParameterChanges();
    auto* restoredMacro1 = findParameter(*restored, "macro1");
    require(restoredMacro1 != nullptr, "restored macro parameter exposed");
    require(std::abs(restoredMacro1->convertFrom0to1(restoredMacro1->getValue()) + 4.0f) < 0.001f,
            "MIDI mapping restored with plugin state");

    restored->releaseResources();
    processor->releaseResources();
}

void testModelAndLatentRecall(const juce::File& modelFile)
{
    auto original = std::make_unique<RavePluginProcessor>();
    original->prepareToPlay(48000.0, 8);
    require(original->startModelLoad(modelFile), "model load starts");

    const auto loadDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (original->isModelLoading() && std::chrono::steady_clock::now() < loadDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    require(original->finishModelLoadIfReady(), "loaded model activates");
    require(original->latentDimensionCount() == 2, "fixture latent dimensions available");
    original->macroParameterReference(0).setValueNotifyingHost(
        original->macroParameterReference(0).convertTo0to1(1.25f));
    original->macroParameterReference(1).setValueNotifyingHost(
        original->macroParameterReference(1).convertTo0to1(-2.5f));

    juce::MemoryBlock state;
    original->getStateInformation(state);

    auto restored = std::make_unique<RavePluginProcessor>();
    restored->prepareToPlay(48000.0, 8);
    restored->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    const auto restoreDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (restored->isModelLoading() && std::chrono::steady_clock::now() < restoreDeadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    require(restored->finishModelLoadIfReady(), "restored model activates");
    require(restored->latentDimensionCount() == 2, "restored model metadata");
    require(std::abs(restored->latentControl(0) - 1.25f) < 0.001f,
            "first latent restored");
    require(std::abs(restored->latentControl(1) + 2.5f) < 0.001f,
            "second latent restored");

    restored->releaseResources();
    original->releaseResources();
}
void testFailedLoadKeepsPreviousModel(const juce::File& modelFile, const juce::File& junkFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "first model activates");
    require(processor->latentDimensionCount() == 2, "first model latent count");
    require(processor->setLatentControl(0, 1.5f), "latent value set before failure");

    require(loadModel(*processor, junkFile), "failing replacement is consumed");
    require(processor->latentDimensionCount() == 2,
            "previous model retained after failed replacement");
    require(std::abs(processor->latentControl(0) - 1.5f) < 0.001f,
            "latent state untouched by failed replacement");
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("failed"), "failure status is visible");
    require(status.containsIgnoreCase("current state") && status.containsIgnoreCase("active"),
            "failure status composes retention with the current active lifecycle state");

    juce::AudioBuffer<float> buffer(2, 8);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, 0.25f * static_cast<float>(sample + 1));
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}

void testFailedLoadCannotOverwriteNewerReleaseStatus(const juce::File& modelFile,
                                                      const juce::File& junkFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "model activates before failed-load interleave");

    bool seamRan = false;
    processor->failedLoadStatusInterleaveForTesting = [&] {
        seamRan = true;
        processor->releaseResources();
    };
    require(loadModel(*processor, junkFile), "failed load result is consumed after release interleave");
    require(seamRan, "failed-load status interleave seam ran");

    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("model load failed"),
            "candidate qualification failure context is preserved");
    require(status.containsIgnoreCase("installed but not running")
                && status.containsIgnoreCase("dry pass-through"),
            "failed-load commit uses the newer release lifecycle state");
    require(!status.containsIgnoreCase("previous model still active")
                && !status.containsIgnoreCase("active —"),
            "failed-load commit emits no stale active wording");

    processor->refreshLifecycleStatus();
    require(processor->modelStatus() == status,
            "subsequent timer-equivalent refresh cannot regress the composed status");
}

void testSampleRateMismatchIsRejectedBeforeActivation(const juce::File& modelFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(44100.0, 8); // the fixture model is exported at 48 kHz
    require(loadModel(*processor, modelFile), "mismatched candidate load is consumed");
    require(processor->latentDimensionCount() == 0,
            "mismatched candidate never became active");
    require(processor->modelStatus().containsIgnoreCase("sample rate"),
            "status names the sample-rate conflict");

    juce::AudioBuffer<float> buffer(2, 8);
    buffer.clear();
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}

void testEditorClosureDuringLoad(const juce::File& modelFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);

    std::unique_ptr<juce::AudioProcessorEditor> editor(processor->createEditor());
    require(editor != nullptr, "editor opens before activation");
    require(processor->startModelLoad(modelFile), "load starts with editor open");
    editor.reset(); // close the editor while the load is in flight
    require(waitForModelSettled(*processor), "load completes without an editor");
    require(processor->finishModelLoadIfReady(), "loaded model activates without an editor");
    require(processor->latentDimensionCount() == 2, "latent metadata available after editor closure");

    editor.reset(processor->createEditor());
    require(editor != nullptr, "editor reopens after activation");
    editor.reset();

    juce::AudioBuffer<float> buffer(2, 8);
    buffer.clear();
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}

void testRepeatedReplacementRemainsBounded(const juce::File& modelFile, const juce::File& junkFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);

    for (int cycle = 0; cycle < 2; ++cycle)
    {
        require(loadModel(*processor, modelFile), "successful cycle activates");
        require(processor->latentDimensionCount() == 2, "successful cycle latent count");
    }

    require(loadModel(*processor, junkFile), "failed cycle is consumed");
    require(processor->latentDimensionCount() == 2, "failed cycle retains previous model");

    require(loadModel(*processor, modelFile), "post-failure cycle activates");
    require(processor->latentDimensionCount() == 2, "post-failure cycle latent count");
    require(processor->modelRevision() >= 4, "each replacement attempt bumps the revision");

    juce::AudioBuffer<float> buffer(2, 8);
    buffer.clear();
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}
void testRestoreBeforePrepareIsDeferred(const juce::File& modelFile)
{
    // Build a saved state that references an existing model file.
    auto original = std::make_unique<RavePluginProcessor>();
    original->prepareToPlay(48000.0, 8);
    require(loadModel(*original, modelFile), "state source model activates");
    juce::MemoryBlock state;
    original->getStateInformation(state);
    original->releaseResources();

    // A processor that has never seen a host configuration must not start a
    // restore (or any load) against invented default configuration values.
    auto restored = std::make_unique<RavePluginProcessor>();
    restored->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    require(!restored->isModelLoading(), "restore does not start before prepare");
    require(restored->modelStatus().containsIgnoreCase(
                "Waiting for the host audio configuration"),
            "deferred restore status is actionable");

    // Once the host provides a configuration, that single restore call
    // proceeds automatically.
    restored->prepareToPlay(48000.0, 8);
    require(restored->isModelLoading(), "restore starts once a configuration exists");
    require(waitForModelSettled(*restored), "deferred restore settles");
    require(restored->finishModelLoadIfReady(), "deferred restore activates");
    require(restored->latentDimensionCount() == 2, "deferred restore latent count");

    restored->releaseResources();
}

void testReprepareDuringLoadKeepsPreviousModel(const juce::File& modelFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "previous model activates at 48 kHz");
    require(processor->setLatentControl(0, 1.0f), "latent set on the active model");

    // The replacement is qualified at 48 kHz in the background, then the host
    // reprepares at 44.1 kHz before activation; activation must refuse the
    // stale candidate instead of replacing the previous usable backend.
    require(processor->startModelLoad(modelFile), "replacement load starts");
    processor->prepareToPlay(44100.0, 8);
    require(waitForModelSettled(*processor), "replacement load settles");
    require(processor->finishModelLoadIfReady(), "stale candidate result is consumed");
    require(processor->latentDimensionCount() == 2,
            "previous model survives the rate change");
    // Latent storage is reset by the host reprepare itself (RAVE-04); the
    // refusal must at least keep the previous controls addressable.
    require(processor->setLatentControl(0, 1.0f), "previous latents still addressable");
    require(std::abs(processor->latentControl(0) - 1.0f) < 0.001f,
            "previous model accepts latent updates after refusal");
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("activation failed")
                || status.containsIgnoreCase("sample rate"),
            "status explains why the stale candidate was refused");

    juce::AudioBuffer<float> buffer(2, 8);
    buffer.clear();
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}

void testIncompatibleReprepareReportsStatus(const juce::File& modelFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "model activates at 48 kHz");
    require(processor->latentDimensionCount() == 2, "latent metadata available");
    require(processor->modelStatus().containsIgnoreCase("active"),
            "active claim before the incompatible reprepare");

    // The host reproves at an incompatible rate: the visible status must
    // report the incompatibility and the dry fallback instead of leaving the
    // active claim up while the model is silent.
    processor->prepareToPlay(44100.0, 8);
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("44100") && status.containsIgnoreCase("48000"),
            "status names the sample-rate conflict");
    require(status.containsIgnoreCase("dry pass-through"),
            "status names the bounded dry fallback");
    require(!status.containsIgnoreCase("active —"),
            "no active claim while the model is silent");

    juce::AudioBuffer<float> buffer(2, 8);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, 0.25f * static_cast<float>(sample + 1));
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    // Returning to the compatible configuration restores the active claim.
    processor->prepareToPlay(48000.0, 8);
    require(processor->modelStatus().containsIgnoreCase("active"),
            "active claim restored at the compatible configuration");
    require(processor->latentDimensionCount() == 2, "latent metadata restored");

    processor->releaseResources();
}

// Reset succeeds for the loader-qualification call and the activation-time
// worker start, then refuses; a later host reprepare must surface the failing
// checked reset in the visible status.
class OneShotResetPluginBackend final : public rave::ModelBackend
{
public:
    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}

    bool reset(std::string& errorMessage) override
    {
        const auto call = resetCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        if (call <= 2)
            return true;
        errorMessage = "reset refuses on call " + std::to_string(call);
        return false;
    }

    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f;
        return true;
    }

    std::atomic<int> resetCalls { 0 };
};

void testResetFailureAtReprepareReportsStatus(const juce::File& modelFile)
{
    auto backend = std::make_shared<OneShotResetPluginBackend>();
    auto processor = std::make_unique<RavePluginProcessor>(
        [backend]() -> rave::ModelBackendPtr { return backend; });
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "one-shot reset model activates");
    require(backend->resetCalls.load() == 2,
            "qualification and activation resets consumed before the reprepare");
    require(processor->modelStatus().containsIgnoreCase("active"),
            "activation claims the active model");

    // The host reprepare re-runs the engine prepare; the next checked reset
    // refuses and the status must visibly report the failure.
    processor->prepareToPlay(48000.0, 8);
    require(backend->resetCalls.load() == 3, "reprepare reached the backend reset");
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("reset") && status.containsIgnoreCase("dry pass-through"),
            "reset failure at reprepare is visible");
    require(!status.containsIgnoreCase("active —"),
            "no active claim while the model cannot start");

    juce::AudioBuffer<float> buffer(2, 8);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, 0.25f * static_cast<float>(sample + 1));
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}
// Prepare throws for calls in [prepareThrowFromCall, prepareThrowFromCall + prepareThrowCount); other lifecycle calls succeed.
class PluginNthCallThrowingBackend final : public rave::ModelBackend
{
public:
    void prepare(double, std::size_t) override
    {
        const auto call = prepareCalls.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto offset = call - prepareThrowFromCall;
        if (prepareThrowCount > 0 && offset >= 0 && offset < prepareThrowCount)
            throw std::runtime_error("prepare refuses on call " + std::to_string(call));
    }

    bool load(const std::string&, std::string&) override { return true; }
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 0; }

    bool process(const std::span<const float> input,
                 const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f;
        return true;
    }

    std::atomic<int> prepareCalls { 0 };
    int prepareThrowFromCall = 0;
    int prepareThrowCount = 0;
};

void testReleaseReportsInstalledNotRunning(const juce::File& modelFile)
{
    auto processor = std::make_unique<RavePluginProcessor>();
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "model activates");
    require(processor->modelStatus().containsIgnoreCase("active"), "active claim before release");

    // Release/device stop must publish the usability transition so the visible
    // status renders installed-but-not-running instead of the active claim.
    processor->releaseResources();
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("not running"),
            "release renders installed-but-not-running");
    require(!status.containsIgnoreCase("active —"), "no stale active claim after release");

    // A reprepare restores the active claim via successful prepare/start.
    processor->prepareToPlay(48000.0, 8);
    require(processor->modelStatus().containsIgnoreCase("active"),
            "active claim restored after reprepare");
    processor->releaseResources();
}

void testPrepareThrowAtReprepareReportsStatus(const juce::File& modelFile)
{
    // Qualification prepares (call 1) and activation worker-prepare (call 2)
    // succeed; the host reprepare (call 3) throws inside the backend.
    auto backend = std::make_shared<PluginNthCallThrowingBackend>();
    backend->prepareThrowFromCall = 3;
    backend->prepareThrowCount = 1;
    auto processor = std::make_unique<RavePluginProcessor>(
        [backend]() -> rave::ModelBackendPtr { return backend; });
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "model activates before the throwing reprepare");

    // prepareToPlay must return normally with the backend prepare throw
    // contained, and the status must name the failure and the dry fallback.
    processor->prepareToPlay(48000.0, 8);
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("backend prepare failed")
                && status.containsIgnoreCase("dry pass-through"),
            "prepare throw is visible in the status");
    require(!status.containsIgnoreCase("active —"), "no active claim after the prepare throw");

    juce::AudioBuffer<float> buffer(2, 8);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, 0.25f * static_cast<float>(sample + 1));
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
}
// Fresh backend instances share one reset counter; calls 4 and 5 fail so a
// replacement candidate fails at activation start (call 4) and the rollback
// restart of the previous latent-capable backend (call 5) also fails, which
// abandons the backend.
class LatentSharedCounterResetBackend final : public rave::ModelBackend
{
public:
    explicit LatentSharedCounterResetBackend(std::shared_ptr<std::atomic<int>> sharedCounter)
        : resetCounter(std::move(sharedCounter)) {}

    bool load(const std::string&, std::string&) override { return true; }
    void prepare(double, std::size_t) override {}

    bool reset(std::string& errorMessage) override
    {
        const auto call = resetCounter->fetch_add(1, std::memory_order_relaxed) + 1;
        if (call < failFromCall || call >= failFromCall + failCount)
            return true;
        errorMessage = "reset refuses on call " + std::to_string(call);
        return false;
    }

    std::size_t latentDimensionCount() const noexcept override { return 2; }

    bool process(const std::span<const float> input,
                 const std::span<const float> latentControls,
                 const std::span<float> output) override
    {
        for (std::size_t index = 0; index < input.size(); ++index)
            output[index] = input[index] * 3.0f + latentControls[0];
        return true;
    }

    std::shared_ptr<std::atomic<int>> resetCounter;
    int failFromCall = 4;
    int failCount = 2;
};

#if RAVE_HAS_LIBTORCH
class RelinkTwelveLatentBackend final : public rave::ModelBackend
{
public:
    explicit RelinkTwelveLatentBackend(std::shared_ptr<std::atomic<bool>> reject,
                                       const bool failActivation = false)
        : rejectLoad(std::move(reject)), failActivationReset(failActivation) {}
    bool load(const std::string&, std::string& error) override
    {
        if (!rejectLoad->load(std::memory_order_acquire)) return true;
        error = "injected qualification refusal";
        return false;
    }
    void prepare(double, std::size_t) override {}
    bool reset(std::string& error) override
    {
        ++resetCalls;
        if (failActivationReset && resetCalls == 2)
        {
            error = "injected activation reset refusal";
            return false;
        }
        return true;
    }
    std::size_t latentDimensionCount() const noexcept override { return 12; }
    bool process(const std::span<const float> input, const std::span<const float>,
                 const std::span<float> output) override
    {
        for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * 3.0f;
        return true;
    }
private:
    std::shared_ptr<std::atomic<bool>> rejectLoad;
    bool failActivationReset = false;
    int resetCalls = 0;
};

class BlockingRestoreBackend final : public rave::ModelBackend
{
public:
    BlockingRestoreBackend(std::shared_ptr<std::atomic<bool>> enteredValue,
                           std::shared_ptr<std::atomic<bool>> releaseValue)
        : entered(std::move(enteredValue)), release(std::move(releaseValue)) {}
    bool load(const std::string&, std::string&) override
    {
        entered->store(true, std::memory_order_release);
        while (!release->load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }
    void prepare(double, std::size_t) override {}
    bool reset(std::string&) override { return true; }
    std::size_t latentDimensionCount() const noexcept override { return 12; }
    bool process(std::span<const float>, std::span<const float>,
                 std::span<float>) override { return true; }
private:
    std::shared_ptr<std::atomic<bool>> entered;
    std::shared_ptr<std::atomic<bool>> release;
};

void testInFlightRestoreSnapshotRetainsDesiredState()
{
    auto entered = std::make_shared<std::atomic<bool>>(false);
    auto release = std::make_shared<std::atomic<bool>>(false);
    auto modelFile = juce::File::createTempFile("rave-pending-restore.ts");
    require(modelFile.replaceWithText("blocking backend ignores contents"),
            "create pending restore identity");
    RavePluginProcessor processor([entered, release] {
        return std::make_shared<BlockingRestoreBackend>(entered, release);
    });
    processor.prepareToPlay(48000.0, 8);

    juce::XmlElement restored("RavePluginState");
    restored.setAttribute("version", 2);
    restored.setAttribute("modelPath", modelFile.getFullPathName());
    auto* latents = restored.createNewChildElement("Latents");
    latents->setAttribute("count", 4);
    latents->setAttribute("firstIndex", 8);
    for (int index = 8; index < 12; ++index)
        latents->setAttribute("v" + juce::String(index), 0.125 * index);
    auto state = xmlState(restored);
    processor.setStateInformation(state.getData(), int(state.getSize()));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered->load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(entered->load(std::memory_order_acquire), "restore qualification is in flight");

    juce::MemoryBlock snapshot;
    processor.getStateInformation(snapshot);
    auto parsed = juce::AudioProcessor::getXmlFromBinary(
        snapshot.getData(), int(snapshot.getSize()));
    auto* savedLatents = parsed != nullptr ? parsed->getChildByName("Latents") : nullptr;
    require(parsed && parsed->getStringAttribute("modelPath") == modelFile.getFullPathName(),
            "in-flight restore snapshot retains desired model identity");
    require(savedLatents && savedLatents->getIntAttribute("count") == 4
                && std::abs(savedLatents->getDoubleAttribute("v8") - 1.0) < 0.001,
            "in-flight restore snapshot retains desired dynamic latents");

    release->store(true, std::memory_order_release);
    require(waitForModelSettled(processor) && processor.finishModelLoadIfReady(),
            "pending restore completes after coherent snapshot");
    processor.releaseResources();
    modelFile.deleteFile();
}

void testDeferredRestoreDisappearanceBecomesRelinkRequired()
{
    auto reject = std::make_shared<std::atomic<bool>>(false);
    auto modelFile = juce::File::createTempFile("rave-deferred-disappears.ts");
    require(modelFile.replaceWithText("injected backend ignores file contents"),
            "create deferred restore model identity");
    RavePluginProcessor processor([reject] {
        return std::make_shared<RelinkTwelveLatentBackend>(reject);
    });

    juce::XmlElement restored("RavePluginState");
    restored.setAttribute("version", 2);
    restored.setAttribute("modelPath", modelFile.getFullPathName());
    restored.setAttribute("dryWet", 1.0);
    for (int index = 0; index < 8; ++index)
        restored.setAttribute("macro" + juce::String(index + 1), -0.8 + index * 0.2);
    auto* latents = restored.createNewChildElement("Latents");
    latents->setAttribute("count", 4);
    latents->setAttribute("firstIndex", 8);
    for (int index = 8; index < 12; ++index)
        latents->setAttribute("v" + juce::String(index), -0.8 + index * 0.2);
    auto state = xmlState(restored);
    processor.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    require(modelFile.deleteFile(), "model disappears before host prepare");
    processor.prepareToPlay(48000.0, 8);
    require(processor.isRelinkRequired()
                && processor.requestedModelPath() == modelFile.getFullPathName()
                && processor.modelStatus().containsIgnoreCase("relink required"),
            "newest disappeared deferred restore becomes actionable relink state");
    juce::MemoryBlock retained;
    processor.getStateInformation(retained);
    auto parsed = juce::AudioProcessor::getXmlFromBinary(
        retained.getData(), static_cast<int>(retained.getSize()));
    require(parsed != nullptr && parsed->getChildByName("Latents") != nullptr
                && parsed->getChildByName("Latents")->getIntAttribute("count") == 4,
            "disappeared restore retains all twelve latent values");

    require(modelFile.replaceWithText("recreated relink model"), "recreate relink model");
    require(processor.relinkMissingModel(modelFile), "relink API is immediately ready");
    require(waitForModelSettled(processor) && processor.finishModelLoadIfReady(),
            "recreated model relinks successfully");
    require(!processor.isRelinkRequired() && processor.latentDimensionCount() == 12,
            "successful relink restores model and clears requirement");
    for (std::size_t index = 0; index < 12; ++index)
        require(std::abs(processor.latentControl(index) - (-0.8f + float(index) * 0.2f)) < .011f,
                "successful relink restores retained latent by index");
    processor.releaseResources();

    auto oldFile = juce::File::createTempFile("rave-stale-deferred.ts");
    auto newestFile = juce::File::createTempFile("rave-newest-deferred.ts");
    require(oldFile.replaceWithText("old") && newestFile.replaceWithText("new"),
            "create stale-generation fixtures");
    RavePluginProcessor stale([reject] {
        return std::make_shared<RelinkTwelveLatentBackend>(reject);
    });
    restored.setAttribute("modelPath", oldFile.getFullPathName());
    state = xmlState(restored);
    stale.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    require(stale.startModelLoad(newestFile), "newer deferred request supersedes restore");
    require(oldFile.deleteFile(), "stale restored model disappears");
    stale.prepareToPlay(48000.0, 8);
    require(waitForModelSettled(stale) && stale.finishModelLoadIfReady(),
            "newest deferred model activates");
    require(!stale.isRelinkRequired() && stale.requestedModelPath().isEmpty(),
            "stale disappearance cannot create relink state");
    stale.releaseResources();

    RavePluginProcessor activationRace([reject] {
        return std::make_shared<RelinkTwelveLatentBackend>(reject);
    });
    activationRace.prepareToPlay(48000.0, 8);
    reject->store(false, std::memory_order_release);
    require(activationRace.startModelLoad(newestFile), "activation-race candidate starts");
    require(waitForModelSettled(activationRace), "activation-race candidate qualifies");
    std::atomic<bool> activationReachedMutation { false };
    std::atomic<bool> activationTransactionEntered { false };
    std::thread activationThread;
    activationRace.activationMutationInterleaveForTesting = [&] {
        activationReachedMutation.store(true, std::memory_order_release);
    };
    bool snapshotSeamEntered = false;
    activationRace.stateSnapshotInterleaveForTesting = [&] {
        if (snapshotSeamEntered)
            return;
        snapshotSeamEntered = true;
        activationThread = std::thread([&] {
            activationRace.runActivationMutationTransactionForTesting([&] {
                activationTransactionEntered.store(true, std::memory_order_release);
            });
        });
        while (!activationReachedMutation.load(std::memory_order_acquire))
            std::this_thread::yield();
        require(!activationTransactionEntered.load(std::memory_order_acquire),
                "activation transaction cannot enter during coherent state snapshot");
    };
    juce::MemoryBlock blockedSnapshot;
    activationRace.getStateInformation(blockedSnapshot);
    activationRace.stateSnapshotInterleaveForTesting = {};
    activationThread.join();
    require(activationTransactionEntered.load(std::memory_order_acquire),
            "activation transaction proceeds after coherent snapshot releases");
    activationRace.activationMutationInterleaveForTesting = {};
    require(activationRace.finishModelLoadIfReady()
                && activationRace.latentDimensionCount() == 12,
            "qualified candidate activates after blocked snapshot test");

    require(activationRace.startModelLoad(newestFile), "stale activation candidate restarts");
    require(waitForModelSettled(activationRace), "stale activation candidate qualifies");
    bool newerRestoreCommitted = false;
    activationRace.activationMutationInterleaveForTesting = [&] {
        juce::XmlElement empty("RavePluginState");
        empty.setAttribute("version", 2);
        auto emptyState = xmlState(empty);
        activationRace.setStateInformation(emptyState.getData(), int(emptyState.getSize()));
        newerRestoreCommitted = true;
    };
    require(activationRace.finishModelLoadIfReady(), "stale activation result consumed");
    activationRace.activationMutationInterleaveForTesting = {};
    require(newerRestoreCommitted && activationRace.latentDimensionCount() == 0
                && !activationRace.modelStatus().containsIgnoreCase("active"),
            "newer empty restore wins before activation mutation transaction");
    juce::AudioBuffer<float> emptyRestoreAudio(2, 2048);
    juce::MidiBuffer emptyRestoreMidi;
    for (int blockIndex = 0; blockIndex < 3; ++blockIndex)
    {
        for (int channel = 0; channel < emptyRestoreAudio.getNumChannels(); ++channel)
            for (int sample = 0; sample < emptyRestoreAudio.getNumSamples(); ++sample)
                emptyRestoreAudio.setSample(channel, sample, 1.0f);
        activationRace.processBlock(emptyRestoreAudio, emptyRestoreMidi);
    }
    require(std::abs(emptyRestoreAudio.getSample(0, 2047) - 1.0f) < 0.00001f,
            "active-to-empty restore preserves prepared aligned dry transport");
    activationRace.releaseResources();

    modelFile.deleteFile();
    newestFile.deleteFile();
}

void testMissingRelinkSuppressionAndTwelveLatents(const juce::File& modelFile)
{
    auto reject = std::make_shared<std::atomic<bool>>(false);
    auto failActivation = std::make_shared<std::atomic<bool>>(false);
    RavePluginProcessor processor([reject, failActivation] {
        return std::make_shared<RelinkTwelveLatentBackend>(
            reject, failActivation->load(std::memory_order_acquire));
    });
    processor.prepareToPlay(48000.0, 2048);
    require(loadModel(processor, modelFile), "injected twelve-latent model activates");
    processor.dryWetParameterReference().setValueNotifyingHost(1.0f);

    juce::XmlElement missing("RavePluginState");
    missing.setAttribute("version", 2); missing.setAttribute("modelPath", "/missing/relink-model.ts");
    missing.setAttribute("dryWet", 1.0);
    for (int i = 0; i < 8; ++i) missing.setAttribute("macro" + juce::String(i + 1), -0.8 + i * 0.2);
    auto* latents = missing.createNewChildElement("Latents");
    latents->setAttribute("count", 4); latents->setAttribute("firstIndex", 8);
    for (int i = 8; i < 12; ++i) latents->setAttribute("v" + juce::String(i), -0.8 + i * 0.2);
    auto state = xmlState(missing);
    processor.setStateInformation(state.getData(), int(state.getSize()));
    require(processor.isRelinkRequired() && processor.requestedModelPath().contains("relink-model.ts"),
            "missing restore retains relink identity");
    require(std::abs(processor.dryWetParameterReference().convertFrom0to1(
                processor.dryWetParameterReference().getValue()) - 1.0f) < .001f,
            "missing restore preserves host dry/wet");
    juce::MemoryBlock retained; processor.getStateInformation(retained);
    auto parsed = juce::AudioProcessor::getXmlFromBinary(retained.getData(), int(retained.getSize()));
    require(parsed && parsed->getChildByName("Latents")->getIntAttribute("count") == 4,
            "all twelve values retain eight authoritative macros plus four dynamic latents");
    for (int i = 0; i < 8; ++i)
        require(std::abs(float(parsed->getDoubleAttribute("macro" + juce::String(i + 1)))
                             - (-0.8f + float(i) * 0.2f)) < .011f,
                "retained macro value serializes by index");
    for (int i = 8; i < 12; ++i)
        require(std::abs(float(parsed->getChildByName("Latents")->getDoubleAttribute(
                                 "v" + juce::String(i)))
                             - (-0.8f + float(i) * 0.2f)) < .011f,
                "retained dynamic latent serializes by index");
    juce::AudioBuffer<float> audio(2, 2048); juce::MidiBuffer midi;
    for (int block = 0; block < 3; ++block)
    {
        audio.clear();
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < audio.getNumSamples(); ++sample) audio.setSample(channel, sample, 1.0f);
        processor.processBlock(audio, midi);
        requireFiniteOutput(audio);
    }
    require(std::abs(audio.getSample(0, 2047) - 1.0f) < .00001f,
            "missing restore suppresses wet to exactly aligned dry");

    reject->store(true, std::memory_order_release);
    missing.setAttribute("modelPath", modelFile.getFullPathName());
    missing.setAttribute("relinkRequired", true);
    auto existingReplacementState = xmlState(missing);
    processor.setStateInformation(existingReplacementState.getData(),
                                  int(existingReplacementState.getSize()));
    require(waitForModelSettled(processor) && processor.finishModelLoadIfReady(),
            "existing-path replacement qualification failure settles");
    require(processor.isRelinkRequired()
                && processor.modelStatus().containsIgnoreCase("injected qualification refusal")
                && processor.modelStatus().containsIgnoreCase("aligned dry")
                && processor.modelStatus().containsIgnoreCase("relink required")
                && !processor.modelStatus().containsIgnoreCase("active"),
            "qualification-failed relink reports concrete suppressed outcome");
    juce::MemoryBlock failed; processor.getStateInformation(failed);
    parsed = juce::AudioProcessor::getXmlFromBinary(failed.getData(), int(failed.getSize()));
    require(parsed && parsed->getChildByName("Latents")->getIntAttribute("count") == 4,
            "failed relink retains dynamic latent payload");
    for (int channel = 0; channel < 2; ++channel)
        for (int sample = 0; sample < audio.getNumSamples(); ++sample) audio.setSample(channel, sample, 1.0f);
    processor.processBlock(audio, midi);
    require(std::abs(audio.getSample(0, 2047) - 1.0f) < .00001f,
            "failed relink remains suppressed to aligned dry");

    reject->store(false, std::memory_order_release);
    require(processor.relinkMissingModel(modelFile), "successful relink request starts");
    require(waitForModelSettled(processor) && processor.finishModelLoadIfReady(), "successful relink settles");
    require(!processor.isRelinkRequired() && processor.latentDimensionCount() == 12,
            "successful relink clears requirement and restores dimension count");
    for (std::size_t i = 0; i < 12; ++i)
        require(std::abs(processor.latentControl(i) - (-0.8f + float(i) * 0.2f)) < .011f,
                "successful relink restores every retained latent by index");
    bool wetObserved = false;
    for (int block = 0; block < 64 && !wetObserved; ++block)
    {
        audio.clear();
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < audio.getNumSamples(); ++sample)
                audio.setSample(channel, sample, 1.0f);
        processor.processBlock(audio, midi);
        requireFiniteOutput(audio);
        wetObserved = audio.getSample(0, 2047) > 1.1f;
        // Give the real inference worker a bounded scheduling window before
        // the next frame deadline; a bare yield is not a scheduling guarantee.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    require(wetObserved, "successful relink removes suppression and wet output recovers");

    // Re-enter relink state and inject a failure only on the activation reset:
    // qualification reset succeeds, while activation reset refuses and the
    // old backend rolls back successfully behind wet suppression.
    failActivation->store(true, std::memory_order_release);
    processor.setStateInformation(existingReplacementState.getData(),
                                  int(existingReplacementState.getSize()));
    require(processor.isRelinkRequired(),
            "existing-path schema replacement retains relink authority");
    require(waitForModelSettled(processor) && processor.finishModelLoadIfReady(),
            "activation-failing relink settles");
    const auto activationStatus = processor.modelStatus();
    require(processor.isRelinkRequired()
                && activationStatus.containsIgnoreCase("injected activation reset refusal")
                && activationStatus.containsIgnoreCase("aligned dry")
                && activationStatus.containsIgnoreCase("relink required")
                && !activationStatus.containsIgnoreCase("previous model retained")
                && !activationStatus.containsIgnoreCase("active"),
            "activation-failed relink reports suppressed outcome, not rollback audibility");
    juce::MemoryBlock activationFailedState;
    processor.getStateInformation(activationFailedState);
    parsed = juce::AudioProcessor::getXmlFromBinary(
        activationFailedState.getData(), int(activationFailedState.getSize()));
    require(parsed && parsed->getChildByName("Latents")->getIntAttribute("count") == 4,
            "activation-failed relink retains identity and dynamic latent payload");
    for (int block = 0; block < 3; ++block)
    {
        for (int channel = 0; channel < 2; ++channel)
            for (int sample = 0; sample < audio.getNumSamples(); ++sample)
                audio.setSample(channel, sample, 1.0f);
        processor.processBlock(audio, midi);
    }
    require(std::abs(audio.getSample(0, 2047) - 1.0f) < .00001f,
            "activation-failed relink remains suppressed to aligned dry");
    processor.releaseResources();
}
#endif

void testRollbackFailureClearsLatentsAndSerializedModelPath(const juce::File& modelFile)
{
    auto counter = std::make_shared<std::atomic<int>>(0);
    auto processor = std::make_unique<RavePluginProcessor>(
        [counter]() -> rave::ModelBackendPtr {
            return std::make_shared<LatentSharedCounterResetBackend>(counter);
        });
    processor->prepareToPlay(48000.0, 8);
    require(loadModel(*processor, modelFile), "latent previous model activates");
    require(counter->load() == 2, "qualification and activation resets consumed");
    require(processor->latentDimensionCount() == 2, "previous latent count exposed");
    require(processor->setLatentControl(0, 0.75f), "latent set before the failed replacement");

    // Replacement: qualification (call 3) passes; activation start (call 4)
    // fails; the rollback restart of the previous (call 5) also fails, so the
    // backend is abandoned.
    require(loadModel(*processor, modelFile), "failing replacement is consumed");
    require(processor->latentDimensionCount() == 0, "no model means zero latents");
    require(!processor->setLatentControl(0, 1.0f), "latent storage cleared with the abandoned backend");
    const auto status = processor->modelStatus();
    require(status.containsIgnoreCase("activation failed") && status.containsIgnoreCase("dry pass-through"),
            "abandonment status is truthful");
    processor->refreshLifecycleStatus();
    require(processor->modelStatus() == status,
            "immediate lifecycle refresh preserves explicit activation failure context");

    // The dropped model's path must not survive serialization: restoring the
    // state into a fresh processor must not start loading the dead path.
    juce::MemoryBlock state;
    processor->getStateInformation(state);
    auto restorer = std::make_unique<RavePluginProcessor>();
    restorer->prepareToPlay(48000.0, 8);
    restorer->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    require(!restorer->isModelLoading(), "no dropped model path is restored");
    require(!restorer->modelStatus().containsIgnoreCase("Restoring"),
            "no restoring claim for a dropped model");

    // A later healthy replacement activates and serializes its identity again.
    require(loadModel(*processor, modelFile), "healthy replacement activates after abandonment");
    require(processor->latentDimensionCount() == 2, "replacement latent count exposed");
    require(processor->modelStatus().containsIgnoreCase("active"), "replacement claims active");
    juce::MemoryBlock recoveredState;
    processor->getStateInformation(recoveredState);
    auto recoveredRestorer = std::make_unique<RavePluginProcessor>();
    recoveredRestorer->prepareToPlay(48000.0, 8);
    recoveredRestorer->setStateInformation(recoveredState.getData(),
                                           static_cast<int>(recoveredState.getSize()));
    require(recoveredRestorer->isModelLoading(),
            "healthy replacement serializes its model path for restore");
    require(waitForModelSettled(*recoveredRestorer), "restored replacement settles");
    require(recoveredRestorer->finishModelLoadIfReady(), "restored replacement loads");
    require(recoveredRestorer->latentDimensionCount() == 2, "restored replacement latent count");
    recoveredRestorer->releaseResources();

    processor->releaseResources();
}

void testStateRestoreReconcilesMidiMailboxes()
{
    RavePluginProcessor processor;
    processor.prepareToPlay(48000.0, 8);
    juce::AudioBuffer<float> audio(2, 8);
    audio.clear();

    processor.beginMidiLearn(0);
    juce::MidiBuffer dryMidi;
    dryMidi.addEvent(juce::MidiMessage::controllerEvent(1, 70, 127), 0);
    processor.processBlock(audio, dryMidi);
    processor.beginMidiLearn(1);
    juce::MidiBuffer macroMidi;
    macroMidi.addEvent(juce::MidiMessage::controllerEvent(1, 71, 127), 0);
    processor.processBlock(audio, macroMidi);

    juce::MemoryBlock unpublishedMidiState;
    processor.getStateInformation(unpublishedMidiState);
    auto unpublished = juce::AudioProcessor::getXmlFromBinary(
        unpublishedMidiState.getData(), int(unpublishedMidiState.getSize()));
    require(unpublished
                && std::abs(unpublished->getDoubleAttribute("dryWet") - 1.0) < 0.001
                && std::abs(unpublished->getDoubleAttribute("macro1") - 4.0) < 0.001,
            "state snapshot captures newest unpublished MIDI-controlled values");

    juce::XmlElement restored("RavePluginState");
    restored.setAttribute("version", 2);
    restored.setAttribute("dryWet", 0.25);
    restored.setAttribute("macro1", -2.0);
    restored.setAttribute("midiCcDryWet", 70);
    restored.setAttribute("midiCc1", 71);
    auto state = xmlState(restored);
    processor.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    processor.publishPendingMidiParameterChanges();
    require(std::abs(processor.dryWetParameterReference().convertFrom0to1(
                         processor.dryWetParameterReference().getValue()) - 0.25f) < 0.001f
                && std::abs(processor.macroParameterReference(0).convertFrom0to1(
                                processor.macroParameterReference(0).getValue()) + 2.0f) < 0.001f,
            "validated state supersedes unpublished pre-restore MIDI values");

    juce::MidiBuffer stalePublish;
    stalePublish.addEvent(juce::MidiMessage::controllerEvent(1, 71, 127), 0);
    processor.processBlock(audio, stalePublish);
    processor.midiPublishInterleaveForTesting = [&] {
        processor.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    };
    processor.publishPendingMidiParameterChanges();
    processor.midiPublishInterleaveForTesting = {};
    require(std::abs(processor.macroParameterReference(0).convertFrom0to1(
                         processor.macroParameterReference(0).getValue()) + 2.0f) < 0.001f,
            "restore epoch aborts publisher that captured stale MIDI");

    juce::XmlElement nested("RavePluginState");
    nested.setAttribute("version", 2);
    nested.setAttribute("macro1", 1.0);
    nested.setAttribute("midiCc1", 71);
    nested.setAttribute("modelPath", "/definitely/missing/nested-rave-model.ts");
    const auto nestedState = xmlState(nested);
    juce::XmlElement outerNested("RavePluginState");
    outerNested.setAttribute("version", 2);
    outerNested.setAttribute("macro1", -3.0);
    outerNested.setAttribute("midiCc1", 72);
    outerNested.setAttribute("modelPath", "/definitely/missing/outer-rave-model.ts");
    const auto outerNestedState = xmlState(outerNested);
    bool nestedPublisherReturned = false;
    bool nestedRestoreEntered = false;
    processor.stateRestoreMidiAcknowledgeInterleaveForTesting = [&] {
        if (nestedRestoreEntered)
            return;
        nestedRestoreEntered = true;
        processor.setStateInformation(nestedState.getData(), int(nestedState.getSize()));
        processor.publishPendingMidiParameterChanges();
        nestedPublisherReturned = true;
    };
    processor.setStateInformation(
        outerNestedState.getData(), static_cast<int>(outerNestedState.getSize()));
    processor.stateRestoreMidiAcknowledgeInterleaveForTesting = {};
    require(nestedPublisherReturned
                && std::abs(processor.macroParameterReference(0).convertFrom0to1(
                                processor.macroParameterReference(0).getValue()) - 1.0f) < 0.001f
                && processor.requestedModelPath().contains("nested-rave-model.ts")
                && !processor.requestedModelPath().contains("outer-rave-model.ts"),
            "nested restore keeps newer parameters and model identity authoritative");

    juce::MidiBuffer oldPending;
    oldPending.addEvent(juce::MidiMessage::controllerEvent(1, 71, 10), 0);
    processor.processBlock(audio, oldPending);
    bool injected = false;
    bool restoredObservedAtBoundary = false;
    processor.stateRestoreMidiAcknowledgeInterleaveForTesting = [&] {
        restoredObservedAtBoundary = std::abs(
            processor.macroParameterReference(0).convertFrom0to1(
                processor.macroParameterReference(0).getValue()) + 2.0f) < 0.001f;
        injected = true;
        juce::MidiBuffer newer;
        newer.addEvent(juce::MidiMessage::controllerEvent(1, 71, 127), 0);
        processor.processBlock(audio, newer);
    };
    processor.setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    processor.stateRestoreMidiAcknowledgeInterleaveForTesting = {};
    require(injected && restoredObservedAtBoundary,
            "state parameters commit before MIDI acknowledgement boundary");
    processor.publishPendingMidiParameterChanges();
    require(std::abs(processor.macroParameterReference(0).convertFrom0to1(
                         processor.macroParameterReference(0).getValue()) - 4.0f) < 0.001f,
            "newer boundary MIDI remains authoritative and publishes later");

    juce::MidiBuffer pendingBeforeInvalid;
    pendingBeforeInvalid.addEvent(juce::MidiMessage::controllerEvent(1, 71, 0), 0);
    processor.processBlock(audio, pendingBeforeInvalid);
    juce::XmlElement invalid("RavePluginState");
    invalid.setAttribute("version", 99);
    const auto invalidState = xmlState(invalid);
    processor.setStateInformation(invalidState.getData(), static_cast<int>(invalidState.getSize()));
    processor.publishPendingMidiParameterChanges();
    require(std::abs(processor.macroParameterReference(0).convertFrom0to1(
                         processor.macroParameterReference(0).getValue()) + 4.0f) < 0.001f,
            "rejected malformed state does not acknowledge pending MIDI");
    processor.releaseResources();
}

void testLongSysExCallbackAllocationAndMailboxInterleaving()
{
    RavePluginProcessor processor;
    processor.prepareToPlay(48000.0, 8);
    juce::AudioBuffer<float> audio(2, 8);
    audio.clear();

    // MidiBuffer encodes event size in 16 bits. Use a comfortably
    // representable event that still exceeds MidiMessage's inline storage.
    std::array<std::uint8_t, 1024> sysExBytes {};
    sysExBytes.fill(0x55);
    sysExBytes.front() = 0xf0;
    sysExBytes.back() = 0xf7;
    juce::MidiBuffer sysEx;
    require(sysEx.addEvent(sysExBytes.data(), static_cast<int>(sysExBytes.size()), 0),
            "representable long SysEx insertion succeeds");
    require(sysEx.getNumEvents() == 1, "allocation audit processes one actual SysEx event");
    allocationAudit::count.store(0, std::memory_order_relaxed);
    allocationAudit::enabled.store(true, std::memory_order_release);
    processor.processBlock(audio, sysEx);
    allocationAudit::enabled.store(false, std::memory_order_release);
    require(allocationAudit::count.load(std::memory_order_acquire) == 0,
            "long SysEx is inspected without callback allocation");

    processor.beginMidiLearn(1);
    const auto sendMalformed = [&](const std::uint8_t* bytes, const int byteCount) {
        processor.applyRawMidiEventForTesting(bytes, byteCount);
        processor.publishPendingMidiParameterChanges();
        require(processor.isLearningMidiTarget(1),
                "malformed raw event does not consume MIDI learn");
        require(processor.midiControllerForTarget(1) == -1,
                "malformed raw event does not change mapping");
        auto& macro = processor.macroParameterReference(0);
        require(std::abs(macro.convertFrom0to1(macro.getValue())) < 0.001f,
                "malformed raw event does not update control mailbox");
    };
    const std::array<std::uint8_t, 2> shortCc {0xb0, 74};
    const std::array<std::uint8_t, 4> longCc {0xb0, 74, 64, 0};
    const std::array<std::uint8_t, 3> wrongStatus {0x90, 74, 64};
    const std::array<std::uint8_t, 3> highController {0xb0, 0x80, 64};
    const std::array<std::uint8_t, 3> highValue {0xbf, 74, 0x80};
    sendMalformed(shortCc.data(), static_cast<int>(shortCc.size()));
    sendMalformed(longCc.data(), static_cast<int>(longCc.size()));
    sendMalformed(wrongStatus.data(), static_cast<int>(wrongStatus.size()));
    sendMalformed(highController.data(), static_cast<int>(highController.size()));
    sendMalformed(highValue.data(), static_cast<int>(highValue.size()));

    const std::array<std::uint8_t, 3> validCc {0xbf, 74, 64};
    juce::MidiBuffer valid;
    require(valid.addEvent(validCc.data(), static_cast<int>(validCc.size()), 0),
            "valid raw CC insertion succeeds");
    processor.processBlock(audio, valid);
    require(!processor.isLearningMidiTarget(1) && processor.midiControllerForTarget(1) == 74,
            "valid CC consumes learn only after malformed events were ignored");
    processor.publishPendingMidiParameterChanges();
    require(std::abs(processor.macroParameterReference(0).convertFrom0to1(
                         processor.macroParameterReference(0).getValue())
                     - (-4.0f + 8.0f * 64.0f / 127.0f)) < 0.011f,
            "valid CC updates learned target");

    juce::MidiBuffer first;
    first.addEvent(juce::MidiMessage::controllerEvent(1, 74, 10), 0);
    processor.processBlock(audio, first);
    bool injected = false;
    processor.midiPublishInterleaveForTesting = [&] {
        if (injected) return;
        injected = true;
        juce::MidiBuffer newer;
        newer.addEvent(juce::MidiMessage::controllerEvent(1, 74, 127), 0);
        processor.processBlock(audio, newer);
    };
    processor.publishPendingMidiParameterChanges();
    processor.midiPublishInterleaveForTesting = {};
    processor.publishPendingMidiParameterChanges();
    auto& macro = processor.macroParameterReference(0);
    require(std::abs(macro.convertFrom0to1(macro.getValue()) - 4.0f) < 0.001f,
            "forced callback/publisher interleaving converges to newest MIDI value");
    processor.releaseResources();
}
} // namespace

int main(const int argc, const char* const* argv)
{
    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    testStateSchemaMigrationsBoundsAndMacroAuthority();
    testFactoryAudioAndState();
    testStateRestoreReconcilesMidiMailboxes();
    testLongSysExCallbackAllocationAndMailboxInterleaving();
    if (argc == 2)
    {
        testModelAndLatentRecall(juce::File(argv[1]));

        auto junkFile = juce::File::createTempFile("rave-invalid-model");
        junkFile.replaceWithText("this is not a TorchScript model");
        testFailedLoadKeepsPreviousModel(juce::File(argv[1]), junkFile);
        testFailedLoadCannotOverwriteNewerReleaseStatus(juce::File(argv[1]), junkFile);
        testSampleRateMismatchIsRejectedBeforeActivation(juce::File(argv[1]));
        testRestoreBeforePrepareIsDeferred(juce::File(argv[1]));
        testReprepareDuringLoadKeepsPreviousModel(juce::File(argv[1]));
        testIncompatibleReprepareReportsStatus(juce::File(argv[1]));
        testResetFailureAtReprepareReportsStatus(juce::File(argv[1]));
        testReleaseReportsInstalledNotRunning(juce::File(argv[1]));
        testPrepareThrowAtReprepareReportsStatus(juce::File(argv[1]));
        testRollbackFailureClearsLatentsAndSerializedModelPath(juce::File(argv[1]));
#if RAVE_HAS_LIBTORCH
        testInFlightRestoreSnapshotRetainsDesiredState();
        testDeferredRestoreDisappearanceBecomesRelinkRequired();
        testMissingRelinkSuppressionAndTwelveLatents(juce::File(argv[1]));
#endif
        testEditorClosureDuringLoad(juce::File(argv[1]));
        testRepeatedReplacementRemainsBounded(juce::File(argv[1]), junkFile);
        junkFile.deleteFile();
    }

    require(rave_test::juceAssertionCount() == 0, "no JUCE assertions fired in the plugin test");
    std::cout << "PluginProcessor tests passed\n";
}
