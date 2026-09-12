#include "plugin/PluginProcessor.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter();

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
    require(std::equal(left.begin(), left.end(), buffer.getReadPointer(0)), "dry left pass-through");
    require(std::equal(right.begin(), right.end(), buffer.getReadPointer(1)), "dry right pass-through");

    auto* dryWet = findParameter(*processor, "dryWet");
    require(dryWet != nullptr, "dry/wet parameter exposed");
    require(processor->getParameters().size() == 9, "stable dry/wet plus eight macro parameters");
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
    midi.addEvent(juce::MidiMessage::controllerEvent(1, 74, 127), 0);
    processor->processBlock(buffer, midi);
    auto* macro1 = findParameter(*processor, "macro1");
    require(macro1 != nullptr, "first macro parameter exposed");
    require(std::abs(macro1->convertFrom0to1(macro1->getValue()) - 4.0f) < 0.001f,
            "learned MIDI CC controls macro");

    state.reset();
    processor->getStateInformation(state);
    restored->setStateInformation(state.getData(), static_cast<int>(state.getSize()));
    restored->prepareToPlay(48000.0, 4);
    juce::MidiBuffer restoredMidi;
    restoredMidi.addEvent(juce::MidiMessage::controllerEvent(1, 74, 0), 0);
    restored->processBlock(buffer, restoredMidi);
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
    require(original->setLatentControl(0, 1.25f), "first latent set");
    require(original->setLatentControl(1, -2.5f), "second latent set");

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
    require(status.containsIgnoreCase("previous model still active"),
            "failure status explains retention");

    juce::AudioBuffer<float> buffer(2, 8);
    for (int channel = 0; channel < buffer.getNumChannels(); ++channel)
        for (int sample = 0; sample < buffer.getNumSamples(); ++sample)
            buffer.setSample(channel, sample, 0.25f * static_cast<float>(sample + 1));
    juce::MidiBuffer midi;
    processor->processBlock(buffer, midi);
    requireFiniteOutput(buffer);

    processor->releaseResources();
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
} // namespace

int main(const int argc, const char* const* argv)
{
    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    testFactoryAudioAndState();
    if (argc == 2)
    {
        testModelAndLatentRecall(juce::File(argv[1]));

        auto junkFile = juce::File::createTempFile("rave-invalid-model");
        junkFile.replaceWithText("this is not a TorchScript model");
        testFailedLoadKeepsPreviousModel(juce::File(argv[1]), junkFile);
        testSampleRateMismatchIsRejectedBeforeActivation(juce::File(argv[1]));
        testEditorClosureDuringLoad(juce::File(argv[1]));
        testRepeatedReplacementRemainsBounded(juce::File(argv[1]), junkFile);
        junkFile.deleteFile();
    }
    std::cout << "PluginProcessor tests passed\n";
}
