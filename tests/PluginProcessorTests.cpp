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
} // namespace

int main(const int argc, const char* const* argv)
{
    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    testFactoryAudioAndState();
    if (argc == 2)
        testModelAndLatentRecall(juce::File(argv[1]));
    std::cout << "PluginProcessor tests passed\n";
}
