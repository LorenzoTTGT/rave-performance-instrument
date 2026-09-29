#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_gui_extra/juce_gui_extra.h>

#include "ui/GeneratorControls.h"
#include "ui/PerformanceTheme.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class RavePluginProcessor;

class RavePluginEditor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit RavePluginEditor(RavePluginProcessor &processor);
    ~RavePluginEditor() override;

    void paint(juce::Graphics &) override;
    void resized() override;

private:
    void chooseModel();
    void rebuildLatentControls(std::size_t dimensionCount);
    void layoutLatentControls();
    void updateMidiLearnLabels();
    void timerCallback() override;

    rave::ui::PerformanceTheme theme;
    RavePluginProcessor &ownerProcessor;
    std::unique_ptr<juce::Drawable> icon;
    rave::ui::GeneratorControls generator;
    std::unique_ptr<juce::ButtonParameterAttachment> generateAttachment;
    std::unique_ptr<juce::SliderParameterAttachment> depthAttachment, rateAttachment;
    juce::Label title;
    juce::Label status;
    juce::TextButton loadModelButton;
    juce::TextButton midiModeButton{"MIDI"};
    juce::Label dryWetLabel;
    juce::TextButton dryWetMidiButton;
    juce::Slider dryWetSlider{juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight};
    juce::Component latentContent;
    juce::Viewport latentViewport;
    std::vector<std::unique_ptr<juce::Label>> latentLabels;
    std::vector<std::unique_ptr<juce::Slider>> latentSliders;
    std::vector<std::unique_ptr<juce::TextButton>> latentMidiButtons;
    std::vector<std::unique_ptr<juce::SliderParameterAttachment>> latentAttachments;
    std::unique_ptr<juce::SliderParameterAttachment> dryWetAttachment;
    juce::TooltipWindow tooltips{this, 650};
    std::unique_ptr<juce::FileChooser> fileChooser;
    std::uint64_t observedModelRevision = 0;
};
