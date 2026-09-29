#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

namespace rave::ui
{
class GeneratorControls : public juce::Component
{
public:
    GeneratorControls()
    {
        enabled.setButtonText("Generate without audio input");
        enabled.setTooltip(
            "Drive the model decoder directly. Raise Dry / Wet to hear it. Off returns to audio-input processing.");
        addAndMakeVisible(enabled);
        configure(depth, "Motion depth", 0.0, 2.0, 0.5);
        configure(rate, "Motion rate (Hz)", 0.01, 2.0, 0.1);
    }
    void resized() override
    {
        auto area = getLocalBounds();
        enabled.setBounds(area.removeFromTop(22));
        depth.setBounds(area.removeFromTop(22));
        rate.setBounds(area.removeFromTop(22));
    }
    juce::ToggleButton enabled;
    juce::Slider depth, rate;

private:
    void configure(juce::Slider &slider, const juce::String &name, double minimum, double maximum, double initial)
    {
        slider.setSliderStyle(juce::Slider::LinearHorizontal);
        slider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 120, 22);
        slider.setRange(minimum, maximum, 0.01);
        slider.setTextValueSuffix(name == "Motion depth" ? " depth" : " Hz");
        slider.setTitle(name);
        slider.setTooltip(name);
        slider.setValue(initial, juce::dontSendNotification);
        addAndMakeVisible(slider);
    }
};
} // namespace rave::ui
