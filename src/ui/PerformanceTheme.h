#pragma once

#include <juce_gui_extra/juce_gui_extra.h>

namespace rave::ui
{
inline const juce::Colour background{0xff101116};
inline const juce::Colour panel{0xff191b23};
inline const juce::Colour raised{0xff242733};
inline const juce::Colour border{0xff303442};
inline const juce::Colour ink{0xffeeedf5};
inline const juce::Colour muted{0xffa3a7ba};
inline const juce::Colour accent{0xffb4a0ff};

// Device selectors can change height when drivers or advanced options change.
// Keep their complete content reachable through the containing viewport.
class SettingsContent final : public juce::Component
{
public:
    void childBoundsChanged(juce::Component *) override
    {
        int bottom = 0;
        for (auto *child : getChildren())
            bottom = juce::jmax(bottom, child->getBottom());
        if (getHeight() != bottom + 12)
            setSize(getWidth(), bottom + 12);
    }
};

class PerformanceTheme final : public juce::LookAndFeel_V4
{
public:
    PerformanceTheme()
    {
        setColour(juce::ResizableWindow::backgroundColourId, background);
        setColour(juce::Label::textColourId, ink);
        setColour(juce::TextButton::buttonColourId, raised);
        setColour(juce::TextButton::buttonOnColourId, accent);
        setColour(juce::TextButton::textColourOffId, ink);
        setColour(juce::TextButton::textColourOnId, background);
        setColour(juce::Slider::textBoxTextColourId, ink);
        setColour(juce::Slider::textBoxBackgroundColourId, juce::Colours::transparentBlack);
        setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour(juce::Slider::rotarySliderFillColourId, accent);
        setColour(juce::Slider::trackColourId, accent);
        setColour(juce::Slider::thumbColourId, accent);
        setColour(juce::ComboBox::backgroundColourId, raised);
        setColour(juce::ComboBox::outlineColourId, border);
        setColour(juce::ComboBox::textColourId, ink);
        setColour(juce::ComboBox::arrowColourId, muted);
        setColour(juce::PopupMenu::backgroundColourId, panel);
        setColour(juce::PopupMenu::textColourId, ink);
        setColour(juce::PopupMenu::highlightedBackgroundColourId, raised);
        setColour(juce::PopupMenu::highlightedTextColourId, accent);
        setColour(juce::TextEditor::backgroundColourId, raised);
        setColour(juce::TextEditor::textColourId, ink);
        setColour(juce::TextEditor::highlightColourId, accent.withAlpha(0.3f));
        setColour(juce::TextEditor::outlineColourId, border);
        setColour(juce::TextEditor::focusedOutlineColourId, accent);
        setColour(juce::ScrollBar::thumbColourId, border.brighter(0.3f));
        setColour(juce::ListBox::backgroundColourId, panel);
        setColour(juce::ListBox::textColourId, ink);
        setColour(juce::ToggleButton::textColourId, ink);
        setColour(juce::ToggleButton::tickColourId, accent);
    }

    juce::Font getTextButtonFont(juce::TextButton &, int height) override
    {
        return juce::Font(juce::jmin(13.0f, height * 0.48f), juce::Font::bold);
    }

    juce::Label *createSliderTextBox(juce::Slider &slider) override
    {
        auto *label = juce::LookAndFeel_V4::createSliderTextBox(slider);
        label->setColour(juce::Label::outlineColourId, juce::Colours::transparentBlack);
        label->setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
        label->setColour(juce::Label::textColourId, ink);
        label->setFont(juce::Font(13.0f));
        return label;
    }

    void drawButtonBackground(juce::Graphics &g, juce::Button &button, const juce::Colour &colour, bool over,
                              bool down) override
    {
        auto bounds = button.getLocalBounds().toFloat().reduced(0.5f);
        auto fill = colour;
        if (over || button.hasKeyboardFocus(true))
            fill = fill.brighter(0.12f);
        if (down)
            fill = fill.darker(0.15f);
        g.setColour(fill.withMultipliedAlpha(button.isEnabled() ? 1.0f : 0.35f));
        g.fillRoundedRectangle(bounds, 7.0f);
        g.setColour(button.hasKeyboardFocus(true) ? accent : border);
        g.drawRoundedRectangle(bounds, 7.0f, 1.0f);
    }

    void drawRotarySlider(juce::Graphics &g, int x, int y, int width, int height, float position, float start,
                          float end, juce::Slider &slider) override
    {
        auto box = juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y), static_cast<float>(width),
                                          static_cast<float>(height))
                       .reduced(7.0f);
        const auto radius = juce::jmin(box.getWidth(), box.getHeight()) * 0.5f;
        const auto centre = box.getCentre();
        const auto angle = start + position * (end - start);
        const auto arcRadius = radius - 2.0f;
        const auto stroke = juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded);
        juce::Path track;
        track.addCentredArc(centre.x, centre.y, arcRadius, arcRadius, 0.0f, start, end, true);
        g.setColour(border);
        g.strokePath(track, stroke);
        const auto origin = slider.getMinimum() < 0.0 ? (start + end) * 0.5f : start;
        juce::Path value;
        value.addCentredArc(centre.x, centre.y, arcRadius, arcRadius, 0.0f, juce::jmin(origin, angle),
                            juce::jmax(origin, angle), true);
        g.setColour(accent.withAlpha(slider.isEnabled() ? 1.0f : 0.35f));
        g.strokePath(value, stroke);
        auto dial = juce::Rectangle<float>(radius * 1.45f, radius * 1.45f).withCentre(centre);
        g.setGradientFill(
            juce::ColourGradient(raised.brighter(0.08f), dial.getTopLeft(), panel, dial.getBottomRight(), false));
        g.fillEllipse(dial);
        g.setColour(slider.hasKeyboardFocus(true) ? accent : border);
        g.drawEllipse(dial, 1.0f);
        juce::Path pointer;
        pointer.addRoundedRectangle(-1.5f, -radius * 0.60f, 3.0f, radius * 0.24f, 1.5f);
        g.setColour(ink);
        g.fillPath(pointer, juce::AffineTransform::rotation(angle).translated(centre.x, centre.y));
    }
};

inline void configureKnob(juce::Slider &slider, const juce::String &name)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 76, 22);
    slider.setRotaryParameters(juce::MathConstants<float>::pi * 1.25f, juce::MathConstants<float>::pi * 2.75f, true);
    slider.setTitle(name);
    slider.setName(name);
    slider.setWantsKeyboardFocus(true);
    slider.setNumDecimalPlacesToDisplay(2);
    slider.setColour(juce::Slider::textBoxHighlightColourId, accent.withAlpha(0.25f));
}

inline void configureMix(juce::Slider &slider)
{
    configureKnob(slider, "Dry / wet mix");
    slider.setTextValueSuffix({});
    slider.textFromValueFunction = [](double value) { return juce::String(juce::roundToInt(value * 100.0)) + "%"; };
    slider.valueFromTextFunction = [](const juce::String &text) { return text.getDoubleValue() / 100.0; };
    slider.setDoubleClickReturnValue(true, 1.0);
    slider.updateText();
}

inline void drawPanel(juce::Graphics &g, juce::Rectangle<int> bounds)
{
    g.setColour(panel);
    g.fillRoundedRectangle(bounds.toFloat(), 12.0f);
    g.setColour(border.withAlpha(0.65f));
    g.drawRoundedRectangle(bounds.toFloat().reduced(0.5f), 12.0f, 1.0f);
}

inline void drawCaption(juce::Graphics &g, const juce::String &text, juce::Rectangle<int> bounds)
{
    g.setColour(muted);
    g.setFont(juce::Font(11.0f, juce::Font::bold));
    g.drawText(text, bounds, juce::Justification::centredLeft);
}

inline void drawEmptyState(juce::Graphics &g, juce::Rectangle<int> bounds)
{
    const auto centre = bounds.getCentre().toFloat().translated(0.0f, -25.0f);
    g.setColour(accent.withAlpha(0.16f));
    g.fillEllipse(centre.x - 27.0f, centre.y - 27.0f, 54.0f, 54.0f);
    g.setColour(accent);
    for (int i = 0; i < 3; ++i)
    {
        auto ellipse = juce::Rectangle<float>(34.0f, 13.0f).withCentre(centre);
        juce::Path orbit;
        orbit.addEllipse(ellipse);
        g.strokePath(orbit, juce::PathStrokeType(1.1f),
                     juce::AffineTransform::rotation(i * juce::MathConstants<float>::pi / 3.0f, centre.x, centre.y));
    }
    g.setColour(ink);
    g.setFont(juce::Font(16.0f, juce::Font::bold));
    g.drawText("Shape the latent space", bounds.withY(static_cast<int>(centre.y) + 36).withHeight(24),
               juce::Justification::centred);
    g.setColour(muted);
    g.setFont(13.0f);
    g.drawText("Load a model with latent controls to begin.",
               bounds.withY(static_cast<int>(centre.y) + 62).withHeight(22), juce::Justification::centred);
}

struct SurfaceLayout
{
    explicit SurfaceLayout(juce::Rectangle<int> bounds)
    {
        auto area = bounds.reduced(16);
        header = area.removeFromTop(42);
        area.removeFromTop(12);
        toolbar = area.removeFromTop(32);
        area.removeFromTop(12);
        footer = area.removeFromBottom(48);
        area.removeFromBottom(10);
        mix = area.removeFromLeft(138);
        area.removeFromLeft(10);
        controls = area;
    }
    juce::Rectangle<int> header, toolbar, mix, controls, footer;
};
} // namespace rave::ui
