#pragma once

// In-process component tests: no desktop automation or attached native window.
template <typename T> void collectUiComponents(juce::Component &parent, std::vector<T *> &result)
{
    for (auto *child : parent.getChildren())
    {
        if (auto *match = dynamic_cast<T *>(child))
            result.push_back(match);
        collectUiComponents(*child, result);
    }
}

void checkPerformanceEditor(RavePluginProcessor &processor)
{
    std::unique_ptr<juce::AudioProcessorEditor> editor(processor.createEditor());
    std::vector<juce::Drawable *> drawings;
    collectUiComponents(*editor, drawings);
    for (auto *drawing : drawings)
        if (drawing->getParentComponent() == editor.get())
            require(drawing->getDrawableBounds().transformedBy(drawing->getTransform()).getWidth() <= 36.1f,
                    "header SVG geometry scales into its icon slot");
    std::vector<juce::Slider *> sliders;
    std::vector<juce::TextButton *> buttons;
    std::vector<juce::Viewport *> viewports;
    collectUiComponents(*editor, sliders);
    collectUiComponents(*editor, buttons);
    collectUiComponents(*editor, viewports);
    const auto latentViewport = std::find_if(viewports.begin(), viewports.end(),
                                             [](auto *viewport) { return viewport->getViewedComponent() != nullptr; });
    require(latentViewport != viewports.end(), "latent controls have a viewport");
    const auto mixMatch =
        std::find_if(sliders.begin(), sliders.end(), [](auto *slider) { return slider->getName() == "Dry / wet mix"; });
    require(mixMatch != sliders.end(), "mix knob is present");
    auto *mix = *mixMatch;
    const auto oldMix = mix->getValue();
    mix->setValue(0.37, juce::sendNotificationSync);
    require(std::abs(processor.dryWetParameterReference().getValue() - 0.37f) < 0.001f,
            "custom mix knob still writes the host parameter");
    require(mix->getTextFromValue(0.37) == "37%" && std::abs(mix->getValueFromText("37%") - 0.37) < 0.001,
            "mix percent editing uses the correct parameter scale");
    mix->setValue(oldMix, juce::sendNotificationSync);
    for (auto *slider : sliders)
    {
        if (!slider->getName().startsWith("Latent "))
            continue;
        const auto index =
            static_cast<std::size_t>(slider->getName().fromFirstOccurrenceOf(" ", false, false).getIntValue() - 1);
        const auto previous = slider->getValue();
        slider->setValue(0.23, juce::sendNotificationSync);
        const auto actual = index < RavePluginProcessor::macroCount
                                ? processor.macroParameterReference(index).convertFrom0to1(
                                      processor.macroParameterReference(index).getValue())
                                : processor.latentControl(index);
        require(std::abs(actual - 0.23f) < 0.001f, "latent sliders still write macro and dynamic model controls");
        slider->setValue(previous, juce::sendNotificationSync);
    }
    auto mapping =
        std::find_if(buttons.begin(), buttons.end(), [](auto *button) { return button->getButtonText() == "MIDI"; });
    require(mapping != buttons.end(), "MIDI controls have a discoverable toggle");
    for (const bool showMappings : {false, true})
    {
        (*mapping)->setToggleState(showMappings, juce::dontSendNotification);
        (*mapping)->onClick();
        for (const auto size : {juce::Point<int>(640, 440), juce::Point<int>(680, 460), juce::Point<int>(1100, 820)})
        {
            editor->setSize(size.x, size.y);
            for (auto *slider : sliders)
            {
                if (slider->getTitle() == "Motion depth" || slider->getTitle() == "Motion rate (Hz)")
                    continue;
                if (slider->getName().startsWith("Latent "))
                {
                    require(slider->getSliderStyle() == juce::Slider::LinearHorizontal && slider->getHeight() >= 24,
                            "latent sliders remain editable at every supported size");
                    if (!showMappings && processor.latentDimensionCount() <= 16 && size.y >= 460)
                        require((*latentViewport)
                                    ->getLocalBounds()
                                    .contains((*latentViewport)->getLocalArea(slider, slider->getLocalBounds())),
                                "all loaded latent controls fit without scrolling at the default window size");
                }
                else
                    require(slider->getSliderStyle() == juce::Slider::RotaryHorizontalVerticalDrag &&
                                slider->getWidth() >= 76 && slider->getHeight() >= 90,
                            "mix knob and editable value remain usable at every supported size");
                require(slider->getParentComponent()->getLocalBounds().contains(slider->getBounds()),
                        "controls remain inside their parent content at every size");
            }
            for (auto *button : buttons)
                if (button->isVisible())
                    require(button->getParentComponent()->getLocalBounds().contains(button->getBounds()),
                            "visible buttons remain reachable at minimum and maximum sizes");
        }
    }
    (*mapping)->setToggleState(false, juce::dontSendNotification);
    (*mapping)->onClick();
    editor->setSize(680, 460);
    const auto previewDirectory = juce::SystemStats::getEnvironmentVariable("RAVE_UI_PREVIEW_DIR", {});
    if (previewDirectory.isNotEmpty())
    {
        const juce::File directory{juce::String(previewDirectory)};
        require(directory.createDirectory().wasOk(), "create UI preview directory");
        const auto image = editor->createComponentSnapshot(editor->getLocalBounds(), true, 1.0f);
        const auto file =
            directory.getChildFile("plugin-" + juce::String(processor.latentDimensionCount()) + "-latents.png");
        juce::FileOutputStream output(file);
        require(output.openedOk() && output.setPosition(0) && output.truncate().wasOk() &&
                    juce::PNGImageFormat().writeImageToStream(image, output),
                "write in-process UI render");
    }
}
