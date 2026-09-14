#include "plugin/PluginEditor.h"

#include "plugin/PluginProcessor.h"

#include <algorithm>

namespace
{
// juce::String(const char*) rejects non-ASCII literals, so strings containing
// typographic characters are built through an explicit UTF-8 pointer.
[[nodiscard]] juce::String utf8(const char* const text)
{
    return juce::String(juce::CharPointer_UTF8(text));
}
} // namespace

RavePluginEditor::RavePluginEditor(RavePluginProcessor& owner)
    : AudioProcessorEditor(owner), ownerProcessor(owner)
{
    title.setText("RAVE Performance Instrument", juce::dontSendNotification);
    title.setJustificationType(juce::Justification::centred);
    title.setFont(juce::Font(20.0f, juce::Font::bold));
    addAndMakeVisible(title);

    status.setText(ownerProcessor.modelStatus(), juce::dontSendNotification);
    status.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(status);

#if RAVE_HAS_LIBTORCH
    loadModelButton.setButtonText(utf8("Load TorchScript Model…"));
    loadModelButton.setEnabled(!ownerProcessor.isModelLoading());
    loadModelButton.setAccessible(true);
    loadModelButton.setTitle("Load TorchScript Model");
    loadModelButton.setTooltip("Choose a TorchScript model.");
    loadModelButton.onClick = [this] { chooseModel(); };
#else
    loadModelButton.setButtonText("LibTorch backend unavailable");
    loadModelButton.setEnabled(false);
#endif
    addAndMakeVisible(loadModelButton);

    dryWetSlider.setTextValueSuffix(" wet");
    dryWetAttachment = std::make_unique<juce::SliderParameterAttachment>(
        ownerProcessor.dryWetParameterReference(), dryWetSlider, nullptr);
    addAndMakeVisible(dryWetSlider);

    dryWetLabel.setText("Dry / Wet", juce::dontSendNotification);
    dryWetLabel.attachToComponent(&dryWetSlider, true);
    addAndMakeVisible(dryWetLabel);

    dryWetMidiButton.onClick = [this] { ownerProcessor.beginMidiLearn(0); };
    addAndMakeVisible(dryWetMidiButton);
    updateMidiLearnLabels();

    latentViewport.setViewedComponent(&latentContent, false);
    latentViewport.setScrollBarsShown(true, false);
    addAndMakeVisible(latentViewport);
    observedModelRevision = ownerProcessor.modelRevision();
    rebuildLatentControls(ownerProcessor.latentDimensionCount());

    setSize(620, 480);
    startTimerHz(10);
}

RavePluginEditor::~RavePluginEditor()
{
    stopTimer();
    fileChooser.reset();
}

void RavePluginEditor::resized()
{
    auto area = getLocalBounds().reduced(24);
    title.setBounds(area.removeFromTop(40));
    status.setBounds(area.removeFromTop(32));
    area.removeFromTop(16);
    loadModelButton.setBounds(area.removeFromTop(38).reduced(80, 0));
    area.removeFromTop(20);
    auto dryWetArea = area.removeFromTop(40).reduced(72, 0);
    dryWetMidiButton.setBounds(dryWetArea.removeFromRight(82));
    dryWetArea.removeFromRight(8);
    dryWetSlider.setBounds(dryWetArea);
    area.removeFromTop(16);
    latentViewport.setBounds(area);
    layoutLatentControls();
}

void RavePluginEditor::chooseModel()
{
#if RAVE_HAS_LIBTORCH
    fileChooser = std::make_unique<juce::FileChooser>(
        "Choose a TorchScript model", juce::File {}, "*.ts;*.pt;*.pth");
    const auto safeThis = juce::Component::SafePointer<RavePluginEditor>(this);
    fileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                 | juce::FileBrowserComponent::canSelectFiles,
                             [safeThis](const juce::FileChooser& chooser) {
                                 if (safeThis == nullptr)
                                     return;

                                 const auto file = chooser.getResult();
                                 if (!file.existsAsFile())
                                     return;

                                 const auto relinking = safeThis->ownerProcessor.isRelinkRequired();
                                 const auto accepted = relinking
                                     ? safeThis->ownerProcessor.relinkMissingModel(file)
                                     : safeThis->ownerProcessor.startModelLoad(file);
                                 if (accepted)
                                 {
                                     safeThis->loadModelButton.setEnabled(false);
                                     safeThis->status.setText(safeThis->ownerProcessor.modelStatus(),
                                                              juce::dontSendNotification);
                                 }
                             });
#endif
}

void RavePluginEditor::rebuildLatentControls(const std::size_t dimensionCount)
{
    const auto boundedDimensionCount = std::min(
        dimensionCount, RavePluginProcessor::maximumLatentCount);
    latentAttachments.clear();
    latentMidiButtons.clear();
    latentSliders.clear();
    latentLabels.clear();
    latentAttachments.reserve(boundedDimensionCount);
    latentMidiButtons.reserve(boundedDimensionCount);
    latentSliders.reserve(boundedDimensionCount);
    latentLabels.reserve(boundedDimensionCount);

    for (std::size_t index = 0; index < boundedDimensionCount; ++index)
    {
        auto label = std::make_unique<juce::Label>();
        label->setText("Latent " + juce::String(index + 1), juce::dontSendNotification);
        label->setJustificationType(juce::Justification::centredLeft);
        latentContent.addAndMakeVisible(*label);

        auto slider = std::make_unique<juce::Slider>(
            juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
        slider->setRange(-4.0, 4.0, 0.01);
        slider->setValue(ownerProcessor.latentControl(index), juce::dontSendNotification);
        slider->setDoubleClickReturnValue(true, 0.0);
        auto* const sliderPointer = slider.get();
        if (index < RavePluginProcessor::macroCount)
        {
            latentAttachments.push_back(std::make_unique<juce::SliderParameterAttachment>(
                ownerProcessor.macroParameterReference(index), *slider, nullptr));
        }
        else
        {
            slider->onValueChange = [this, index, sliderPointer] {
                static_cast<void>(ownerProcessor.setLatentControl(
                    index, static_cast<float>(sliderPointer->getValue())));
            };
            latentAttachments.push_back(nullptr);
        }
        latentContent.addAndMakeVisible(*slider);

        auto midiButton = std::make_unique<juce::TextButton>();
        if (index < RavePluginProcessor::macroCount)
        {
            midiButton->onClick = [this, index] { ownerProcessor.beginMidiLearn(index + 1); };
            latentContent.addAndMakeVisible(*midiButton);
        }

        latentLabels.push_back(std::move(label));
        latentSliders.push_back(std::move(slider));
        latentMidiButtons.push_back(std::move(midiButton));
    }

    layoutLatentControls();
    updateMidiLearnLabels();
}

void RavePluginEditor::layoutLatentControls()
{
    constexpr int rowHeight = 42;
    const auto contentWidth = std::max(320, latentViewport.getWidth() - 12);
    const auto contentHeight = std::max(latentViewport.getHeight(),
                                        static_cast<int>(latentSliders.size()) * rowHeight);
    latentContent.setSize(contentWidth, contentHeight);

    for (std::size_t index = 0; index < latentSliders.size(); ++index)
    {
        const auto y = static_cast<int>(index) * rowHeight;
        latentLabels[index]->setBounds(0, y, 88, rowHeight);
        if (index < RavePluginProcessor::macroCount)
        {
            latentSliders[index]->setBounds(92, y + 3, contentWidth - 190, rowHeight - 6);
            latentMidiButtons[index]->setBounds(contentWidth - 86, y + 6, 82, rowHeight - 12);
        }
        else
        {
            latentSliders[index]->setBounds(92, y + 3, contentWidth - 100, rowHeight - 6);
        }
    }
}

void RavePluginEditor::updateMidiLearnLabels()
{
    const auto buttonText = [this](const std::size_t target) {
        if (ownerProcessor.isLearningMidiTarget(target))
            return utf8("Move CC…");
        const auto controller = ownerProcessor.midiControllerForTarget(target);
        return controller >= 0 ? "CC " + juce::String(controller) : juce::String("MIDI Learn");
    };

    dryWetMidiButton.setButtonText(buttonText(0));
    const auto buttonCount = std::min(latentMidiButtons.size(), RavePluginProcessor::macroCount);
    for (std::size_t index = 0; index < buttonCount; ++index)
        latentMidiButtons[index]->setButtonText(buttonText(index + 1));
}

void RavePluginEditor::timerCallback()
{
#if RAVE_HAS_LIBTORCH
    const auto relinkRequired = ownerProcessor.isRelinkRequired();
    loadModelButton.setButtonText(relinkRequired ? "Relink Missing Model…"
                                                : utf8("Load TorchScript Model…"));
    loadModelButton.setTitle(relinkRequired ? "Relink missing model" : "Load TorchScript Model");
    loadModelButton.setTooltip(relinkRequired
        ? "Choose the replacement file for the missing saved model."
        : "Choose a TorchScript model.");
#endif
    loadModelButton.setEnabled(!ownerProcessor.isModelLoading());
    status.setText(ownerProcessor.modelStatus(), juce::dontSendNotification);

    const auto revision = ownerProcessor.modelRevision();
    if (revision != observedModelRevision)
    {
        observedModelRevision = revision;
        rebuildLatentControls(ownerProcessor.latentDimensionCount());
    }
    updateMidiLearnLabels();
}
