#include "plugin/PluginEditor.h"
#include "ui/FactoryModels.h"
#include <RaveIconAssets.h>

#include "plugin/PluginProcessor.h"

#include <algorithm>

namespace
{
// juce::String(const char*) rejects non-ASCII literals, so strings containing
// typographic characters are built through an explicit UTF-8 pointer.
[[nodiscard]] juce::String utf8(const char *const text) { return juce::String(juce::CharPointer_UTF8(text)); }
} // namespace

RavePluginEditor::RavePluginEditor(RavePluginProcessor &owner) : AudioProcessorEditor(owner), ownerProcessor(owner)
{
    setLookAndFeel(&theme);
    setOpaque(true);
    icon = juce::Drawable::createFromImageData(RaveIconAssets::raveinstrumenticon_svg,
                                               RaveIconAssets::raveinstrumenticon_svgSize);
    if (icon != nullptr)
    {
        icon->setAccessible(true);
        icon->setTitle("RAVE Performance Instrument original icon");
        addAndMakeVisible(*icon);
    }

    title.setText("RAVE", juce::dontSendNotification);
    title.setJustificationType(juce::Justification::centredLeft);
    title.setFont(juce::Font(27.0f, juce::Font::bold));
    title.setAccessible(true);
    title.setTitle("RAVE Performance Instrument");
    addAndMakeVisible(title);

    status.setText(ownerProcessor.modelStatus(), juce::dontSendNotification);
    status.setJustificationType(juce::Justification::centredLeft);
    status.setFont(juce::Font(12.0f));
    status.setMinimumHorizontalScale(1.0f);
    status.setColour(juce::Label::textColourId, rave::ui::muted);
    status.setTitle("Model and audio status");
    addAndMakeVisible(status);

#if RAVE_HAS_LIBTORCH
    loadModelButton.setButtonText(utf8("Load model…"));
    loadModelButton.setEnabled(!ownerProcessor.isModelLoading());
    loadModelButton.setAccessible(true);
    loadModelButton.setTitle("Load TorchScript Model");
    loadModelButton.setTooltip("Choose a TorchScript model.");
    loadModelButton.onClick = [this]
    {
        const auto safe = juce::Component::SafePointer<RavePluginEditor>(this);
        rave::ui::showFactoryModels(
            loadModelButton,
            [safe](juce::File file)
            {
                if (safe == nullptr)
                    return;
                if (safe->ownerProcessor.isRelinkRequired())
                    static_cast<void>(safe->ownerProcessor.relinkMissingModel(file));
                else
                    static_cast<void>(safe->ownerProcessor.startModelLoad(file));
            },
            [safe]
            {
                if (safe != nullptr)
                    safe->chooseModel();
            });
    };
#else
    loadModelButton.setButtonText("LibTorch backend unavailable");
    loadModelButton.setEnabled(false);
#endif
    loadModelButton.setColour(juce::TextButton::buttonColourId, rave::ui::accent);
    loadModelButton.setColour(juce::TextButton::textColourOffId, rave::ui::background);
    addAndMakeVisible(loadModelButton);
    addAndMakeVisible(generator);
    generateAttachment =
        std::make_unique<juce::ButtonParameterAttachment>(ownerProcessor.generatorParameter(0), generator.enabled);
    depthAttachment =
        std::make_unique<juce::SliderParameterAttachment>(ownerProcessor.generatorParameter(1), generator.depth);
    rateAttachment =
        std::make_unique<juce::SliderParameterAttachment>(ownerProcessor.generatorParameter(2), generator.rate);
    midiModeButton.setClickingTogglesState(true);
    midiModeButton.setTitle("Show MIDI mappings");
    midiModeButton.setTooltip("Show or hide MIDI learn assignments.");
    midiModeButton.onClick = [this]
    {
        resized();
        repaint();
    };
    addAndMakeVisible(midiModeButton);

    dryWetSlider.setTextValueSuffix(" wet");
    dryWetAttachment = std::make_unique<juce::SliderParameterAttachment>(ownerProcessor.dryWetParameterReference(),
                                                                         dryWetSlider, nullptr);
    rave::ui::configureMix(dryWetSlider);
    addAndMakeVisible(dryWetSlider);

    dryWetLabel.setText("DRY / WET", juce::dontSendNotification);
    dryWetLabel.setJustificationType(juce::Justification::centred);
    dryWetLabel.setFont(juce::Font(11.0f, juce::Font::bold));
    dryWetLabel.setColour(juce::Label::textColourId, rave::ui::muted);
    addAndMakeVisible(dryWetLabel);

    dryWetMidiButton.onClick = [this] { ownerProcessor.beginMidiLearn(0); };
    addAndMakeVisible(dryWetMidiButton);
    updateMidiLearnLabels();

    latentViewport.setViewedComponent(&latentContent, false);
    latentViewport.setScrollBarsShown(true, false);
    addAndMakeVisible(latentViewport);
    observedModelRevision = ownerProcessor.modelRevision();
    rebuildLatentControls(ownerProcessor.latentDimensionCount());

    setResizable(true, true);
    setResizeLimits(640, 440, 1100, 820);
    setSize(680, 460);
    startTimerHz(10);
}

RavePluginEditor::~RavePluginEditor()
{
    stopTimer();
    fileChooser.reset();
    setLookAndFeel(nullptr);
}

void RavePluginEditor::paint(juce::Graphics &g)
{
    g.fillAll(rave::ui::background);
    const rave::ui::SurfaceLayout layout(getLocalBounds());
    rave::ui::drawPanel(g, layout.mix);
    rave::ui::drawPanel(g, layout.controls);
    rave::ui::drawCaption(g, "NEURAL AUDIO INSTRUMENT", layout.header.withTrimmedLeft(140));
    rave::ui::drawCaption(g, "LATENT SPACE", layout.controls.reduced(16).removeFromTop(18));
    if (latentSliders.empty())
        rave::ui::drawEmptyState(g, layout.controls.withTrimmedTop(116));
    g.setColour(rave::ui::border);
    g.drawHorizontalLine(layout.footer.getY(), 16.0f, static_cast<float>(getWidth() - 16));
}

void RavePluginEditor::resized()
{
    const rave::ui::SurfaceLayout layout(getLocalBounds());
    auto header = layout.header;
    if (icon != nullptr)
        icon->setTransformToFit(header.removeFromLeft(36).reduced(2).toFloat(), juce::RectanglePlacement::centred);
    header.removeFromLeft(8);
    title.setBounds(header.removeFromLeft(94));
    auto toolbar = layout.toolbar;
    loadModelButton.setBounds(toolbar.removeFromLeft(160));
    midiModeButton.setBounds(toolbar.removeFromRight(70));
    status.setBounds(layout.footer.reduced(0, 4));
    auto mix = layout.mix.reduced(12);
    dryWetLabel.setBounds(mix.removeFromTop(26));
    dryWetSlider.setBounds(mix.removeFromTop(146));
    dryWetMidiButton.setVisible(midiModeButton.getToggleState());
    dryWetMidiButton.setBounds(mix.removeFromTop(28));
    auto content = layout.controls.reduced(10).withTrimmedTop(18);
    generator.setBounds(content.removeFromTop(66));
    latentViewport.setBounds(content);
    layoutLatentControls();
}

void RavePluginEditor::chooseModel()
{
#if RAVE_HAS_LIBTORCH
    fileChooser = std::make_unique<juce::FileChooser>("Choose a TorchScript model", juce::File{}, "*.ts;*.pt;*.pth");
    const auto safeThis = juce::Component::SafePointer<RavePluginEditor>(this);
    fileChooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
                             [safeThis](const juce::FileChooser &chooser)
                             {
                                 if (safeThis == nullptr)
                                     return;

                                 const auto file = chooser.getResult();
                                 if (!file.existsAsFile())
                                     return;

                                 const auto relinking = safeThis->ownerProcessor.isRelinkRequired();
                                 const auto accepted = relinking ? safeThis->ownerProcessor.relinkMissingModel(file)
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
    const auto boundedDimensionCount = std::min(dimensionCount, RavePluginProcessor::maximumLatentCount);
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
        label->setJustificationType(juce::Justification::centred);
        label->setFont(juce::Font(12.0f));
        label->setColour(juce::Label::textColourId, rave::ui::muted);
        latentContent.addAndMakeVisible(*label);

        auto slider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
        slider->setRange(-4.0, 4.0, 0.01);
        slider->setValue(ownerProcessor.latentControl(index), juce::dontSendNotification);
        slider->setDoubleClickReturnValue(true, 0.0);
        auto *const sliderPointer = slider.get();
        if (index < RavePluginProcessor::macroCount)
        {
            latentAttachments.push_back(std::make_unique<juce::SliderParameterAttachment>(
                ownerProcessor.macroParameterReference(index), *slider, nullptr));
        }
        else
        {
            slider->onValueChange = [this, index, sliderPointer] {
                static_cast<void>(
                    ownerProcessor.setLatentControl(index, static_cast<float>(sliderPointer->getValue())));
            };
            latentAttachments.push_back(nullptr);
        }
        const auto name = "Latent " + juce::String(index + 1);
        slider->setSliderStyle(juce::Slider::LinearHorizontal);
        slider->setTextBoxStyle(juce::Slider::TextBoxRight, false, 42, 18);
        slider->setTitle(name);
        slider->setName(name);
        slider->setWantsKeyboardFocus(true);
        slider->setNumDecimalPlacesToDisplay(2);
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
    const bool mapping = midiModeButton.getToggleState();
    const int rowHeight = mapping ? 58 : 40;
    const auto contentWidth = std::max(1, latentViewport.getWidth() - 12);
    const int columns = std::max(1, contentWidth / 100);
    const int cellWidth = contentWidth / columns;
    const int rows = (static_cast<int>(latentSliders.size()) + columns - 1) / columns;
    latentContent.setSize(contentWidth, std::max(latentViewport.getHeight(), rows * rowHeight));
    for (std::size_t index = 0; index < latentSliders.size(); ++index)
    {
        const int x = (static_cast<int>(index) % columns) * cellWidth;
        const int y = (static_cast<int>(index) / columns) * rowHeight;
        latentLabels[index]->setBounds(x, y, cellWidth, 14);
        latentSliders[index]->setBounds(x + 4, y + 14, cellWidth - 8, 24);
        latentMidiButtons[index]->setVisible(mapping && index < RavePluginProcessor::macroCount);
        latentMidiButtons[index]->setBounds(x + 5, y + 38, cellWidth - 10, 18);
    }
    repaint();
}

void RavePluginEditor::updateMidiLearnLabels()
{
    const auto buttonText = [this](const std::size_t target)
    {
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
    loadModelButton.setButtonText(relinkRequired ? utf8("Relink model…") : utf8("Load model…"));
    loadModelButton.setTitle(relinkRequired ? "Relink missing model" : "Load TorchScript Model");
    loadModelButton.setTooltip(relinkRequired ? "Choose the replacement file for the missing saved model."
                                              : "Choose a TorchScript model.");
#endif
#if RAVE_HAS_LIBTORCH
    loadModelButton.setEnabled(!ownerProcessor.isModelLoading());
#else
    loadModelButton.setEnabled(false);
#endif
    status.setTooltip(ownerProcessor.modelStatus());
    status.setText(ownerProcessor.modelStatus(), juce::dontSendNotification);

    const auto revision = ownerProcessor.modelRevision();
    if (revision != observedModelRevision)
    {
        observedModelRevision = revision;
        rebuildLatentControls(ownerProcessor.latentDimensionCount());
    }
    updateMidiLearnLabels();
}
