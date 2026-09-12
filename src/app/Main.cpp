#include <JuceHeader.h>

#include "engine/RaveAudioEngine.h"

#if RAVE_HAS_LIBTORCH
#include "model/BackgroundModelLoader.h"
#include "model/TorchScriptBackend.h"
#endif

namespace
{
class MainComponent final : public juce::Component,
                            private juce::Timer
{
public:
    MainComponent()
    {
        title.setText("RAVE Performance Instrument", juce::dontSendNotification);
        title.setJustificationType(juce::Justification::centred);
        title.setFont(juce::Font(24.0f, juce::Font::bold));
        addAndMakeVisible(title);

        status.setText("Audio pass-through ready — no model loaded", juce::dontSendNotification);
        status.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(status);

        dryWet.setRange(0.0, 1.0, 0.01);
        dryWet.setValue(0.0);
        dryWet.setTextValueSuffix(" wet");
        dryWet.onValueChange = [this] { engine.setDryWet(static_cast<float>(dryWet.getValue())); };
        addAndMakeVisible(dryWet);

        dryWetLabel.setText("Dry / Wet", juce::dontSendNotification);
        dryWetLabel.attachToComponent(&dryWet, true);
        addAndMakeVisible(dryWetLabel);

        latentViewport.setViewedComponent(&latentContent, false);
        latentViewport.setScrollBarsShown(true, false);
        addAndMakeVisible(latentViewport);

#if RAVE_HAS_LIBTORCH
        modelLoader = std::make_unique<rave::BackgroundModelLoader>([] {
            return std::make_shared<rave::TorchScriptBackend>();
        });
        loadModelButton.setButtonText("Load TorchScript Model…");
        loadModelButton.onClick = [this] { chooseModel(); };
        addAndMakeVisible(loadModelButton);
        startTimerHz(10);
#else
        loadModelButton.setButtonText("LibTorch backend unavailable");
        loadModelButton.setEnabled(false);
        addAndMakeVisible(loadModelButton);
#endif

        const auto error = deviceManager.initialiseWithDefaultDevices(2, 2);
        if (error.isNotEmpty())
            status.setText("Audio device error: " + error, juce::dontSendNotification);
        else
            deviceManager.addAudioCallback(&engine);

        setSize(720, 420);
    }

    ~MainComponent() override
    {
        stopTimer();
        fileChooser.reset();
        deviceManager.removeAudioCallback(&engine);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(32);
        title.setBounds(area.removeFromTop(52));
        status.setBounds(area.removeFromTop(36));
        area.removeFromTop(28);
        loadModelButton.setBounds(area.removeFromTop(40).reduced(140, 0));
        area.removeFromTop(20);
        dryWet.setBounds(area.removeFromTop(44).reduced(100, 0));
        area.removeFromTop(16);
        latentViewport.setBounds(area);
        layoutLatentControls();
    }

private:
    void rebuildLatentControls(const std::size_t dimensionCount)
    {
        latentSliders.clear();
        latentLabels.clear();
        latentSliders.reserve(dimensionCount);
        latentLabels.reserve(dimensionCount);

        for (std::size_t index = 0; index < dimensionCount; ++index)
        {
            auto label = std::make_unique<juce::Label>();
            label->setText("Latent " + juce::String(index + 1), juce::dontSendNotification);
            label->setJustificationType(juce::Justification::centredLeft);
            latentContent.addAndMakeVisible(*label);

            auto slider = std::make_unique<juce::Slider>(
                juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight);
            slider->setRange(-4.0, 4.0, 0.01);
            slider->setValue(0.0, juce::dontSendNotification);
            slider->setDoubleClickReturnValue(true, 0.0);
            auto* const sliderPointer = slider.get();
            slider->onValueChange = [this, index, sliderPointer] {
                static_cast<void>(engine.setLatentControl(
                    index, static_cast<float>(sliderPointer->getValue())));
            };
            latentContent.addAndMakeVisible(*slider);

            latentLabels.push_back(std::move(label));
            latentSliders.push_back(std::move(slider));
        }

        layoutLatentControls();
    }

    void layoutLatentControls()
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
            latentSliders[index]->setBounds(92, y + 3, contentWidth - 100, rowHeight - 6);
        }
    }

#if RAVE_HAS_LIBTORCH
    void chooseModel()
    {
        fileChooser = std::make_unique<juce::FileChooser>(
            "Choose a TorchScript model", juce::File {}, "*.ts;*.pt;*.pth");
        const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
        fileChooser->launchAsync(juce::FileBrowserComponent::openMode
                                     | juce::FileBrowserComponent::canSelectFiles,
                                 [safeThis](const juce::FileChooser& chooser) {
                                     if (safeThis == nullptr)
                                         return;

                                     const auto file = chooser.getResult();
                                     if (!file.existsAsFile())
                                         return;

                                     if (safeThis->modelLoader->start(
                                             file.getFullPathName().toStdString(),
                                             safeThis->engine.runtimeConfiguration()))
                                     {
                                         safeThis->loadModelButton.setEnabled(false);
                                         safeThis->status.setText("Loading " + file.getFileName() + "…",
                                                                  juce::dontSendNotification);
                                     }
                                 });
    }

    void timerCallback() override
    {
        const auto loaderState = modelLoader->state();
        if (loaderState != rave::BackgroundModelLoader::State::succeeded
            && loaderState != rave::BackgroundModelLoader::State::failed)
            return;

        auto result = modelLoader->takeResult();
        loadModelButton.setEnabled(true);
        if (result.state == rave::BackgroundModelLoader::State::failed)
        {
            // Qualification failed, so the previous active model was never
            // replaced and remains playable.
            status.setText(juce::String("Model load failed: ") + juce::String(result.errorMessage)
                               + (engine.hasModelBackend() ? " — previous model still active" : ""),
                           juce::dontSendNotification);
            return;
        }

        // Detaching the audio callback before activation keeps every join and
        // preparation step off the audio thread.
        deviceManager.removeAudioCallback(&engine);
        juce::String activationError;
        try
        {
            if (!engine.activateModelBackend(std::move(result.backend)))
                activationError = "candidate was not accepted";
        }
        catch (const std::exception& exception)
        {
            activationError = exception.what();
        }
        catch (...)
        {
            activationError = "unknown activation error";
        }
        deviceManager.addAudioCallback(&engine);

        if (activationError.isNotEmpty())
        {
            status.setText("Model activation failed: " + activationError
                               + (engine.hasModelBackend()
                                      ? " — previous model still active"
                                      : ""),
                           juce::dontSendNotification);
            return;
        }

        rebuildLatentControls(engine.latentDimensionCount());
        const auto latentCount = engine.latentDimensionCount();
        status.setText(
            latentCount > 0
                ? "Model active — " + juce::String(latentCount) + " latent dimensions"
                : "Model active — no compatible latent metadata",
            juce::dontSendNotification);
    }
#else
    void timerCallback() override {}
#endif

    juce::AudioDeviceManager deviceManager;
    rave::RaveAudioEngine engine;
    juce::Label title;
    juce::Label status;
    juce::TextButton loadModelButton;
    juce::Label dryWetLabel;
    juce::Slider dryWet { juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight };
    juce::Component latentContent;
    juce::Viewport latentViewport;
    std::vector<std::unique_ptr<juce::Label>> latentLabels;
    std::vector<std::unique_ptr<juce::Slider>> latentSliders;
    std::unique_ptr<juce::FileChooser> fileChooser;
#if RAVE_HAS_LIBTORCH
    std::unique_ptr<rave::BackgroundModelLoader> modelLoader;
#endif
};

class MainWindow final : public juce::DocumentWindow
{
public:
    explicit MainWindow(const juce::String& name)
        : DocumentWindow(name,
                         juce::Desktop::getInstance().getDefaultLookAndFeel()
                             .findColour(juce::ResizableWindow::backgroundColourId),
                         DocumentWindow::allButtons)
    {
        setUsingNativeTitleBar(true);
        setContentOwned(new MainComponent(), true);
        setResizable(true, true);
        centreWithSize(getWidth(), getHeight());
        setVisible(true);
    }

    void closeButtonPressed() override
    {
        juce::JUCEApplication::getInstance()->systemRequestedQuit();
    }
};

class RaveApplication final : public juce::JUCEApplication
{
public:
    [[nodiscard]] const juce::String getApplicationName() override
    {
        return "RAVE Performance Instrument";
    }

    [[nodiscard]] const juce::String getApplicationVersion() override
    {
        return "0.1.0";
    }

    void initialise(const juce::String&) override
    {
        window = std::make_unique<MainWindow>(getApplicationName());
    }

    void shutdown() override
    {
        window.reset();
    }

private:
    std::unique_ptr<MainWindow> window;
};
} // namespace

START_JUCE_APPLICATION(RaveApplication)
