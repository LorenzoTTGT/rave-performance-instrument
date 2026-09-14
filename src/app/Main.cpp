#include <JuceHeader.h>
#include <RaveIconAssets.h>

#include "engine/LifecycleStatusText.h"
#include "engine/RaveAudioEngine.h"
#include "app/StandalonePresetModelCoordinator.h"
#include "state/LatestRequestGeneration.h"
#include "state/StandaloneSessionState.h"

#if RAVE_HAS_LIBTORCH
#include "model/BackgroundModelLoader.h"
#include "model/TorchScriptBackend.h"
#endif

#include <algorithm>
#include <optional>

namespace
{
[[nodiscard]] juce::String utf8(const char* const text)
{
    return juce::String(juce::CharPointer_UTF8(text));
}

constexpr auto presetWildcard = "*.ravepreset";
constexpr auto presetExtension = ".ravepreset";

class MainComponent final : public juce::Component,
                            private juce::Timer,
                            private juce::MidiInputCallback
{
public:
    MainComponent() : deviceSelector(deviceManager, 0, 2, 0, 2, false, false, true, false)
    {
        icon = juce::Drawable::createFromImageData(RaveIconAssets::raveinstrumenticon_svg,
                                                   RaveIconAssets::raveinstrumenticon_svgSize);
        if (icon != nullptr)
        {
            icon->setAccessible(true);
            icon->setTitle("RAVE Performance Instrument original icon");
            addAndMakeVisible(*icon);
        }

        title.setText("RAVE Performance Instrument", juce::dontSendNotification);
        title.setJustificationType(juce::Justification::centred);
        title.setFont(juce::Font(24.0f, juce::Font::bold));
        title.setAccessible(true);
        title.setTitle("RAVE Performance Instrument");
        addAndMakeVisible(title);

        status.setText(utf8("Audio pass-through ready — no model loaded"),
                       juce::dontSendNotification);
        status.setJustificationType(juce::Justification::centred);
        status.setAccessible(true);
        status.setTitle("Model and audio status");
        addAndMakeVisible(status);

        configureButton(loadModelButton, "Load TorchScript Model…", "Choose a TorchScript model");
        loadModelButton.onClick = [this] { chooseModel(false); };
#if !RAVE_HAS_LIBTORCH
        loadModelButton.setButtonText("LibTorch backend unavailable");
        loadModelButton.setEnabled(false);
#endif

        configureButton(relinkButton, "Relink Missing Model…",
                        "Choose a replacement for the missing preset model");
        relinkButton.onClick = [this] { chooseModel(true); };
        relinkButton.setEnabled(false);

        configureButton(savePresetButton, "Save Preset…",
                        "Save a RAVE standalone .ravepreset file");
        savePresetButton.onClick = [this] { savePreset(); };
        configureButton(loadPresetButton, "Load Preset…",
                        "Load a RAVE standalone .ravepreset file");
        loadPresetButton.onClick = [this] { loadPreset(); };

        dryWet.setRange(0.0, 1.0, 0.01);
        dryWet.setTextValueSuffix(" wet");
        dryWet.setAccessible(true);
        dryWet.setTitle("Dry wet mix");
        dryWet.setTooltip("Amount of processed audio. MIDI learn is available "
                          "beside this control.");
        dryWet.onValueChange = [this]
        {
            session.setDryWet(static_cast<float>(dryWet.getValue()));
            engine.setDryWet(session.dryWet());
        };
        addAndMakeVisible(dryWet);
        dryWetLabel.setText("Dry / Wet", juce::dontSendNotification);
        dryWetLabel.attachToComponent(&dryWet, true);
        addAndMakeVisible(dryWetLabel);
        configureButton(dryWetLearnButton, "MIDI Learn", "Learn a MIDI CC for dry/wet");
        dryWetLearnButton.onClick = [this]
        { session.beginMidiLearn(rave::StandaloneSessionState::dryWetTarget); };
        configureButton(dryWetClearButton, "Clear", "Clear the dry/wet MIDI CC assignment");
        dryWetClearButton.onClick = [this]
        { session.clearMidiMapping(rave::StandaloneSessionState::dryWetTarget); };

        midiInputLabel.setText("MIDI input", juce::dontSendNotification);
        midiInputLabel.setAccessible(true);
        midiInputLabel.setTitle("MIDI input selection");
        addAndMakeVisible(midiInputLabel);
        midiInput.setAccessible(true);
        midiInput.setTitle("MIDI input");
        midiInput.setTooltip("Select one MIDI input. Only that input supplies CC controls.");
        midiInput.onChange = [this] { selectMidiInputFromControl(); };
        addAndMakeVisible(midiInput);
        midiStatus.setJustificationType(juce::Justification::centredLeft);
        midiStatus.setAccessible(true);
        midiStatus.setTitle("MIDI connection status");
        addAndMakeVisible(midiStatus);

        deviceSelector.setAccessible(true);
        deviceSelector.setTitle("Audio input and output device settings");
        addAndMakeVisible(deviceSelector);

        latentViewport.setViewedComponent(&latentContent, false);
        latentViewport.setScrollBarsShown(true, false);
        latentViewport.setAccessible(true);
        latentViewport.setTitle("Latent controls");
        addAndMakeVisible(latentViewport);

        const auto error = deviceManager.initialiseWithDefaultDevices(2, 2);
        if (error.isNotEmpty())
            status.setText("Audio device error: " + error, juce::dontSendNotification);
        else
            deviceManager.addAudioCallback(&engine);
        syncAudioIdentityFromManager();
        refreshMidiInputs();
        setSize(760, 620);
        startTimerHz(20);
    }

    ~MainComponent() override
    {
        stopTimer();
        disableSelectedMidiInput();
        fileChooser.reset();
        deviceManager.removeAudioCallback(&engine);
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(18);
        auto titleArea = area.removeFromTop(34);
        if (icon != nullptr)
            icon->setBounds(titleArea.removeFromLeft(34).reduced(2));
        title.setBounds(titleArea);
        status.setBounds(area.removeFromTop(28));
        area.removeFromTop(6);

        auto modelRow = area.removeFromTop(30);
        loadModelButton.setBounds(modelRow.removeFromLeft(modelRow.getWidth() / 2).reduced(2, 0));
        relinkButton.setBounds(modelRow.reduced(2, 0));
        auto presetRow = area.removeFromTop(30);
        savePresetButton.setBounds(
            presetRow.removeFromLeft(presetRow.getWidth() / 2).reduced(2, 0));
        loadPresetButton.setBounds(presetRow.reduced(2, 0));
        area.removeFromTop(6);

        auto midiRow = area.removeFromTop(28);
        midiInputLabel.setBounds(midiRow.removeFromLeft(78));
        midiStatus.setBounds(midiRow.removeFromRight(std::max(120, midiRow.getWidth() / 3)));
        midiInput.setBounds(midiRow.reduced(2, 0));

        auto mixRow = area.removeFromTop(36);
        dryWetLearnButton.setBounds(mixRow.removeFromRight(96));
        mixRow.removeFromRight(4);
        dryWetClearButton.setBounds(mixRow.removeFromRight(52));
        mixRow.removeFromRight(8);
        dryWet.setBounds(mixRow.reduced(82, 0));
        area.removeFromTop(6);

        const auto settingsHeight = std::min(150, std::max(112, area.getHeight() / 3));
        deviceSelector.setBounds(area.removeFromTop(settingsHeight));
        area.removeFromTop(6);
        latentViewport.setBounds(area);
        layoutLatentControls();
    }

private:
    struct ModelRequest
    {
        juce::File file;
        std::uint64_t generation = 0;
        bool relink = false;
        // Preset and relink replacements deliberately mute a previous model
        // until the requested identity activates. Ordinary candidate loads keep
        // the previous-working-model behavior.
        bool replacement = false;
    };

    void configureButton(juce::TextButton& button, const juce::String& text,
                         const juce::String& tooltip)
    {
        button.setButtonText(text);
        button.setAccessible(true);
        button.setTitle(text);
        button.setTooltip(tooltip);
        addAndMakeVisible(button);
    }

    void rebuildLatentControls(const std::size_t dimensionCount)
    {
        const auto clampedCount =
            std::min(dimensionCount, rave::StandaloneSessionState::maximumLatents);
        session.setLatentCount(clampedCount);
        latentSliders.clear();
        latentLabels.clear();
        latentLearnButtons.clear();
        latentClearButtons.clear();
        latentSliders.reserve(clampedCount);
        latentLabels.reserve(clampedCount);
        latentLearnButtons.reserve(clampedCount);
        latentClearButtons.reserve(clampedCount);
        for (std::size_t index = 0; index < clampedCount; ++index)
        {
            auto label = std::make_unique<juce::Label>();
            label->setText("Latent " + juce::String(index + 1), juce::dontSendNotification);
            label->setJustificationType(juce::Justification::centredLeft);
            latentContent.addAndMakeVisible(*label);
            auto slider = std::make_unique<juce::Slider>(juce::Slider::LinearHorizontal,
                                                         juce::Slider::TextBoxRight);
            slider->setRange(-4.0, 4.0, 0.01);
            slider->setValue(session.latent(index), juce::dontSendNotification);
            slider->setDoubleClickReturnValue(true, 0.0);
            slider->setAccessible(true);
            slider->setTitle("Latent " + juce::String(index + 1));
            slider->setTooltip("Latent " + juce::String(index + 1) + " offset");
            auto* pointer = slider.get();
            slider->onValueChange = [this, index, pointer]
            {
                static_cast<void>(
                    session.setLatent(index, static_cast<float>(pointer->getValue())));
                static_cast<void>(engine.setLatentControl(index, session.latent(index)));
            };
            latentContent.addAndMakeVisible(*slider);
            auto learn = std::make_unique<juce::TextButton>();
            learn->setAccessible(true);
            learn->setTitle("Learn MIDI CC for latent " + juce::String(index + 1));
            learn->setTooltip("Learn a MIDI CC for this latent");
            learn->onClick = [this, index] { session.beginMidiLearn(static_cast<int>(index)); };
            latentContent.addAndMakeVisible(*learn);
            auto clear = std::make_unique<juce::TextButton>("Clear");
            clear->setAccessible(true);
            clear->setTitle("Clear MIDI CC for latent " + juce::String(index + 1));
            clear->setTooltip("Clear the MIDI CC assignment for this latent");
            clear->onClick = [this, index] { session.clearMidiMapping(static_cast<int>(index)); };
            latentContent.addAndMakeVisible(*clear);
            latentLabels.push_back(std::move(label));
            latentSliders.push_back(std::move(slider));
            latentLearnButtons.push_back(std::move(learn));
            latentClearButtons.push_back(std::move(clear));
        }
        layoutLatentControls();
    }

    void layoutLatentControls()
    {
        constexpr int rowHeight = 42;
        const auto contentWidth = std::max(420, latentViewport.getWidth() - 14);
        latentContent.setSize(contentWidth,
                              std::max(latentViewport.getHeight(),
                                       static_cast<int>(latentSliders.size()) * rowHeight));
        for (std::size_t index = 0; index < latentSliders.size(); ++index)
        {
            const auto y = static_cast<int>(index) * rowHeight;
            latentLabels[index]->setBounds(0, y, 76, rowHeight);
            latentClearButtons[index]->setBounds(contentWidth - 50, y + 7, 48, rowHeight - 14);
            latentLearnButtons[index]->setBounds(contentWidth - 148, y + 7, 94, rowHeight - 14);
            latentSliders[index]->setBounds(80, y + 3, contentWidth - 232, rowHeight - 6);
        }
    }

    void updateMidiLabels()
    {
        const auto textFor = [this](const int target)
        {
            // StandaloneSessionState intentionally has no widget access from the MIDI
            // callback.
            return session.midiController(target) >= 0
                       ? "CC " + juce::String(session.midiController(target))
                       : "MIDI Learn";
        };
        dryWetLearnButton.setButtonText(textFor(rave::StandaloneSessionState::dryWetTarget));
        for (std::size_t index = 0; index < latentLearnButtons.size(); ++index)
            latentLearnButtons[index]->setButtonText(textFor(static_cast<int>(index)));
    }

    void syncControlsAndEngine()
    {
        const auto state = session.snapshot(); // message-thread snapshot of all realtime values
        dryWet.setValue(state.dryWet, juce::dontSendNotification);
        engine.setDryWet(state.dryWet);
        const auto engineCount = engine.latentDimensionCount();
        // A restored preset owns its shape until its named model either activates
        // or is explicitly cleared. In particular, do not let a fresh/old engine
        // with zero dimensions destroy restored latent values and MIDI mappings.
        if (!presetModelCoordinator.presetControlsAreAuthoritative()
            && engineCount != latentSliders.size())
            rebuildLatentControls(engineCount);
        const auto count = std::min(state.latents.size(), latentSliders.size());
        for (std::size_t index = 0; index < count; ++index)
        {
            latentSliders[index]->setValue(state.latents[index], juce::dontSendNotification);
            static_cast<void>(engine.setLatentControl(index, state.latents[index]));
        }
        updateMidiLabels();
    }

    void syncAudioIdentityFromManager()
    {
        const auto setup = deviceManager.getAudioDeviceSetup();
        session.setAudioSetup(deviceManager.getCurrentAudioDeviceType(), setup.outputDeviceName,
                              setup.inputDeviceName);
    }

    bool applyAudioSetup(const rave::StandaloneSessionState::Snapshot& preset)
    {
        juce::String error;
        if (preset.audioDeviceType.isNotEmpty())
            deviceManager.setCurrentAudioDeviceType(preset.audioDeviceType, true);
        {
            auto setup = deviceManager.getAudioDeviceSetup();
            if (preset.audioInputId.isNotEmpty())
                setup.inputDeviceName = preset.audioInputId;
            if (preset.audioOutputId.isNotEmpty())
                setup.outputDeviceName = preset.audioOutputId;
            error = deviceManager.setAudioDeviceSetup(setup, true);
        }
        if (error.isNotEmpty())
        {
            status.setText("Audio device error: " + error, juce::dontSendNotification);
            return false;
        }
        syncAudioIdentityFromManager();
        return true;
    }

    void refreshMidiInputs()
    {
        const auto devices = juce::MidiInput::getAvailableDevices();
        juce::StringArray ids;
        for (const auto& device : devices)
            ids.add(device.identifier);
        if (ids != knownMidiIds)
        {
            knownMidiIds = ids;
            refreshingMidiControl = true;
            midiInput.clear(juce::dontSendNotification);
            midiInput.addItem("No MIDI input", 1);
            for (int index = 0; index < devices.size(); ++index)
                midiInput.addItem(devices.getReference(index).name, index + 2);
            refreshingMidiControl = false;
        }
        const auto requested = session.snapshot().midiInputId;
        auto selected = 0;
        for (int index = 0; index < devices.size(); ++index)
            if (devices.getReference(index).identifier == requested)
                selected = index + 2;
        refreshingMidiControl = true;
        midiInput.setSelectedId(selected == 0 ? 1 : selected, juce::dontSendNotification);
        refreshingMidiControl = false;
        if (requested.isNotEmpty() && selected == 0)
        {
            // Keep the saved identifier for reconnection, while removing the stale
            // callback.
            disableSelectedMidiInput();
            midiStatus.setText("Selected input unavailable; waiting to reconnect",
                               juce::dontSendNotification);
        }
        else if (selected != 0)
        {
            enableMidiInput(requested);
            midiStatus.setText("Selected input active", juce::dontSendNotification);
        }
        else
            midiStatus.setText("No MIDI input selected", juce::dontSendNotification);
    }

    void disableSelectedMidiInput()
    {
        if (activeMidiInputId.isNotEmpty())
        {
            deviceManager.removeMidiInputDeviceCallback(activeMidiInputId, this);
            deviceManager.setMidiInputDeviceEnabled(activeMidiInputId, false);
            activeMidiInputId.clear();
        }
    }

    void enableMidiInput(const juce::String& identifier)
    {
        if (activeMidiInputId == identifier)
            return;
        disableSelectedMidiInput();
        // Exactly one available input owns this callback at a time.
        deviceManager.setMidiInputDeviceEnabled(identifier, true);
        deviceManager.addMidiInputDeviceCallback(identifier, this);
        activeMidiInputId = identifier;
    }

    void selectMidiInputFromControl()
    {
        if (refreshingMidiControl)
            return;
        const auto devices = juce::MidiInput::getAvailableDevices();
        const auto selected = midiInput.getSelectedId() - 2;
        if (selected < 0 || selected >= devices.size())
        {
            disableSelectedMidiInput();
            session.setMidiInputId({});
            refreshMidiInputs();
            return;
        }
        const auto& device = devices.getReference(selected);
        session.setMidiInputId(device.identifier);
        enableMidiInput(device.identifier);
        refreshMidiInputs();
    }

    void handleIncomingMidiMessage(juce::MidiInput*, const juce::MidiMessage& message) override
    {
        if (message.isController())
            static_cast<void>(
                session.applyMidiCc(message.getControllerNumber(), message.getControllerValue()));
    }

    void savePreset()
    {
        syncAudioIdentityFromManager();
        fileChooser =
            std::make_unique<juce::FileChooser>("Save RAVE preset", juce::File{}, presetWildcard);
        const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
        fileChooser->launchAsync(
            juce::FileBrowserComponent::saveMode | juce::FileBrowserComponent::canSelectFiles,
            [safeThis](const juce::FileChooser& chooser)
            {
                if (safeThis == nullptr)
                    return;
                auto file = chooser.getResult();
                if (file.getFileNameWithoutExtension().isEmpty())
                    return;
                if (!file.hasFileExtension(presetExtension))
                    file = file.withFileExtension(presetExtension);
                juce::MemoryBlock data;
                if (safeThis->session.serialize(data) &&
                    file.replaceWithData(data.getData(), data.getSize()))
                    safeThis->status.setText("Preset saved: " + file.getFileName(),
                                             juce::dontSendNotification);
                else
                    safeThis->status.setText("Could not save preset", juce::dontSendNotification);
            });
    }

    void loadPreset()
    {
        fileChooser =
            std::make_unique<juce::FileChooser>("Load RAVE preset", juce::File{}, presetWildcard);
        const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
        fileChooser->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
            [safeThis](const juce::FileChooser& chooser)
            {
                if (safeThis == nullptr)
                    return;
                const auto file = chooser.getResult();
                juce::MemoryBlock data;
                rave::StandaloneSessionState parsed;
                if (!file.existsAsFile()
                    || !rave::StandaloneSessionState::readBoundedPresetFile(file, data)
                    || !parsed.deserialize(data.getData(), data.getSize()))
                {
                    safeThis->status.setText("Preset load failed: invalid .ravepreset",
                                             juce::dontSendNotification);
                    return;
                }
                safeThis->applyPreset(parsed.snapshot());
            });
    }

    void applyPreset(const rave::StandaloneSessionState::Snapshot& preset)
    {
        // A preset supersedes any in-flight model request even when it names no
        // model or a missing one; no older background result may activate
        // afterward.
        const auto generation = requestGate.begin();
        // Validation happened in a temporary state above, so this message-thread
        // application is fail-closed. The coordinator is also the timer's
        // authority boundary for the restored control shape.
        if (!presetModelCoordinator.applyPreset(preset, generation))
            return;
        engine.setWetSuppressed(true);
        applyAudioSetup(preset);
        rebuildLatentControls(preset.latents.size());
        refreshMidiInputs();
        if (preset.midiInputId.isNotEmpty())
        {
            const auto devices = juce::MidiInput::getAvailableDevices();
            for (int index = 0; index < devices.size(); ++index)
                if (devices.getReference(index).identifier == preset.midiInputId)
                {
                    refreshingMidiControl = true;
                    midiInput.setSelectedId(index + 2, juce::sendNotification);
                    refreshingMidiControl = false;
                    selectMidiInputFromControl();
                    break;
                }
        }
        syncControlsAndEngine();
        if (preset.modelPath.isEmpty())
            rave::clearStandaloneModelBackend(
                engine,
                [this] { deviceManager.removeAudioCallback(&engine); },
                [this] { deviceManager.addAudioCallback(&engine); });
#if RAVE_HAS_LIBTORCH
        if (preset.modelPath.isNotEmpty() && juce::File(preset.modelPath).existsAsFile())
            requestModel(juce::File(preset.modelPath), false, true, generation);
        else if (preset.modelPath.isNotEmpty())
        {
            presetModelCoordinator.markPresetModelMissing(generation);
            relinkRequired = presetModelCoordinator.relinkRequired();
            relinkButton.setEnabled(relinkRequired);
            status.setText("Preset model is missing; relink required: " +
                               juce::File(preset.modelPath).getFileName(),
                           juce::dontSendNotification);
        }
        else
        {
            relinkRequired = false;
            relinkButton.setEnabled(false);
            status.setText("Preset loaded — no model selected (aligned dry)",
                           juce::dontSendNotification);
        }
#else
        status.setText(preset.modelPath.isEmpty()
                           ? "Preset loaded — no model selected (aligned dry)"
                           : "Preset loaded; LibTorch backend unavailable",
                       juce::dontSendNotification);
#endif
    }

#if RAVE_HAS_LIBTORCH
    void chooseModel(const bool relink)
    {
        fileChooser = std::make_unique<juce::FileChooser>(
            relink ? "Relink missing TorchScript model" : "Choose a TorchScript model",
            juce::File{}, "*.ts;*.pt;*.pth");
        const auto safeThis = juce::Component::SafePointer<MainComponent>(this);
        fileChooser->launchAsync(juce::FileBrowserComponent::openMode |
                                     juce::FileBrowserComponent::canSelectFiles,
                                 [safeThis, relink](const juce::FileChooser& chooser)
                                 {
                                     if (safeThis == nullptr)
                                         return;
                                     const auto file = chooser.getResult();
                                     if (file.existsAsFile())
                                         safeThis->requestModel(file, relink);
                                 });
    }

    void requestModel(const juce::File& file,
                      const bool relink,
                      const bool replacement = false,
                      const std::optional<std::uint64_t> generation = std::nullopt)
    {
        const auto requestGeneration = generation.value_or(requestGate.begin());
        const auto replacesPresetIdentity = presetModelCoordinator.beginModelRequest(
            requestGeneration, replacement || relink);
        if (replacesPresetIdentity)
            engine.setWetSuppressed(true);
        startModelRequest(ModelRequest{file, requestGeneration, relink, replacesPresetIdentity});
    }

    void startModelRequest(const ModelRequest& request)
    {
        if (!requestGate.isCurrent(request.generation))
            return;

        const auto configuration = engine.runtimeConfiguration();
        if (configuration.sampleRate <= 0.0 || configuration.maximumBlockSize == 0)
        {
            deferredRequest.replace(request, request.generation);
            status.setText("Audio device not ready — model request deferred",
                           juce::dontSendNotification);
            return;
        }

        if (modelLoader->state() == rave::BackgroundModelLoader::State::loading)
        {
            queuedRequest.replace(request, request.generation);
            status.setText("Queued latest model request: " + request.file.getFileName(),
                           juce::dontSendNotification);
            return;
        }

        if (!modelLoader->start(request.file.getFullPathName().toStdString(), configuration))
        {
            if (request.replacement)
            {
                presetModelCoordinator.markReplacementFailed(request.generation);
                relinkRequired = presetModelCoordinator.relinkRequired();
                relinkButton.setEnabled(relinkRequired);
            }
            status.setText("Could not start model load", juce::dontSendNotification);
            return;
        }

        pendingRequest = request;
        loadModelButton.setEnabled(false);
        status.setText("Loading " + request.file.getFileName() + utf8("…"),
                       juce::dontSendNotification);
    }

    void startQueuedRequestIfAny()
    {
        if (const auto request = queuedRequest.takeCurrent(requestGate); request.has_value())
            startModelRequest(*request);
    }

    void startDeferredRequestIfReady()
    {
        const auto configuration = engine.runtimeConfiguration();
        if (configuration.sampleRate <= 0.0 || configuration.maximumBlockSize == 0 ||
            modelLoader->state() == rave::BackgroundModelLoader::State::loading)
            return;

        if (const auto request = deferredRequest.takeCurrent(requestGate); request.has_value())
            startModelRequest(*request);
    }

    void pollModelLoader()
    {
        const auto loaderState = modelLoader->state();
        if (loaderState != rave::BackgroundModelLoader::State::succeeded &&
            loaderState != rave::BackgroundModelLoader::State::failed)
            return;
        const auto result = modelLoader->takeResult();
        loadModelButton.setEnabled(true);
        const auto request = pendingRequest;
        pendingRequest.reset();
        if (!request.has_value() || !requestGate.isCurrent(request->generation))
        {
            startQueuedRequestIfAny();
            return;
        }
        if (result.state == rave::BackgroundModelLoader::State::failed)
        {
            if (request->replacement)
            {
                presetModelCoordinator.markReplacementFailed(request->generation);
                relinkRequired = presetModelCoordinator.relinkRequired();
                relinkButton.setEnabled(relinkRequired);
            }
            const auto lifecycleSnapshot = engine.lifecycleStatusSnapshot();
            status.setText(
                rave::standaloneModelLoadFailureStatus(result.errorMessage,
                                                       request->replacement,
                                                       relinkRequired,
                                                       lifecycleSnapshot),
                juce::dontSendNotification);
            // Keep the concrete qualification/load event until a genuinely
            // newer lifecycle outcome exists. Coordinator authority separately
            // protects replacement/relink status while wet remains suppressed.
            lifecycleStatusGate.markObserved(lifecycleSnapshot.revision);
            startQueuedRequestIfAny();
            return;
        }
        deviceManager.removeAudioCallback(&engine);
        rave::ScopedCallbackReattachment callbackReattachment(
            [this] { deviceManager.addAudioCallback(&engine); });
        std::string failure;
        bool activated = false;
        try
        {
            activated = engine.activateModelBackend(std::move(result.backend), &failure);
        }
        catch (const std::exception& exception)
        {
            failure = exception.what();
        }
        catch (...)
        {
            failure = "unknown activation error";
        }
        if (!requestGate.isCurrent(request->generation))
        {
            startQueuedRequestIfAny();
            return;
        }
        if (!activated)
        {
            if (request->replacement)
            {
                presetModelCoordinator.markReplacementFailed(request->generation);
                relinkRequired = presetModelCoordinator.relinkRequired();
                relinkButton.setEnabled(relinkRequired);
            }
            callbackReattachment.reattach();
            const auto lifecycleSnapshot = engine.lifecycleStatusSnapshot();
            const auto cause = rave::candidateFailureCause(
                failure.empty() ? "candidate was not accepted" : failure);
            status.setText(
                rave::standaloneModelFailureStatus("Model activation failed", cause,
                                                   request->replacement, relinkRequired,
                                                   lifecycleSnapshot),
                juce::dontSendNotification);
            // The event already includes the coherent post-reattach outcome;
            // only a genuinely later lifecycle event may supersede it.
            lifecycleStatusGate.markObserved(lifecycleSnapshot.revision);
            startQueuedRequestIfAny();
            return;
        }
        // Reconcile the preset's retained shape before normal engine authority
        // resumes. Matching latent values and MIDI mappings survive by index.
        if (!presetModelCoordinator.reconcileActivatedModel(request->generation,
                                                             engine.latentDimensionCount()))
        {
            startQueuedRequestIfAny();
            return;
        }
        session.setModelPath(request->file.getFullPathName());
        rebuildLatentControls(engine.latentDimensionCount());
        syncControlsAndEngine();
        if (!presetModelCoordinator.completeActivation(request->generation))
        {
            startQueuedRequestIfAny();
            return;
        }
        relinkRequired = false;
        relinkButton.setEnabled(false);
        // Controls and coordinator authority are committed while detached. Keep
        // replacement wet suppressed through reattachment, then permit frames.
        callbackReattachment.reattach();
        if (request->replacement)
            engine.setWetSuppressed(false);
        refreshLifecycleStatus();
        startQueuedRequestIfAny();
    }
#else
    void chooseModel(bool) {}
#endif

    void refreshLifecycleStatus()
    {
        const auto snapshot = engine.lifecycleStatusSnapshot();
        if (!presetModelCoordinator.shouldPublishLifecycleRevision(
                snapshot.revision, lifecycleStatusGate))
            return;
        status.setText(
            rave::lifecycleStatusText(snapshot.diagnostic, snapshot.modelUsable,
                                      snapshot.modelInstalled, {}, snapshot.latentDimensionCount,
                                      utf8("Audio pass-through ready — no model loaded")),
            juce::dontSendNotification);
    }

    void timerCallback() override
    {
        syncAudioIdentityFromManager();
        refreshMidiInputs();
        syncControlsAndEngine();
        refreshLifecycleStatus();
#if RAVE_HAS_LIBTORCH
        pollModelLoader();
        startDeferredRequestIfReady();
#endif
    }

    juce::AudioDeviceManager deviceManager;
    rave::RaveAudioEngine engine;
    rave::StandaloneSessionState session;
    rave::StandalonePresetModelCoordinator presetModelCoordinator { session };
    rave::LatestRequestGeneration requestGate;
    juce::AudioDeviceSelectorComponent deviceSelector;
    std::unique_ptr<juce::Drawable> icon;
    juce::Label title, status, dryWetLabel, midiInputLabel, midiStatus;
    juce::TextButton loadModelButton, relinkButton, savePresetButton, loadPresetButton,
        dryWetLearnButton, dryWetClearButton;
    juce::ComboBox midiInput;
    juce::Slider dryWet{juce::Slider::LinearHorizontal, juce::Slider::TextBoxRight};
    juce::Component latentContent;
    juce::Viewport latentViewport;
    std::vector<std::unique_ptr<juce::Label>> latentLabels;
    std::vector<std::unique_ptr<juce::Slider>> latentSliders;
    std::vector<std::unique_ptr<juce::TextButton>> latentLearnButtons, latentClearButtons;
    std::unique_ptr<juce::FileChooser> fileChooser;
    juce::StringArray knownMidiIds;
    juce::String activeMidiInputId;
    bool refreshingMidiControl = false;
    rave::LifecycleStatusRevisionGate lifecycleStatusGate;
#if RAVE_HAS_LIBTORCH
    bool relinkRequired = false;
    std::unique_ptr<rave::BackgroundModelLoader> modelLoader =
        std::make_unique<rave::BackgroundModelLoader>(
            [] { return std::make_shared<rave::TorchScriptBackend>(); });
    std::optional<ModelRequest> pendingRequest;
    rave::LatestRequestSlot<ModelRequest> deferredRequest;
    rave::LatestRequestSlot<ModelRequest> queuedRequest;
#endif
};

class MainWindow final : public juce::DocumentWindow
{
public:
    explicit MainWindow(const juce::String& name)
        : DocumentWindow(name,
                         juce::Desktop::getInstance().getDefaultLookAndFeel().findColour(
                             juce::ResizableWindow::backgroundColourId),
                         DocumentWindow::allButtons)
    {
        setUsingNativeTitleBar(true);
        setContentOwned(new MainComponent(), true);
        setResizable(true, true);
        setResizeLimits(480, 420, 1600, 1200);
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
    [[nodiscard]] const juce::String getApplicationVersion() override { return "0.1.0"; }
    void initialise(const juce::String&) override
    {
        window = std::make_unique<MainWindow>(getApplicationName());
    }
    void shutdown() override { window.reset(); }

private:
    std::unique_ptr<MainWindow> window;
};
} // namespace
START_JUCE_APPLICATION(RaveApplication)
