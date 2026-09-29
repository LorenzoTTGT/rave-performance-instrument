#define RAVE_UI_TEST 1
#include "app/Main.cpp"

#include <cstdlib>
#include <iostream>

namespace
{
void checkStandalone(bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> void findStandaloneControls(juce::Component &parent, std::vector<T *> &matches)
{
    for (auto *child : parent.getChildren())
    {
        if (auto *match = dynamic_cast<T *>(child))
            matches.push_back(match);
        findStandaloneControls(*child, matches);
    }
}

void renderStandalone(MainComponent &component, const juce::String &name)
{
    const auto directory = juce::SystemStats::getEnvironmentVariable("RAVE_UI_PREVIEW_DIR", {});
    if (directory.isNotEmpty())
    {
        const auto folder = juce::File(juce::String(directory));
        checkStandalone(folder.createDirectory().wasOk(), "create standalone render directory");
        juce::FileOutputStream output(folder.getChildFile(name + ".png"));
        checkStandalone(output.openedOk() && output.setPosition(0) && output.truncate().wasOk() &&
                            juce::PNGImageFormat().writeImageToStream(
                                component.createComponentSnapshot(component.getLocalBounds()), output),
                        "render standalone component without opening a desktop window");
    }
}
} // namespace

void testStandaloneUi()
{
    const auto root = juce::File::getSpecialLocation(juce::File::tempDirectory)
                          .getChildFile("rave-factory-" + juce::Uuid().toString());
    checkStandalone(root.createDirectory().wasOk(), "create isolated factory layout test");
    for (const auto &layout :
         {juce::String("Standalone/App"), juce::String("Plugin.vst3/Contents/x86_64-linux/Plugin.so"),
          juce::String("Plugin.vst3/Contents/x86_64-win/RaveRuntime/RaveImplementation.dll"),
          juce::String("App.app/Contents/MacOS/App")})
    {
        const auto module = root.getChildFile(layout);
        auto models = layout.startsWith("Standalone")
                          ? module.getSiblingFile("Models")
                          : root.getChildFile(layout.upToFirstOccurrenceOf("/Contents/", false, false))
                                .getChildFile("Contents/Resources/Models");
        checkStandalone(models.createDirectory().wasOk() && models.getChildFile("manifest.json").replaceWithText("{}"),
                        "create packaged model manifest");
        checkStandalone(rave::ui::factoryModelDirectory(module) == models,
                        "factory discovery survives native artifact relocation");
    }
    checkStandalone(root.deleteRecursively(), "remove isolated factory layout test");
    MainComponent component(false); // No device access in component tests.
    std::vector<juce::Drawable *> drawings;
    findStandaloneControls(component, drawings);
    for (auto *drawing : drawings)
        if (drawing->getParentComponent() == &component)
            checkStandalone(drawing->getDrawableBounds().transformedBy(drawing->getTransform()).getWidth() <= 36.1f,
                            "standalone header SVG geometry stays within its slot");
    std::vector<juce::Slider *> sliders;
    findStandaloneControls(component, sliders);
    const auto mix =
        std::find_if(sliders.begin(), sliders.end(), [](auto *slider) { return slider->getName() == "Dry / wet mix"; });
    checkStandalone(mix != sliders.end() && (*mix)->getTextFromValue(0.5) == "50%",
                    "standalone mix uses percentage text from its first frame");
    std::vector<juce::TextButton *> buttons;
    findStandaloneControls(component, buttons);
    auto audio =
        std::find_if(buttons.begin(), buttons.end(), [](auto *button) { return button->getButtonText() == "Audio"; });
    auto midi =
        std::find_if(buttons.begin(), buttons.end(), [](auto *button) { return button->getButtonText() == "MIDI"; });
    checkStandalone(audio != buttons.end() && midi != buttons.end(), "settings and MIDI toggles exist");
    for (const auto size : {juce::Point<int>(640, 480), juce::Point<int>(700, 480), juce::Point<int>(1100, 820)})
    {
        component.setSize(size.x, size.y);
        for (bool settings : {false, true})
        {
            (*audio)->setToggleState(settings, juce::dontSendNotification);
            (*audio)->onClick();
            (*midi)->setToggleState(true, juce::dontSendNotification);
            (*midi)->onClick();
            for (auto *button : buttons)
                if (button->getParentComponent() == &component && button->isVisible())
                    checkStandalone(component.getLocalBounds().contains(button->getBounds()),
                                    "standalone controls remain inside every supported window size");
            std::vector<juce::AudioDeviceSelectorComponent *> selectors;
            findStandaloneControls(component, selectors);
            checkStandalone(selectors.size() == 1, "audio selector is retained");
            checkStandalone(selectors.front()->getParentComponent()->getHeight() >= selectors.front()->getBottom(),
                            "audio settings remain reachable through the scrollable panel");
        }
    }
    component.setSize(700, 480);
    renderStandalone(component, "standalone-settings");
    (*audio)->setToggleState(false, juce::dontSendNotification);
    (*audio)->onClick();
    (*midi)->setToggleState(false, juce::dontSendNotification);
    (*midi)->onClick();
    renderStandalone(component, "standalone-performance");
}
