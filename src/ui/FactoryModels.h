#pragma once
#include <juce_gui_basics/juce_gui_basics.h>

namespace rave::ui
{
inline juce::File
factoryModelDirectory(const juce::File &module = juce::File::getSpecialLocation(juce::File::currentExecutableFile))
{
    auto directory = module.getParentDirectory();
    for (int level = 0; level < 4; ++level)
    {
        for (const auto &relative : {"Models", "Resources/Models"})
        {
            const auto candidate = directory.getChildFile(relative);
            if (candidate.getChildFile("manifest.json").existsAsFile())
                return candidate;
        }
        directory = directory.getParentDirectory();
    }
    return {};
}

inline void showFactoryModels(juce::Component &anchor, std::function<void(juce::File)> selected,
                              std::function<void()> browse)
{
    juce::PopupMenu menu;
    juce::Array<juce::File> files;
    const auto root = factoryModelDirectory();
    const auto manifest = root.isDirectory() ? juce::JSON::parse(root.getChildFile("manifest.json")) : juce::var{};
    const auto entries = manifest.getProperty("models", juce::var{});
    if (const auto *models = entries.getArray())
        for (const auto &model : *models)
        {
            const auto file = root.getChildFile(model.getProperty("file", "").toString());
            if (!file.existsAsFile())
                continue;
            files.add(file);
            menu.addItem(files.size(),
                         model.getProperty("name", "Model").toString() + " / " +
                             juce::String(static_cast<double>(model.getProperty("sample_rate", 48000)) / 1000.0, 1) +
                             " kHz / " + model.getProperty("license", "").toString());
        }
    if (files.isEmpty())
        menu.addItem(-1, "Factory models are not installed", false);
    menu.addSeparator();
    menu.addItem(1000, "Browse for a model...");
    menu.showMenuAsync(juce::PopupMenu::Options().withTargetComponent(&anchor),
                       [files, selected = std::move(selected), browse = std::move(browse)](int result)
                       {
                           if (result == 1000)
                               browse();
                           else if (result > 0 && result <= files.size())
                               selected(files[result - 1]);
                       });
}
} // namespace rave::ui
