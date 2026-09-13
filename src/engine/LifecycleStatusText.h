#pragma once

#include "engine/RaveAudioEngine.h"

#include <cstddef>
#include <string>

#include <juce_core/juce_core.h>

namespace rave
{
// Shared presenter for engine lifecycle state, used verbatim by the plugin
// status surface and the standalone app status label so both render identical,
// truthful text:
//
//   - a recorded lifecycle diagnostic (incompatible reprepare, failing checked
//     reset/start) is rendered verbatim, because it explains why the installed
//     model is silent (bounded dry pass-through),
//   - otherwise a usable model is claimed active with its latent dimension
//     count (with the active model name when the caller tracks one),
//   - otherwise an installed model is reported as installed but not running
//     (bounded dry pass-through), for example after release/device stop,
//   - otherwise the caller's no-model text.
//
// All typographic literals are built through explicit UTF-8 pointers because
// juce::String(const char*) rejects non-ASCII source literals.
[[nodiscard]] inline juce::String lifecycleStatusText(const std::string& lifecycleDiagnostic,
                                                      const bool modelUsable,
                                                      const bool modelInstalled,
                                                      const juce::String& activeModelName,
                                                      const std::size_t latentDimensionCount,
                                                      const juce::String& noModelText)
{
    if (!lifecycleDiagnostic.empty())
        return juce::String(juce::CharPointer_UTF8(lifecycleDiagnostic.c_str()));

    if (modelUsable)
    {
        auto text = activeModelName.isNotEmpty()
            ? activeModelName + juce::String(juce::CharPointer_UTF8(" active — "))
            : juce::String(juce::CharPointer_UTF8("Model active — "));
        text += juce::String(static_cast<int>(latentDimensionCount));
        text += " latent dimensions";
        return text;
    }

    if (modelInstalled)
        return juce::String(juce::CharPointer_UTF8(
            "Model installed but not running — bounded dry pass-through"));

    return noModelText;
}
} // namespace rave
