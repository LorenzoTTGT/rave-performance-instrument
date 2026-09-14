#pragma once

#include "engine/RaveAudioEngine.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include <juce_core/juce_core.h>

namespace rave
{
class LifecycleStatusRevisionGate
{
public:
    [[nodiscard]] bool shouldPublish(const std::uint64_t revision) noexcept
    {
        if (revision <= observedRevision)
            return false;
        observedRevision = revision;
        return true;
    }

    // Preserve an explicit event rendered from this coherent lifecycle
    // revision. Only a later lifecycle change may supersede it.
    void markObserved(const std::uint64_t revision) noexcept
    {
        if (revision > observedRevision)
            observedRevision = revision;
    }

private:
    std::uint64_t observedRevision = 0;
};

// Activation failure strings may carry an engine-appended rollback outcome.
// Remove only recognized suffixes: backend diagnostics are otherwise opaque
// and may legitimately begin with or contain semicolons.
[[nodiscard]] inline std::string candidateFailureCause(const std::string& activationFailure)
{
    static constexpr const char* exactSuffixes[] {
        "; previous model retained",
        "; previous model is installed but not usable — remaining in bounded dry pass-through",
        "; no previous model active",
        "; no previous model to retain — remaining in bounded dry pass-through"
    };
    for (const auto* suffix : exactSuffixes)
    {
        const std::string suffixText(suffix);
        if (activationFailure.size() >= suffixText.size()
            && activationFailure.compare(activationFailure.size() - suffixText.size(),
                                         suffixText.size(), suffixText) == 0)
            return activationFailure.substr(0, activationFailure.size() - suffixText.size());
    }

    static constexpr const char* rollbackPrefixes[] {
        "; previous model could not be restarted",
        "; rollback threw"
    };
    static constexpr const char* rollbackEnding = " — remaining in bounded dry pass-through";
    if (activationFailure.size() >= std::char_traits<char>::length(rollbackEnding)
        && activationFailure.compare(activationFailure.size() - std::char_traits<char>::length(rollbackEnding),
                                     std::char_traits<char>::length(rollbackEnding), rollbackEnding) == 0)
    {
        for (const auto* prefix : rollbackPrefixes)
        {
            const auto position = activationFailure.rfind(prefix);
            if (position != std::string::npos)
                return activationFailure.substr(0, position);
        }
    }
    return activationFailure;
}

[[nodiscard]] inline std::size_t standaloneLatentControlCount(
    const RaveAudioEngine::LifecycleStatusSnapshot& snapshot) noexcept
{
    return snapshot.modelInstalled ? snapshot.latentDimensionCount : 0;
}

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

[[nodiscard]] inline juce::String contextualLifecycleStatus(
    const juce::String& eventContext,
    const RaveAudioEngine::LifecycleStatusSnapshot& snapshot,
    const juce::String& activeModelName,
    const juce::String& noModelText)
{
    const auto current = lifecycleStatusText(snapshot.diagnostic,
                                             snapshot.modelUsable,
                                             snapshot.modelInstalled,
                                             activeModelName,
                                             snapshot.latentDimensionCount,
                                             noModelText);
    return eventContext.isEmpty()
        ? current
        : eventContext + juce::String(juce::CharPointer_UTF8(" — current state: ")) + current;
}
} // namespace rave
