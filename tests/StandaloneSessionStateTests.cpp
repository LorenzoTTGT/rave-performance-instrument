#include "state/LatestRequestGeneration.h"
#include "state/StandaloneSessionState.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace
{
void require(const bool condition, const char* const message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void testRoundTripAndMidi()
{
    rave::StandaloneSessionState state;
    state.setLatentCount(12);
    for (std::size_t index = 0; index < 12; ++index)
        require(state.setLatent(index, static_cast<float>(index) / 4.0f - 1.0f),
                "set every latent");

    state.setDryWet(0.75f);
    state.setGenerator(true, 1.2f, 0.3f);
    state.setModelPath("model.ts");
    state.setMidiInputId("midi-1");
    state.setAudioSetup("CoreAudio", "out-1", "in-1");

    state.beginMidiLearn(10);
    require(state.applyMidiCc(74, 127), "learn latent");
    require(std::abs(state.latent(10) - 4.0f) < 0.001f,
            "CC maps latent to -4..4");

    state.beginMidiLearn(rave::StandaloneSessionState::dryWetTarget);
    require(state.applyMidiCc(74, 0), "reassign uniquely");
    require(state.midiController(10) == -1
                && state.midiController(rave::StandaloneSessionState::dryWetTarget) == 74,
            "unique reassignment");
    state.clearMidiMapping(rave::StandaloneSessionState::dryWetTarget);
    require(state.midiController(rave::StandaloneSessionState::dryWetTarget) == -1,
            "clear mapping");

    state.beginMidiLearn(9);
    require(state.applyMidiCc(21, 64), "learn persistent latent mapping");
    juce::MemoryBlock data;
    require(state.serialize(data), "serialize");

    rave::StandaloneSessionState restored;
    require(restored.deserialize(data.getData(), data.getSize()), "deserialize");
    const auto snapshot = restored.snapshot();
    require(snapshot.latents.size() == 12
                && snapshot.modelPath == "model.ts"
                && snapshot.midiInputId == "midi-1"
                && snapshot.audioOutputId == "out-1"
                && snapshot.audioInputId == "in-1",
            "round trip all identity and latents");
    require(restored.midiController(9) == 21, "mapping round trip");
    require(snapshot.generate && std::abs(snapshot.motionDepth - 1.2f) < 0.001f &&
                std::abs(snapshot.motionRate - 0.3f) < 0.001f,
            "generator preset round trip");
}

void testLatestRequestGeneration()
{
    rave::LatestRequestGeneration requests;
    rave::LatestRequestSlot<int> deferred;

    const auto first = requests.begin();
    deferred.replace(1, first);
    const auto second = requests.begin();
    deferred.replace(2, second);

    require(!requests.isCurrent(first) && requests.isCurrent(second),
            "newest model/preset/relink request wins deterministically");
    const auto newest = deferred.takeCurrent(requests);
    require(newest.has_value() && *newest == 2 && !deferred.hasValue(),
            "newest deferred request is retained and starts exactly once");

    deferred.replace(3, second);
    static_cast<void>(requests.begin());
    require(!deferred.takeCurrent(requests).has_value(),
            "a stale deferred request never starts");
}

void testMalformedInput()
{
    rave::StandaloneSessionState state;
    state.setLatentCount(1);
    auto snapshot = state.snapshot();
    snapshot.latents[0] = std::numeric_limits<float>::quiet_NaN();
    require(!state.restore(snapshot), "reject nonfinite");

    std::vector<char> huge(rave::StandaloneSessionState::maximumSerializedBytes + 1);
    require(!state.deserialize(huge.data(), huge.size()), "reject oversized");

    constexpr char unsupported[] = "<RaveStandaloneState version=\"2\"/>";
    require(!state.deserialize(unsupported, sizeof(unsupported) - 1),
            "reject unsupported version");
    constexpr char tooMany[] = "<RaveStandaloneState version=\"1\" count=\"4097\"/>";
    require(!state.deserialize(tooMany, sizeof(tooMany) - 1),
            "reject excessive latent count");
    constexpr char malformed[] = "not xml";
    require(!state.deserialize(malformed, sizeof(malformed) - 1), "reject malformed XML");
}

void testBoundedPresetFileRead()
{
    auto exact = juce::File::createTempFile("rave-exact-preset");
    auto oversized = juce::File::createTempFile("rave-oversized-preset");
    std::vector<char> exactBytes(rave::StandaloneSessionState::maximumSerializedBytes, 'x');
    std::vector<char> oversizedBytes(rave::StandaloneSessionState::maximumSerializedBytes + 1, 'y');
    require(exact.replaceWithData(exactBytes.data(), exactBytes.size()), "write exact-limit file");
    require(oversized.replaceWithData(oversizedBytes.data(), oversizedBytes.size()),
            "write oversized file");

    juce::MemoryBlock input;
    require(rave::StandaloneSessionState::readBoundedPresetFile(exact, input)
                && input.getSize() == rave::StandaloneSessionState::maximumSerializedBytes,
            "exact-limit preset file is read within bound");
    rave::StandaloneSessionState state;
    state.setDryWet(0.75f);
    require(!state.deserialize(input.getData(), input.getSize())
                && std::abs(state.dryWet() - 0.75f) < 0.001f,
            "invalid exact-limit content rejects transactionally");

    require(!rave::StandaloneSessionState::readBoundedPresetFile(oversized, input)
                && input.getSize() == 0,
            "oversized preset is rejected after at most limit plus one byte");
    require(std::abs(state.dryWet() - 0.75f) < 0.001f,
            "oversized file rejection preserves existing state");
    exact.deleteFile();
    oversized.deleteFile();
}

void testRestorePublishesControlsAtomicallyAgainstMidi()
{
    rave::StandaloneSessionState state;
    state.setLatentCount(1);
    state.beginMidiLearn(0);
    require(state.applyMidiCc(74, 0), "establish pre-restore MIDI mapping");

    auto restored = state.snapshot();
    restored.latents[0] = 1.5f;
    restored.midiControllers[1] = 75;
    bool seamEntered = false;
    state.restoreBeforePublishForTesting = [&] {
        seamEntered = true;
        require(state.applyMidiCc(74, 127),
                "MIDI at restore boundary applies only to previous control bank");
    };
    require(state.restore(restored), "restore complete banked control snapshot");
    state.restoreBeforePublishForTesting = {};
    require(seamEntered && state.midiController(0) == 75
                && std::abs(state.latent(0) - 1.5f) < 0.001f,
            "restore publication supersedes MIDI admitted before bank swap");
    require(state.applyMidiCc(75, 127) && std::abs(state.latent(0) - 4.0f) < 0.001f,
            "MIDI after bank swap is authoritative under restored mapping");

    auto firstRapidRestore = state.snapshot();
    firstRapidRestore.latents[0] = -1.0f;
    firstRapidRestore.midiControllers[1] = 76;
    auto secondRapidRestore = firstRapidRestore;
    secondRapidRestore.latents[0] = 2.0f;
    bool rapidRestoresRan = false;
    state.readerAdmissionInterleaveForTesting = [&] {
        if (rapidRestoresRan)
            return;
        rapidRestoresRan = true;
        require(state.restore(firstRapidRestore) && state.restore(secondRapidRestore),
                "two rapid restores complete during paused reader admission");
    };
    require(state.applyMidiCc(76, 127),
            "CAS admission retries safely across two-bank ABA publication");
    state.readerAdmissionInterleaveForTesting = {};
    require(rapidRestoresRan && state.midiController(0) == 76
                && std::abs(state.latent(0) - 4.0f) < 0.001f,
            "MIDI after rapid restores applies only to the final complete bank");
}

void testConcurrentCountAndRealtimeAccess()
{
    rave::StandaloneSessionState state;
    state.setLatentCount(12);
    state.beginMidiLearn(10);
    require(state.applyMidiCc(74, 64), "establish stress-test MIDI mapping");

    std::atomic<bool> start { false };
    std::atomic<bool> finished { false };
    std::atomic<bool> failed { false };
    std::thread realtimeReader([&] {
        while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();

        while (!finished.load(std::memory_order_acquire))
        {
            const auto observedCount = state.latentCount();
            for (std::size_t index = 0; index < observedCount; ++index)
            {
                if (!std::isfinite(state.latent(index)))
                    failed.store(true, std::memory_order_relaxed);
                static_cast<void>(state.midiController(static_cast<int>(index)));
            }
            static_cast<void>(state.applyMidiCc(74, 96));
        }
    });

    start.store(true, std::memory_order_release);
    for (std::size_t iteration = 0; iteration < 20000; ++iteration)
    {
        const auto nextCount = iteration % 2 == 0 ? std::size_t { 4 } : std::size_t { 64 };
        state.setLatentCount(nextCount);
        if (nextCount > 10)
        {
            state.beginMidiLearn(10);
            static_cast<void>(state.applyMidiCc(74, static_cast<int>(iteration % 128)));
        }
    }
    finished.store(true, std::memory_order_release);
    realtimeReader.join();

    require(!failed.load(std::memory_order_relaxed),
            "concurrent count changes keep realtime reads finite and lifetime-safe");
    require(state.latentCount() == 64, "stress test completes with expected count");
}
}

int main()
{
    testRoundTripAndMidi();
    testLatestRequestGeneration();
    testMalformedInput();
    testBoundedPresetFileRead();
    testRestorePublishesControlsAtomicallyAgainstMidi();
    testConcurrentCountAndRealtimeAccess();
    std::cout << "Standalone session state tests passed\n";
}
