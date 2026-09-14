// RAVE-05 offline qualification harness.
//
// Benchmarks the actual TorchScript backend and the shared RaveAudioEngine
// processing path outside any realtime host, with explicit fail-closed
// threshold evaluation and honest "unverified" labeling for everything an
// offline harness cannot observe (audio devices, hosts, DAWs, xruns,
// perceptual latency, interactive soak).
//
// This tool is a standalone qualification command. It is intentionally NOT
// registered with CTest: the 2-hour soak must never run inside the ordinary
// test suite. See docs/QUALIFICATION.md for the evidence workflow.

#include "qualification/QualificationSupport.h"

#include "engine/RaveAudioEngine.h"
#include "model/ModelBackend.h"
#include "model/ModelQualification.h"
#include "model/TorchScriptBackend.h"

#include <juce_core/juce_core.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__) || defined(__unix__)
#include <sys/resource.h>
#include <stdlib.h>
#endif

namespace
{
using namespace ravequalification;

constexpr std::size_t inferenceQuantum = rave::RaveAudioEngine::inferenceQuantumSamples;
constexpr std::size_t transportLatency = rave::RaveAudioEngine::transportLatencySamples;
constexpr int exitPass = 0;
constexpr int exitThresholdFailure = 1;
constexpr int exitUsageError = 2;
constexpr int exitHashMismatch = 3;
constexpr int exitModelFailure = 4;
constexpr int exitUnexpectedError = 5;

constexpr double callbackP99PeriodFraction = 0.25;
constexpr double callbackMaxPeriodFraction = 0.75;
constexpr double inferenceP99FrameFraction = 0.90;

struct ThresholdResult
{
    std::string id;
    std::string scope;  // "measured" or "unverified"
    std::string status; // "pass", "fail", "not-run", "unverified"
    std::string detail;
};

// ---------------------------------------------------------------------------
// Environment and revision identity
// ---------------------------------------------------------------------------
struct RevisionInfo
{
    bool gitAvailable = false;
    std::string commit = "unknown";
    bool dirty = true; // fail closed: unknown cleanliness reports as dirty
    std::string sourceDir;
    std::size_t dirtyEntries = 0;
};

RevisionInfo captureRevision(const std::string& sourceDir)
{
    RevisionInfo info;
    info.sourceDir = sourceDir;
    // Explicit argument arrays avoid any dependence on shell-style command
    // tokenization or path quoting.
    const auto runGit = [](const juce::StringArray& arguments, juce::String& output) {
        juce::ChildProcess process;
        if (!process.start(arguments))
            return false;
        output = process.readAllProcessOutput().trim();
        return true;
    };

    juce::String revisionOutput;
    if (runGit(juce::StringArray { "git", "-C", sourceDir, "rev-parse", "HEAD" }, revisionOutput)
        && revisionOutput.length() == 40)
    {
        info.commit = revisionOutput.toStdString();
        info.gitAvailable = true;
    }
    if (info.gitAvailable)
    {
        juce::String statusOutput;
        if (runGit(juce::StringArray { "git", "-C", sourceDir, "status", "--porcelain" },
                   statusOutput))
        {
            if (statusOutput.isEmpty())
            {
                info.dirty = false;
                info.dirtyEntries = 0;
            }
            else
            {
                info.dirty = true;
                auto newlines = 0;
                for (const auto character : statusOutput)
                    if (character == '\n')
                        ++newlines;
                info.dirtyEntries = static_cast<std::size_t>(newlines) + 1;
            }
        }
        // If the status command itself fails, unknown cleanliness stays dirty.
    }
    return info;
}

std::string compilerDescription()
{
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

std::string cpuArchitecture()
{
#if defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#else
    return "unknown";
#endif
}

double peakRssMb()
{
#if defined(__APPLE__)
    rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0); // bytes on macOS
#elif defined(__unix__)
    rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
        return static_cast<double>(usage.ru_maxrss) * 1024.0 / (1024.0 * 1024.0); // KiB on Linux
#endif
    return -1.0;
}

// 1-minute system load average, recorded so timing evidence can be read in
// the context of competing machine load (-1 when unavailable).
double systemLoadAverage1m()
{
#if defined(__APPLE__) || defined(__unix__)
    double averages[3] { -1.0, -1.0, -1.0 };
    if (getloadavg(averages, 3) >= 1)
        return averages[0];
#endif
    return -1.0;
}

double nanosecondsToMs(const std::chrono::steady_clock::duration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

// ---------------------------------------------------------------------------
// Paced engine driving. One continuous loop keeps the stimulus oracle in sync
// with the engine's 4096-sample delayed-dry timeline across all segments.
// ---------------------------------------------------------------------------
struct ScheduleEntry
{
    std::string name;
    std::vector<std::size_t> callbackCycle;
    double pacedSeconds = 0.0;
    bool wet = false;
    bool alignmentChecked = false;
    std::uint64_t minimumSamples = 0;
};

struct SegmentResult
{
    std::string name;
    std::vector<std::size_t> callbackSizes;
    double pacedAudioSeconds = 0.0;
    bool wet = false;
    std::uint64_t calls = 0;
    std::uint64_t nonFiniteSamples = 0;
    std::uint64_t pacingLags = 0;
    bool outputFinite = true;
    bool alignmentChecked = false;
    std::uint64_t alignmentComparedSamples = 0;
    std::uint64_t alignmentMismatches = 0;
    TimingSummary timing;
};

struct PacedRunResult
{
    std::vector<SegmentResult> segments;
    std::uint64_t totalCalls = 0;
    std::uint64_t totalSamplesPerChannel = 0;
    double totalPacedAudioSeconds = 0.0;
    double wallClockSeconds = 0.0;
    std::uint64_t alignmentComparedSamples = 0;
    std::uint64_t alignmentMismatches = 0;
    bool outputFinite = true;
};

// Mirrors the engine's 4096-sample delayed-dry contract so the dry path can be
// compared sample-exactly against the deterministic stimulus.
class DelayedDryOracle
{
public:
    // Records the incoming sample; returns the stimulus from `transportLatency`
    // samples ago (the value leaving the ring).
    float push(const float sample)
    {
        const auto slot = writeIndex % transportLatency;
        const auto evicted = ring[slot];
        ring[slot] = sample;
        ++writeIndex;
        return evicted;
    }

private:
    std::vector<float> ring = std::vector<float>(transportLatency, 0.0f);
    std::uint64_t writeIndex = 0;
};

// Exact float comparison (bit identity). The delayed-dry contract is exact by
// construction; -Wfloat-equal is intentionally avoided here without weakening
// the check.
bool exactlyEqual(const float left, const float right)
{
    return std::memcmp(&left, &right, sizeof(float)) == 0;
}

std::vector<ScheduleEntry> buildSchedule(const Options& options)
{
    std::vector<ScheduleEntry> schedule;
    const auto& sizes = options.callbackSizes;
    if (options.mode == Mode::Soak)
    {
        // Contractual soak: 48 kHz / 2048 samples, representative signal, dry
        // path first (alignment-checked), wet path for the remaining duration.
        const auto half = options.durationSeconds / 2.0;
        schedule.push_back({ "soak-dry", sizes, half, false, true });
        schedule.push_back({ "soak-wet", sizes, half, true, false });
        return schedule;
    }

    const auto alternating = options.mode != Mode::Soak;
    const auto segmentCount = sizes.size() + (alternating ? std::size_t { 1 } : std::size_t { 0 });
    const auto perSegment = options.durationSeconds / static_cast<double>(segmentCount);
    const auto half = perSegment / 2.0;
    for (const auto size : sizes)
    {
        schedule.push_back({ "fixed-" + std::to_string(size) + "-dry",
                             { size },
                             half,
                             false,
                             true });
    }
    for (const auto size : sizes)
    {
        schedule.push_back({ "fixed-" + std::to_string(size) + "-wet",
                             { size },
                             half,
                             true,
                             false });
    }
    if (alternating)
        schedule.push_back({ "alternating-wet", sizes, perSegment, true, false });

    // A smoke run must still validate a meaningful post-latency dry window.
    // Complete-cycle rounding below already guarantees at least one full call
    // for every fixed size and one full alternating callback cycle.
    if (options.mode == Mode::Smoke && !schedule.empty())
        schedule.front().minimumSamples = transportLatency + *std::max_element(sizes.begin(), sizes.end());

    return schedule;
}

PacedRunResult runPacedSchedule(rave::RaveAudioEngine& engine,
                                const Options& options,
                                const std::vector<ScheduleEntry>& schedule,
                                const bool emitHeartbeats,
                                const double heartbeatIntervalSeconds,
                                const std::function<void(const PacedRunResult&)>& heartbeat)
{
    PacedRunResult run;
    const auto maximumCallback = *std::max_element(options.callbackSizes.begin(),
                                                   options.callbackSizes.end());
    std::vector<float> input(maximumCallback);
    std::vector<float> evicted(maximumCallback);
    std::vector<std::vector<float>> output(static_cast<std::size_t>(options.channels),
                                           std::vector<float>(maximumCallback));
    DeterministicNoise noise(options.signalSeed, 0.25f);
    DelayedDryOracle oracle;
    const auto runStart = std::chrono::steady_clock::now();
    auto nextDeadline = runStart;
    auto lastHeartbeat = runStart;

    for (const auto& entry : schedule)
    {
        SegmentResult segment;
        segment.name = entry.name;
        segment.callbackSizes = entry.callbackCycle;
        segment.wet = entry.wet;
        segment.alignmentChecked = entry.alignmentChecked;
        TimingRecorder recorder(entry.alignmentChecked || entry.callbackCycle.size() == 1 ? 2000000 : 500000);

        engine.setDryWet(entry.wet ? 1.0f : 0.0f);

        std::size_t cycleIndex = 0;
        const auto segmentSamples = completeCallbackCycleSamples(entry.pacedSeconds,
                                                                 options.sampleRate,
                                                                 entry.callbackCycle,
                                                                 entry.minimumSamples);
        std::uint64_t renderedInSegment = 0;

        while (renderedInSegment < segmentSamples)
        {
            const auto callbackSize = entry.callbackCycle[cycleIndex % entry.callbackCycle.size()];
            ++cycleIndex;
            const auto samples = static_cast<int>(callbackSize);
            const auto sampleCount = static_cast<std::size_t>(samples);

            noise.fill(input.data(), sampleCount);
            for (int sample = 0; sample < samples; ++sample)
                evicted[static_cast<std::size_t>(sample)] =
                    oracle.push(input[static_cast<std::size_t>(sample)]);

            std::vector<const float*> inputPointers(static_cast<std::size_t>(options.channels),
                                                    input.data());
            std::vector<float*> outputPointers(static_cast<std::size_t>(options.channels));
            for (int channel = 0; channel < options.channels; ++channel)
                outputPointers[static_cast<std::size_t>(channel)] =
                    output[static_cast<std::size_t>(channel)].data();

            // Only the engine processing call is measured: stimulus generation
            // and the real-time pacing sleep stay outside the timing window.
            const auto callStart = std::chrono::steady_clock::now();
            engine.processAudio(inputPointers.data(),
                                options.channels,
                                outputPointers.data(),
                                options.channels,
                                samples);
            const auto callEnd = std::chrono::steady_clock::now();
            recorder.add(nanosecondsToMs(callEnd - callStart));
            ++segment.calls;

            for (int channel = 0; channel < options.channels; ++channel)
            {
                const auto& channelOutput = output[static_cast<std::size_t>(channel)];
                for (std::size_t sample = 0; sample < sampleCount; ++sample)
                {
                    if (!std::isfinite(channelOutput[sample]))
                    {
                        ++segment.nonFiniteSamples;
                        segment.outputFinite = false;
                    }
                }
            }

            if (segment.alignmentChecked)
            {
                const auto segmentStartSample = run.totalSamplesPerChannel + renderedInSegment;
                for (int sample = 0; sample < samples; ++sample)
                {
                    const auto globalSample = segmentStartSample + static_cast<std::uint64_t>(sample);
                    if (globalSample < transportLatency)
                        continue; // the transport emits alignment silence before its latency
                    const auto expected = evicted[static_cast<std::size_t>(sample)];
                    for (int channel = 0; channel < options.channels; ++channel)
                    {
                        ++segment.alignmentComparedSamples;
                        if (!exactlyEqual(
                                output[static_cast<std::size_t>(channel)]
                                      [static_cast<std::size_t>(sample)],
                                expected))
                            ++segment.alignmentMismatches;
                    }
                }
            }

            renderedInSegment += static_cast<std::uint64_t>(samples);

            const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(static_cast<double>(samples) / options.sampleRate));
            nextDeadline += period;
            const auto now = std::chrono::steady_clock::now();
            if (nextDeadline > now)
            {
                std::this_thread::sleep_until(nextDeadline);
            }
            else if (now - nextDeadline > std::chrono::milliseconds(1))
            {
                ++segment.pacingLags;
            }

            if (emitHeartbeats && std::chrono::steady_clock::now() - lastHeartbeat
                > std::chrono::duration<double>(heartbeatIntervalSeconds))
            {
                lastHeartbeat = std::chrono::steady_clock::now();
                run.totalPacedAudioSeconds =
                    static_cast<double>(run.totalSamplesPerChannel + renderedInSegment)
                    / options.sampleRate;
                heartbeat(run);
            }
        }

        segment.pacedAudioSeconds = static_cast<double>(renderedInSegment) / options.sampleRate;
        segment.timing = recorder.summarize();
        run.totalCalls += segment.calls;
        run.totalSamplesPerChannel += renderedInSegment;
        run.alignmentComparedSamples += segment.alignmentComparedSamples;
        run.alignmentMismatches += segment.alignmentMismatches;
        run.outputFinite = run.outputFinite && segment.outputFinite;
        run.segments.push_back(std::move(segment));
    }

    const auto runEnd = std::chrono::steady_clock::now();
    run.totalPacedAudioSeconds =
        static_cast<double>(run.totalSamplesPerChannel) / options.sampleRate;
    run.wallClockSeconds = std::chrono::duration<double>(runEnd - runStart).count();
    return run;
}

// ---------------------------------------------------------------------------
// Direct inference benchmark (measured independently of the engine).
// ---------------------------------------------------------------------------
struct InferenceBenchResult
{
    std::uint64_t frames = 0;
    std::uint64_t failures = 0;
    std::uint64_t nonFiniteSamples = 0;
    bool outputFinite = true;
    TimingSummary timing;
};

InferenceBenchResult runInferenceBenchmark(rave::ModelBackend& backend,
                                           const Options& options,
                                           const bool latentPath,
                                           DeterministicNoise& noise,
                                           const std::size_t latentControlCount)
{
    InferenceBenchResult result;
    result.frames = static_cast<std::uint64_t>(options.inferenceFrames);
    TimingRecorder recorder(static_cast<std::size_t>(options.inferenceFrames));
    std::vector<float> input(inferenceQuantum);
    std::vector<float> output(inferenceQuantum);
    std::vector<float> latentControls(latentPath ? latentControlCount : std::size_t { 0 }, 0.0f);

    // Untimed warm-up so first-call graph setup does not pollute the
    // distribution (reset state is managed by the caller).
    for (int frame = 0; frame < 2; ++frame)
    {
        noise.fill(input.data(), input.size());
        static_cast<void>(backend.process(input, latentControls, output));
    }

    for (int frame = 0; frame < options.inferenceFrames; ++frame)
    {
        noise.fill(input.data(), input.size());
        const auto callStart = std::chrono::steady_clock::now();
        const auto processed = backend.process(input, latentControls, output);
        const auto callEnd = std::chrono::steady_clock::now();
        if (!processed)
        {
            ++result.failures;
            result.outputFinite = false;
            continue;
        }
        for (const auto sample : output)
        {
            if (!std::isfinite(sample))
            {
                ++result.nonFiniteSamples;
                result.outputFinite = false;
            }
        }
        recorder.add(nanosecondsToMs(callEnd - callStart));
    }
    result.timing = recorder.summarize();
    return result;
}

// ---------------------------------------------------------------------------
// Deterministic overload/recovery probes through the documented engine test
// seam (publishResultForTesting), driven on a separate no-model engine while
// both the callback driver and the worker are stopped, honoring the seam's
// contract. This exercises commit-layer overload semantics: malformed, future,
// duplicate, and post-deadline results are rejected; the fallback renders
// aligned dry; recovery begins with a current sequence-valid block and fades
// in over the contractual 5 ms ramp; stale results are never replayed.
// ---------------------------------------------------------------------------
struct ProbeReport
{
    bool ran = false;
    bool allPassed = false;
    bool preLatencySilence = false;
    bool malformedRejected = false;
    bool futureRejected = false;
    bool validCommitted = false;
    bool duplicateRejected = false;
    bool postDeadlineRejected = false;
    bool fallbackNotReplayed = false;
    bool recoveryAccepted = false;
    bool recoveryFadeVerified = false;
    std::uint64_t alignmentErrors = 0;
    std::uint64_t lateResults = 0;
    std::uint64_t deadlineMisses = 0;
    std::string notes;
};

ProbeReport runOverloadRecoveryProbes()
{
    ProbeReport probe;
    probe.ran = true;

    rave::RaveAudioEngine engine;
    engine.prepare(48000.0, 64, 1);
    engine.setDryWet(1.0f); // the smoothing ramp settles long before the checked recovery window
    std::vector<float> input(64, 1.0f);
    std::vector<float> output(64, -1.0f);
    const float* inputs[] { input.data() };
    float* outputs[] { output.data() };
    const auto drive = [&](std::uint64_t samples, float* capturedOutput) {
        std::uint64_t driven = 0;
        while (driven < samples)
        {
            const auto chunk = static_cast<int>(std::min<std::uint64_t>(64, samples - driven));
            engine.processAudio(inputs, 1, outputs, 1, chunk);
            if (capturedOutput != nullptr)
                std::copy_n(output.begin(), static_cast<std::size_t>(chunk),
                            capturedOutput + driven);
            driven += static_cast<std::uint64_t>(chunk);
        }
    };

    // The transport must emit exactly `transportLatency` samples of alignment
    // silence before the delayed path starts rendering. Drive one submission
    // window first; frame 0's deadline (3856) has not passed, so the identity
    // probes below still run before their deadlines.
    std::vector<float> first(2048, -1.0f);
    drive(2048, first.data());
    bool silenceCorrect = true;
    for (const auto sample : first)
        if (!exactlyEqual(sample, 0.0f))
            silenceCorrect = false;

    std::vector<float> wet(inferenceQuantum, 3.0f);
    probe.malformedRejected = !engine.publishResultForTesting(wet.data(), wet.size() - 1, 0);
    probe.futureRejected = !engine.publishResultForTesting(wet.data(), wet.size(), 9);
    probe.validCommitted = engine.publishResultForTesting(wet.data(), wet.size(), 0);
    probe.duplicateRejected = !engine.publishResultForTesting(wet.data(), wet.size(), 0);

    // Finish the transport-silence window (frame 0 commits wet at 3856).
    std::vector<float> second(2048, -1.0f);
    drive(2048, second.data());
    for (const auto sample : second)
        if (!exactlyEqual(sample, 0.0f))
            silenceCorrect = false;
    probe.preLatencySilence = silenceCorrect
        && engine.runtimeTelemetry().latencySamples == transportLatency;

    // Drive to frame 1's commitment deadline (5904) without publishing it;
    // one more sample makes any frame-1 publication provably late.
    drive(5904 - transportLatency, nullptr);
    drive(1, nullptr);
    probe.postDeadlineRejected = !engine.publishResultForTesting(wet.data(), wet.size(), 1);

    // Recovery: frame 2 must first be submitted (its submission sample is
    // 6143) before it may be published, and it is still well before its
    // deadline (7952).
    drive(6144 - 5905, nullptr);
    probe.recoveryAccepted = engine.publishResultForTesting(wet.data(), wet.size(), 2);

    // Render frame 1's playback window (aligned dry fallback, never replayed
    // stale wet) followed by frame 2's 5 ms recovery fade-in to wet.
    constexpr std::uint64_t renderBase = 6144;
    std::vector<float> rendered(8500 - renderBase, 0.0f);
    drive(8500 - renderBase, rendered.data());
    const auto local = [](const std::uint64_t globalSample) {
        return static_cast<std::size_t>(globalSample - renderBase);
    };

    // Frame 1 plays at global samples 6144..8191; it committed dry.
    bool frameOneDry = true;
    for (auto index = local(6144) + 240; index < local(8192); ++index)
        if (!exactlyEqual(rendered[index], 1.0f))
            frameOneDry = false;
    probe.fallbackNotReplayed = frameOneDry;

    // Frame 2 plays from 8192: 240-sample dry-to-wet ramp, then full wet.
    bool fadeCorrect = true;
    for (std::uint64_t position = 0; position < 240; ++position)
    {
        const auto expected = 1.0f + 2.0f * static_cast<float>(position) / 240.0f;
        if (std::abs(rendered[local(8192 + position)] - expected) > 0.0001f)
            fadeCorrect = false;
    }
    for (auto index = local(8192 + 240); index < rendered.size(); ++index)
        if (std::abs(rendered[index] - 3.0f) > 0.0001f)
            fadeCorrect = false;
    probe.recoveryFadeVerified = fadeCorrect;

    const auto telemetry = engine.runtimeTelemetry();
    probe.alignmentErrors = telemetry.alignmentErrors;
    probe.lateResults = telemetry.lateResults;
    probe.deadlineMisses = telemetry.deadlineMisses;

    probe.allPassed = probe.preLatencySilence && probe.malformedRejected && probe.futureRejected
        && probe.validCommitted && probe.duplicateRejected && probe.postDeadlineRejected
        && probe.fallbackNotReplayed && probe.recoveryAccepted && probe.recoveryFadeVerified
        && probe.alignmentErrors == 3 && probe.lateResults == 1;
    return probe;
}

// ---------------------------------------------------------------------------
// Report rendering
// ---------------------------------------------------------------------------
void writeThresholdArray(JsonWriter& json, const std::vector<ThresholdResult>& thresholds)
{
    json.key("thresholds");
    json.beginArray();
    for (const auto& threshold : thresholds)
    {
        json.beginObject();
        json.key("id");
        json.value(threshold.id);
        json.key("scope");
        json.value(threshold.scope);
        json.key("status");
        json.value(threshold.status);
        json.key("detail");
        json.value(threshold.detail);
        json.endObject();
    }
    json.endArray();
}

void writeUnverifiedArray(JsonWriter& json)
{
    struct UnverifiedItem
    {
        const char* id;
        const char* reason;
    };
    static const std::vector<UnverifiedItem> items {
        { "realtime_host_xruns",
          "The offline harness drives RaveAudioEngine::processAudio directly; no audio device, "
          "driver, or host is involved, so host/device xruns cannot be observed here." },
        { "audio_device_io_latency",
          "Device input/output latency requires a real audio device and is not measurable offline." },
        { "impulse_wet_path_latency",
          "Wet-path impulse latency through the nonlinear real model is not measurable with an "
          "offline driver; the fixed 4096-sample transport latency is verified by contract instead." },
        { "perceptual_model_latency",
          "Perceptual latency is a human judgment and cannot be automated." },
        { "interactive_host_soak",
          "The 30-minute per-host interactive soak (Logic, REAPER, Ableton) requires manual host "
          "sessions." },
        { "standalone_real_device_changes",
          "Standalone audio-device changes and MIDI disconnect/reconnect require real hardware." },
        { "daw_scan_automation_recall",
          "VST3/AU scanning, editor-closed recall, automation, bypass, transport, and "
          "multiple-instance behavior in hosts require manual DAW sessions." },
        { "memory_under_interactive_host",
          "Peak RSS here covers the offline harness only; host-session memory growth needs the "
          "interactive soak." },
        { "realtime_scheduling_effects",
          "The offline harness runs at normal scheduling priority; callback wall times are upper "
          "bounds and are not realtime-priority measurements." }
    };
    json.key("unverified");
    json.beginArray();
    for (const auto& item : items)
    {
        json.beginObject();
        json.key("id");
        json.value(item.id);
        json.key("reason");
        json.value(item.reason);
        json.endObject();
    }
    json.endArray();
}
} // namespace

int main(const int argc, const char* const* argv)
{
    std::ios::sync_with_stdio(false);

    try
    {
        const auto parsed = parseOptions(argc, argv);
        if (!parsed.ok)
        {
            if (!parsed.error.empty())
            {
                std::cerr << "rave_qualification: " << parsed.error << "\n\n" << parsed.usage;
                return exitUsageError;
            }
            std::cout << parsed.usage;
            return exitPass;
        }
        const Options options = parsed.options;

        const auto helpRequested = [argc, argv] {
            for (int index = 1; index < argc; ++index)
                if (std::string(argv[index]) == "--help" || std::string(argv[index]) == "-h")
                    return true;
            return false;
        }();
        if (helpRequested)
        {
            std::cout << parsed.usage;
            return exitPass;
        }

        // ---- Input gating (fail closed, before anything loads) ----
        {
            std::ifstream probe(options.modelPath, std::ios::binary);
            if (!probe.good())
            {
                std::cerr << "rave_qualification: model file not found or unreadable: "
                          << options.modelPath << "\n";
                return exitUsageError;
            }
        }

        std::string modelHash;
        std::uintmax_t modelSize = 0;
        std::string hashError;
        std::cout << "hashing model (SHA-256)...\n";
        if (!sha256HexOfFile(options.modelPath, modelHash, modelSize, hashError))
        {
            std::cerr << "rave_qualification: " << hashError << "\n";
            return exitUsageError;
        }
        bool hashMatchedExpected = false;
        if (!options.expectedSha256.empty())
        {
            auto expected = options.expectedSha256;
            std::transform(expected.begin(), expected.end(), expected.begin(),
                           [](const unsigned char character) {
                               return static_cast<char>(std::tolower(character));
                           });
            hashMatchedExpected = expected == modelHash;
            if (!hashMatchedExpected)
            {
                std::cerr << "rave_qualification: SHA-256 mismatch; refusing to load.\n"
                          << "  expected: " << expected << "\n"
                          << "  actual:   " << modelHash << "\n"
                          << "TorchScript models are executable untrusted input; verify the\n"
                          << "documented hash before loading (see docs/QUALIFICATION.md).\n";
                return exitHashMismatch;
            }
        }

        // ---- Revision and environment identity ----
        const auto sourceDir = options.sourceDir.empty() ? std::string(RAVE_QUALIFICATION_SOURCE_DIR)
                                                         : options.sourceDir;
        const auto revision = captureRevision(sourceDir);

        // ---- Model load and offline qualification (existing production seams) ----
        auto backend = std::make_shared<rave::TorchScriptBackend>();
        std::string loadError;
        std::cout << "loading TorchScript backend...\n";
        if (!backend->load(options.modelPath, loadError))
        {
            std::cerr << "rave_qualification: model load failed: " << loadError << "\n";
            return exitModelFailure;
        }
        const auto modelSampleRate = backend->modelSampleRate();
        const auto latentDimensions = backend->latentDimensionCount();
        const auto inputChannels = backend->inputChannelCount();
        const auto outputChannels = backend->outputChannelCount();

        std::cout << "qualifying model at " << static_cast<long>(options.sampleRate) << " Hz / "
                  << inferenceQuantum << " samples...\n";
        const auto qualificationDiagnostic = rave::qualifyModelBackend(
            *backend, rave::ModelRuntimeConfiguration { options.sampleRate, inferenceQuantum });
        if (!qualificationDiagnostic.empty())
        {
            std::cerr << "rave_qualification: model qualification failed: "
                      << qualificationDiagnostic << "\n";
            return exitModelFailure;
        }

        // ---- Direct inference benchmark (independent of the engine) ----
        // The engine worker always snapshots the full latent control vector, so
        // a model with latent dimensions runs the production wet path as
        // encode -> zero offsets -> decode; measure that first, then the plain
        // forward path as a reference.
        const auto latentControlCount = latentDimensions;
        std::cout << "direct inference benchmark (" << options.inferenceFrames
                  << " frames on the engine production "
                  << (latentControlCount > 0 ? "encode/decode" : "forward") << " path, then "
                  << (latentControlCount > 0 ? "forward" : "encode/decode") << ")...\n";
        DeterministicNoise benchNoise(options.signalSeed + 1, 0.25f);
        static_cast<void>(backend->prepare(options.sampleRate, inferenceQuantum));
        std::string resetError;
        if (!backend->reset(resetError))
        {
            std::cerr << "rave_qualification: model reset failed before benchmark: " << resetError
                      << "\n";
            return exitModelFailure;
        }
        const auto latentBench = runInferenceBenchmark(*backend,
                                                       options,
                                                       latentControlCount > 0,
                                                       benchNoise,
                                                       latentControlCount);
        if (!backend->reset(resetError))
        {
            std::cerr << "rave_qualification: model reset failed between benchmarks: "
                      << resetError << "\n";
            return exitModelFailure;
        }
        const auto forwardBench = runInferenceBenchmark(*backend,
                                                        options,
                                                        false,
                                                        benchNoise,
                                                        0);
        if (!backend->reset(resetError))
        {
            std::cerr << "rave_qualification: model reset failed after benchmark: " << resetError
                      << "\n";
            return exitModelFailure;
        }

        const auto frameDurationMs = static_cast<double>(inferenceQuantum) / options.sampleRate
            * 1000.0;
        const auto inferenceP99LimitMs = frameDurationMs * inferenceP99FrameFraction;

        // ---- Shared engine paced run (production processing path) ----
        std::cout << "paced engine run (callback matrix, real-time pacing, timing excludes "
                     "pacing)...\n";
        rave::RaveAudioEngine engine;
        engine.prepare(options.sampleRate, inferenceQuantum, options.channels);
        std::string activationFailure;
        if (!engine.activateModelBackend(backend, &activationFailure))
        {
            std::cerr << "rave_qualification: engine activation failed: " << activationFailure
                      << "\n";
            return exitModelFailure;
        }
        const auto activationTelemetry = engine.runtimeTelemetry();
        const bool activationLatencyFixed = activationTelemetry.latencySamples == transportLatency;

        const auto schedule = buildSchedule(options);
        const auto heartbeatInterval = options.mode == Mode::Soak ? 30.0 : 0.0;
        const auto heartbeat = [&](const PacedRunResult& current) {
            const auto telemetry = engine.runtimeTelemetry();
            std::ostringstream line;
            line << "[rave-qualification progress] audioRendered=" << current.totalPacedAudioSeconds
                 << "s callbacks=" << current.totalCalls
                 << " deadlineMisses=" << telemetry.deadlineMisses
                 << " queueDrops=" << telemetry.queueDrops
                 << " processingErrors=" << telemetry.processingErrors
                 << " peakRssMb=" << peakRssMb() << "\n";
            std::cerr << line.str();
            std::cerr.flush();
        };
        const auto paced = runPacedSchedule(engine,
                                            options,
                                            schedule,
                                            options.mode == Mode::Soak,
                                            heartbeatInterval,
                                            heartbeat);

        const auto telemetryAfterRun = engine.runtimeTelemetry();
        const bool telemetryClean = telemetryAfterRun.deadlineMisses == 0
            && telemetryAfterRun.queueDrops == 0 && telemetryAfterRun.lateResults == 0
            && telemetryAfterRun.processingErrors == 0 && telemetryAfterRun.resetErrors == 0
            && telemetryAfterRun.alignmentErrors == 0;
        const bool latencyConstantDuringRun = activationLatencyFixed
            && telemetryAfterRun.latencySamples == transportLatency;
        const bool transportReady = telemetryAfterRun.transportReady;

        engine.release();
        const bool releasedNotUsable = !engine.hasUsableModel();
        const bool releasedLatencyStillFixed = engine.runtimeTelemetry().latencySamples
            == transportLatency;

        // ---- Deterministic seam-driven overload/recovery probes ----
        ProbeReport probes;
        if (options.mode != Mode::Soak)
            probes = runOverloadRecoveryProbes();

        // ---- Threshold evaluation (explicit; fail-closed on measured scope) ----
        std::vector<ThresholdResult> thresholds;

        thresholds.push_back(
            { "model_identity_hash",
              "measured",
              options.expectedSha256.empty()
                  ? "pass" // recorded identity; no expectation supplied
                  : (hashMatchedExpected ? "pass" : "fail"),
              options.expectedSha256.empty()
                  ? "no --expect-sha256 supplied; identity recorded without a gate"
                  : (hashMatchedExpected ? "file SHA-256 matched the expected digest"
                                         : "file SHA-256 did not match the expected digest") });

        thresholds.push_back(
            { "model_qualification_offline",
              "measured",
              qualificationDiagnostic.empty() ? "pass" : "fail",
              qualificationDiagnostic.empty()
                  ? "load + prepare + checked reset + warm-up passed at "
                    + std::to_string(static_cast<long>(options.sampleRate)) + " Hz / "
                    + std::to_string(inferenceQuantum) + " samples"
                  : qualificationDiagnostic });

        const auto inferenceThresholdPass = latentBench.outputFinite && latentBench.failures == 0
            && latentBench.timing.p99Ms <= inferenceP99LimitMs;
        thresholds.push_back(
            { "inference_p99_le_90pct_of_frame_duration",
              "measured",
              inferenceThresholdPass ? "pass" : "fail",
              "production latent path p99 " + std::to_string(latentBench.timing.p99Ms)
                  + " ms vs limit " + std::to_string(inferenceP99LimitMs) + " ms (frame duration "
                  + std::to_string(frameDurationMs) + " ms)" });

        thresholds.push_back(
            { "inference_output_finite",
              "measured",
              (latentBench.outputFinite && forwardBench.outputFinite) ? "pass" : "fail",
              "latent path non-finite " + std::to_string(latentBench.nonFiniteSamples)
                  + ", failures " + std::to_string(latentBench.failures) + "; forward path "
                  "non-finite " + std::to_string(forwardBench.nonFiniteSamples) + ", failures "
                  + std::to_string(forwardBench.failures) });

        bool callbackP99Pass = true;
        bool callbackMaxPass = true;
        std::string callbackP99Detail;
        std::string callbackMaxDetail;
        for (const auto& segment : paced.segments)
        {
            if (segment.callbackSizes.size() != 1)
                continue;
            const auto periodMs = static_cast<double>(segment.callbackSizes.front())
                / options.sampleRate * 1000.0;
            const auto p99Limit = periodMs * callbackP99PeriodFraction;
            const auto maxLimit = periodMs * callbackMaxPeriodFraction;
            const auto segmentPass = segment.outputFinite
                && segment.timing.p99Ms <= p99Limit;
            const auto segmentMaxPass = segment.outputFinite && segment.timing.maxMs <= maxLimit;
            callbackP99Pass = callbackP99Pass && segmentPass;
            callbackMaxPass = callbackMaxPass && segmentMaxPass;
            callbackP99Detail += (callbackP99Detail.empty() ? "" : "; ")
                + segment.name + ": p99 " + std::to_string(segment.timing.p99Ms) + "/"
                + std::to_string(p99Limit) + " ms";
            callbackMaxDetail += (callbackMaxDetail.empty() ? "" : "; ")
                + segment.name + ": max " + std::to_string(segment.timing.maxMs) + "/"
                + std::to_string(maxLimit) + " ms";
        }
        thresholds.push_back({ "callback_p99_le_25pct_of_period",
                               "measured",
                               callbackP99Pass ? "pass" : "fail",
                               callbackP99Detail });
        thresholds.push_back({ "callback_max_le_75pct_of_period",
                               "measured",
                               callbackMaxPass ? "pass" : "fail",
                               callbackMaxDetail });
        thresholds.push_back(
            { "callback_output_finite",
              "measured",
              paced.outputFinite ? "pass" : "fail",
              "non-finite samples rendered across the paced run: "
                  + std::to_string(std::accumulate(paced.segments.begin(),
                                                   paced.segments.end(),
                                                   std::uint64_t { 0 },
                                                   [](const std::uint64_t sum,
                                                      const SegmentResult& segment) {
                                                       return sum + segment.nonFiniteSamples;
                                                   })) });

        const bool alignmentVerified = paced.alignmentComparedSamples > 0
            && paced.alignmentMismatches == 0;
        thresholds.push_back(
            { "delayed_dry_alignment_exact",
              "measured",
              alignmentVerified ? "pass" : "fail",
              std::to_string(paced.alignmentMismatches) + " mismatches over "
                  + std::to_string(paced.alignmentComparedSamples)
                  + " compared samples (0% wet segments, exact float equality)" });

        thresholds.push_back(
            { "reported_latency_fixed_at_transport_contract",
              "measured",
              (latencyConstantDuringRun && releasedLatencyStillFixed) ? "pass" : "fail",
              "reported latency " + std::to_string(transportLatency)
                  + " samples held at activation, through the paced run, and after release" });

        thresholds.push_back(
            { "normal_load_telemetry_clean",
              "measured",
              telemetryClean ? "pass" : "fail",
              "deadlineMisses=" + std::to_string(telemetryAfterRun.deadlineMisses)
                  + " queueDrops=" + std::to_string(telemetryAfterRun.queueDrops)
                  + " lateResults=" + std::to_string(telemetryAfterRun.lateResults)
                  + " processingErrors=" + std::to_string(telemetryAfterRun.processingErrors)
                  + " resetErrors=" + std::to_string(telemetryAfterRun.resetErrors)
                  + " alignmentErrors=" + std::to_string(telemetryAfterRun.alignmentErrors) });

        if (probes.ran)
        {
            thresholds.push_back(
                { "overload_recovery_commit_semantics",
                  "measured",
                  probes.allPassed ? "pass" : "fail",
                  "deterministic publishResultForTesting probes on a stopped no-model engine "
                  "(see overloadRecoveryProbes in the report)" });
        }
        else
        {
            thresholds.push_back(
                { "overload_recovery_commit_semantics",
                  "measured",
                  "not-run",
                  "probe phase is skipped in soak mode to keep the soak uninterrupted" });
        }

        const auto verdictFailed = std::any_of(thresholds.begin(), thresholds.end(),
                                               [](const ThresholdResult& threshold) {
                                                   return threshold.scope == "measured"
                                                       && threshold.status == "fail";
                                               });
        const std::string verdict = verdictFailed ? "fail" : "pass";

        // ---- JSON report (stable schema rave-qualification/1) ----
        std::string jsonText;
        {
            JsonWriter json(jsonText);
            json.beginObject();
            json.key("schema");
            json.value("rave-qualification/1");
            json.key("tool");
            json.value("rave_qualification");
            json.key("generatedAtUtc");
            json.value(formatUtcIso8601(std::time(nullptr)));
            json.key("mode");
            json.value(options.mode == Mode::Smoke
                           ? "smoke"
                           : (options.mode == Mode::Soak ? "soak" : "benchmark"));
            json.key("scope");
            json.value("offline non-realtime qualification; no audio device, host, or DAW involved");

            json.key("revision");
            json.beginObject();
            json.key("gitAvailable");
            json.value(revision.gitAvailable);
            json.key("commit");
            json.value(revision.commit);
            json.key("dirty");
            json.value(revision.dirty);
            json.key("dirtyEntries");
            json.valueUnsigned(revision.dirtyEntries);
            json.key("sourceDir");
            json.value(revision.sourceDir);
            json.endObject();

            json.key("environment");
            json.beginObject();
            json.key("computerName");
            json.value(juce::SystemStats::getComputerName().toStdString());
            json.key("operatingSystem");
            json.value(juce::SystemStats::getOperatingSystemName().toStdString());
            json.key("cpuModel");
            json.value(juce::SystemStats::getCpuModel().toStdString());
            json.key("cpuArchitecture");
            json.value(cpuArchitecture());
            json.key("logicalCores");
            json.valueUnsigned(std::thread::hardware_concurrency());
            json.key("systemLoadAverage1mAtReport");
            json.value(systemLoadAverage1m(), 2);
            json.key("memoryMb");
            json.value(static_cast<double>(juce::SystemStats::getMemorySizeInMegabytes()), 0);
            json.key("is64Bit");
            json.value(juce::SystemStats::isOperatingSystem64Bit());
            json.key("buildType");
            json.value(RAVE_QUALIFICATION_BUILD_TYPE);
            json.key("cxxCompiler");
            json.value(compilerDescription());
            json.key("libtorchVersion");
            json.value(RAVE_QUALIFICATION_LIBTORCH_VERSION);
            json.key("juceVersion");
            json.value(RAVE_QUALIFICATION_JUCE_VERSION);
            json.key("audioDevice");
            json.value("offline driver (no audio device)");
            json.endObject();

            json.key("model");
            json.beginObject();
            json.key("path");
            json.value(options.modelPath);
            json.key("absolutePath");
            json.value(juce::File::getCurrentWorkingDirectory()
                           .getChildFile(juce::String(options.modelPath))
                           .getFullPathName()
                           .toStdString());
            json.key("sizeBytes");
            json.valueUnsigned(static_cast<std::uint64_t>(modelSize));
            json.key("sha256");
            json.value(modelHash);
            if (!options.expectedSha256.empty())
            {
                json.key("expectedSha256");
                json.value(options.expectedSha256);
                json.key("hashMatchedExpected");
                json.value(hashMatchedExpected);
            }
            json.key("declaredSampleRate");
            json.value(modelSampleRate);
            json.key("latentDimensions");
            json.valueUnsigned(latentDimensions);
            json.key("inputChannels");
            json.valueUnsigned(inputChannels);
            json.key("outputChannels");
            json.valueUnsigned(outputChannels);
            json.key("qualificationDiagnostic");
            json.value(qualificationDiagnostic);
            json.endObject();

            json.key("runConfiguration");
            json.beginObject();
            json.key("sampleRate");
            json.value(options.sampleRate, 1);
            json.key("frameSamples");
            json.valueUnsigned(inferenceQuantum);
            json.key("frameDurationMs");
            json.value(frameDurationMs);
            json.key("reportedTransportLatencySamples");
            json.valueUnsigned(transportLatency);
            json.key("callbackSizes");
            json.beginArray();
            for (const auto size : options.callbackSizes)
                json.valueUnsigned(size);
            json.endArray();
            json.key("channels");
            json.value(options.channels);
            json.key("pacedRealtime");
            json.value(true);
            json.key("callbackTimingExcludesPacing");
            json.value(true);
            json.key("inferenceMeasuredIndependently");
            json.value(true);
            json.key("dryWetSchedule");
            json.value("0.0 (alignment-checked dry pass) then 1.0 (wet pass); alternating "
                       "segment wet");
            json.key("signal");
            json.value("deterministic LCG white noise, peak 0.25, seed "
                       + std::to_string(options.signalSeed));
            json.key("durationSecondsRequested");
            json.value(options.durationSeconds, 3);
            json.endObject();

            auto writeTiming = [&](JsonWriter& writer, const TimingSummary& timing) {
                writer.key("count");
                writer.valueUnsigned(timing.count);
                writer.key("p50Ms");
                writer.value(timing.p50Ms);
                writer.key("p95Ms");
                writer.value(timing.p95Ms);
                writer.key("p99Ms");
                writer.value(timing.p99Ms);
                writer.key("maxMs");
                writer.value(timing.maxMs);
                writer.key("meanMs");
                writer.value(timing.meanMs);
                writer.key("storedSamples");
                writer.valueUnsigned(timing.storedSamples);
                writer.key("reservoirEstimated");
                writer.value(timing.reservoirEstimated);
            };

            json.key("inference");
            json.beginObject();
            json.key("productionPath");
            json.value(latentControlCount > 0 ? "encode -> per-dimension offsets -> decode"
                                              : "forward");
            auto writeBench = [&](const char* name, const InferenceBenchResult& bench) {
                json.key(name);
                json.beginObject();
                writeTiming(json, bench.timing);
                json.key("failures");
                json.valueUnsigned(bench.failures);
                json.key("nonFiniteSamples");
                json.valueUnsigned(bench.nonFiniteSamples);
                json.key("outputFinite");
                json.value(bench.outputFinite);
                json.endObject();
            };
            writeBench("latentPath", latentBench);
            writeBench("forwardPath", forwardBench);
            json.key("frameDurationMs");
            json.value(frameDurationMs);
            json.key("p99LimitMs");
            json.value(inferenceP99LimitMs);
            json.key("threshold");
            json.value(inferenceThresholdPass ? "pass" : "fail");
            json.endObject();

            json.key("callbacks");
            json.beginObject();
            json.key("totalCalls");
            json.valueUnsigned(paced.totalCalls);
            json.key("pacedAudioSeconds");
            json.value(paced.totalPacedAudioSeconds, 3);
            json.key("wallClockSeconds");
            json.value(paced.wallClockSeconds, 3);
            json.key("segments");
            json.beginArray();
            for (const auto& segment : paced.segments)
            {
                json.beginObject();
                json.key("name");
                json.value(segment.name);
                json.key("callbackSizes");
                json.beginArray();
                for (const auto size : segment.callbackSizes)
                    json.valueUnsigned(size);
                json.endArray();
                json.key("wet");
                json.value(segment.wet);
                json.key("pacedAudioSeconds");
                json.value(segment.pacedAudioSeconds, 3);
                writeTiming(json, segment.timing);
                json.key("pacingLags");
                json.valueUnsigned(segment.pacingLags);
                json.key("nonFiniteSamples");
                json.valueUnsigned(segment.nonFiniteSamples);
                json.key("outputFinite");
                json.value(segment.outputFinite);
                if (segment.alignmentChecked)
                {
                    json.key("alignmentComparedSamples");
                    json.valueUnsigned(segment.alignmentComparedSamples);
                    json.key("alignmentMismatches");
                    json.valueUnsigned(segment.alignmentMismatches);
                }
                json.endObject();
            }
            json.endArray();
            json.endObject();

            json.key("latency");
            json.beginObject();
            json.key("reportedSamples");
            json.valueUnsigned(telemetryAfterRun.latencySamples);
            json.key("expectedSamples");
            json.valueUnsigned(transportLatency);
            json.key("constantThroughActivationRunAndRelease");
            json.value(latencyConstantDuringRun && releasedLatencyStillFixed);
            json.key("transportReady");
            json.value(transportReady);
            json.key("notUsableAfterRelease");
            json.value(releasedNotUsable);
            json.key("delayedDryAlignmentComparedSamples");
            json.valueUnsigned(paced.alignmentComparedSamples);
            json.key("delayedDryAlignmentMismatches");
            json.valueUnsigned(paced.alignmentMismatches);
            json.endObject();

            json.key("telemetry");
            json.beginObject();
            json.key("epoch");
            json.value("after the paced normal-load run, before release");
            json.key("latencySamples");
            json.valueUnsigned(telemetryAfterRun.latencySamples);
            json.key("transportReady");
            json.value(telemetryAfterRun.transportReady);
            json.key("deadlineMisses");
            json.valueUnsigned(telemetryAfterRun.deadlineMisses);
            json.key("queueDrops");
            json.valueUnsigned(telemetryAfterRun.queueDrops);
            json.key("lateResults");
            json.valueUnsigned(telemetryAfterRun.lateResults);
            json.key("processingErrors");
            json.valueUnsigned(telemetryAfterRun.processingErrors);
            json.key("resetErrors");
            json.valueUnsigned(telemetryAfterRun.resetErrors);
            json.key("alignmentErrors");
            json.valueUnsigned(telemetryAfterRun.alignmentErrors);
            json.key("allCountersZero");
            json.value(telemetryClean);
            json.endObject();

            json.key("overloadRecoveryProbes");
            json.beginObject();
            json.key("ran");
            json.value(probes.ran);
            json.key("engineSeam");
            json.value(probes.ran ? "RaveAudioEngine::publishResultForTesting on a stopped "
                                    "no-model engine (seam contract honored)"
                                  : "skipped in soak mode");
            json.key("preLatencySilence");
            json.value(probes.preLatencySilence);
            json.key("malformedSizeRejected");
            json.value(probes.malformedRejected);
            json.key("futureFrameRejected");
            json.value(probes.futureRejected);
            json.key("validResultCommitted");
            json.value(probes.validCommitted);
            json.key("duplicateRejected");
            json.value(probes.duplicateRejected);
            json.key("postDeadlineResultRejected");
            json.value(probes.postDeadlineRejected);
            json.key("missedFrameRenderedAlignedDryNotReplayed");
            json.value(probes.fallbackNotReplayed);
            json.key("recoveryResultAcceptedBeforeDeadline");
            json.value(probes.recoveryAccepted);
            json.key("recoveryFiveMsFadeVerified");
            json.value(probes.recoveryFadeVerified);
            json.key("alignmentErrors");
            json.valueUnsigned(probes.alignmentErrors);
            json.key("lateResults");
            json.valueUnsigned(probes.lateResults);
            json.key("deadlineMisses");
            json.valueUnsigned(probes.deadlineMisses);
            json.key("allPassed");
            json.value(probes.ran && probes.allPassed);
            json.endObject();

            json.key("memory");
            json.beginObject();
            json.key("model");
            json.value("fixed preallocated buffers; bounded timing reservoirs; no growth with "
                       "soak length");
            json.key("timingReservoirCapacitySamples");
            json.valueUnsigned(2000000);
            json.key("peakRssMb");
            json.value(peakRssMb(), 1);
            json.endObject();

            writeThresholdArray(json, thresholds);
            writeUnverifiedArray(json);

            json.key("verdict");
            json.beginObject();
            json.key("status");
            json.value(verdict);
            json.key("scope");
            json.value(verdict == "pass"
                           ? "offline automated measurements passed; host/device/xrun/DAW/"
                             "perceptual/manual and long-soak requirements remain unverified "
                             "and are NOT satisfied by this report"
                           : "one or more measured offline thresholds failed");
            json.endObject();
            json.endObject();
        }

        // ---- Human-readable summary ----
        {
            std::ostringstream human;
            human << "\n=== RAVE-05 offline qualification (rave_qualification) ===\n";
            human << "mode: "
                  << (options.mode == Mode::Smoke
                          ? "smoke"
                          : (options.mode == Mode::Soak ? "soak" : "benchmark"))
                  << "  schema: rave-qualification/1\n";
            human << "revision: " << revision.commit << (revision.dirty ? " (dirty" : " (clean")
                  << (revision.gitAvailable ? ")" : ", git unavailable)") << "\n";
            human << "model: " << options.modelPath << "\n";
            human << "  sha256: " << modelHash << " (" << modelSize << " bytes)\n";
            if (!options.expectedSha256.empty())
                human << "  expected: " << options.expectedSha256 << " -> "
                      << (hashMatchedExpected ? "MATCH" : "MISMATCH") << "\n";
            human << "  declared rate: " << modelSampleRate << " Hz, latents: " << latentDimensions
                  << ", channels: " << inputChannels << "/" << outputChannels << "\n";
            human << "run: " << static_cast<long>(options.sampleRate) << " Hz, frame "
                  << inferenceQuantum << " samples, " << options.channels
                  << " channel(s), paced " << paced.totalPacedAudioSeconds << " s audio in "
                  << paced.wallClockSeconds << " s wall, load1m=" << systemLoadAverage1m()
                  << "\n";
            human << "\n-- direct inference (independent of engine) --\n";
            human << "  latent(production): count=" << latentBench.timing.count
                  << " p50=" << latentBench.timing.p50Ms << " p95=" << latentBench.timing.p95Ms
                  << " p99=" << latentBench.timing.p99Ms << " max=" << latentBench.timing.maxMs
                  << " ms (limit p99<=" << inferenceP99LimitMs << " ms) -> "
                  << (inferenceThresholdPass ? "PASS" : "FAIL") << "\n";
            human << "  forward(reference):  p99=" << forwardBench.timing.p99Ms
                  << " max=" << forwardBench.timing.maxMs << " ms, finite="
                  << (forwardBench.outputFinite ? "yes" : "no") << "\n";
            human << "\n-- callback matrix (timing excludes pacing) --\n";
            for (const auto& segment : paced.segments)
            {
                human << "  " << segment.name << ": calls=" << segment.timing.count
                      << " p50=" << segment.timing.p50Ms << " p95=" << segment.timing.p95Ms
                      << " p99=" << segment.timing.p99Ms << " max=" << segment.timing.maxMs
                      << " ms lags=" << segment.pacingLags << " finite="
                      << (segment.outputFinite ? "yes" : "no");
                if (segment.callbackSizes.size() == 1)
                {
                    const auto periodMs = static_cast<double>(segment.callbackSizes.front())
                        / options.sampleRate * 1000.0;
                    human << "  [p99<=" << periodMs * callbackP99PeriodFraction << " max<="
                          << periodMs * callbackMaxPeriodFraction << " ms]";
                }
                human << "\n";
            }
            human << "\n-- latency & telemetry --\n";
            human << "  reported latency: " << telemetryAfterRun.latencySamples << " samples ("
                  << (latencyConstantDuringRun && releasedLatencyStillFixed
                          ? "constant through activation/run/release"
                          : "NOT constant")
                  << "), transportReady=" << (transportReady ? "yes" : "no") << "\n";
            human << "  delayed-dry alignment: " << paced.alignmentMismatches << " mismatches / "
                  << paced.alignmentComparedSamples << " compared samples\n";
            human << "  telemetry: misses=" << telemetryAfterRun.deadlineMisses
                  << " queueDrops=" << telemetryAfterRun.queueDrops
                  << " late=" << telemetryAfterRun.lateResults
                  << " processing=" << telemetryAfterRun.processingErrors
                  << " reset=" << telemetryAfterRun.resetErrors
                  << " alignment=" << telemetryAfterRun.alignmentErrors
                  << (telemetryClean ? "  -> clean" : "  -> NOT clean") << "\n";
            if (probes.ran)
                human << "  seam overload/recovery probes: "
                      << (probes.allPassed ? "all passed" : "FAILED")
                      << " (alignmentErrors=" << probes.alignmentErrors
                      << " late=" << probes.lateResults << ")\n";
            human << "\n-- thresholds --\n";
            for (const auto& threshold : thresholds)
                human << "  [" << threshold.status << "] " << threshold.id << " ("
                      << threshold.scope << ") " << threshold.detail << "\n";
            human << "\nverdict: " << verdict << "\n";
            human << "scope: offline automated measurements only. Host/device/xrun/DAW/"
                     "perceptual and manual checklist items remain unverified by this run and\n"
                     "are listed in the JSON report (\"unverified\").\n";
            std::cout << human.str();
        }

        // ---- Emit the machine-readable report ----
        if (!options.jsonPath.empty())
        {
            std::ofstream jsonFile(options.jsonPath, std::ios::binary);
            if (!jsonFile)
            {
                std::cerr << "rave_qualification: cannot write JSON report to: "
                          << options.jsonPath << "\n";
                return exitUsageError;
            }
            jsonFile << jsonText << "\n";
            std::cout << "JSON report written to: " << options.jsonPath << "\n";
        }
        else
        {
            std::cout << jsonText << "\n";
        }

        return verdictFailed ? exitThresholdFailure : exitPass;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "rave_qualification: unexpected error: " << exception.what() << "\n";
        return exitUnexpectedError;
    }
    catch (...)
    {
        std::cerr << "rave_qualification: unexpected unknown error\n";
        return exitUnexpectedError;
    }
}
