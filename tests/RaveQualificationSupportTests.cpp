// Unit tests for the RAVE-05 qualification harness support logic: SHA-256,
// bounded timing statistics, the deterministic stimulus, JSON escaping, CLI
// validation, and UTC formatting. These are fast, deterministic, and safe to
// run in ordinary CTest. The long soak harness itself is intentionally NOT
// registered with CTest (see docs/QUALIFICATION.md).

#include "qualification/QualificationSupport.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
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

void testSha256KnownVectors()
{
    using ravequalification::Sha256;

    Sha256 empty;
    require(Sha256::hex(empty.finish())
                == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "SHA-256 of the empty string matches the known vector");

    Sha256 abc;
    const std::string message = "abc";
    abc.update(reinterpret_cast<const std::uint8_t*>(message.data()), message.size());
    require(Sha256::hex(abc.finish())
                == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 of \"abc\" matches the known vector");

    // Two-block boundary crossing plus the classic million-'a' vector.
    const std::string millionA(1000000, 'a');
    Sha256 big;
    for (std::size_t offset = 0; offset < millionA.size(); offset += 997)
    {
        const auto chunk = millionA.substr(offset, 997);
        big.update(reinterpret_cast<const std::uint8_t*>(chunk.data()), chunk.size());
    }
    require(Sha256::hex(big.finish())
                == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
            "SHA-256 of one million 'a' characters matches (odd chunk sizes)");

    Sha256 exactBlock;
    const std::string exact(64, 'x');
    exactBlock.update(reinterpret_cast<const std::uint8_t*>(exact.data()), exact.size());
    Sha256 splitBlock;
    splitBlock.update(reinterpret_cast<const std::uint8_t*>(exact.data()), 1);
    splitBlock.update(reinterpret_cast<const std::uint8_t*>(exact.data() + 1), 63);
    require(Sha256::hex(exactBlock.finish()) == Sha256::hex(splitBlock.finish()),
            "block-aligned and split updates produce identical digests");

    require(ravequalification::isValidSha256Hex(
                "a12ad61a2b0b5ee2329a72993bd94386a571600b37dd23feaa0a404940468d68"),
            "valid digest accepted");
    require(!ravequalification::isValidSha256Hex("a12ad61a"), "short digest rejected");
    require(!ravequalification::isValidSha256Hex(std::string(63, 'a') + "g"),
            "non-hex digest rejected");
}

void testTimingRecorder()
{
    ravequalification::TimingRecorder recorder(1000);
    for (int index = 1; index <= 100; ++index)
        recorder.add(static_cast<double>(index)); // ms
    const auto summary = recorder.summarize();
    require(summary.count == 100, "summary counts every sample");
    require(!summary.reservoirEstimated, "within capacity is not reservoir-estimated");
    require(summary.storedSamples == 100, "all samples stored within capacity");
    require(std::abs(summary.p50Ms - 50.5) < 1e-9, "p50 of 1..100 is 50.5");
    require(std::abs(summary.p95Ms - 95.05) < 1e-9, "p95 of 1..100 is 95.05");
    require(std::abs(summary.p99Ms - 99.01) < 1e-9, "p99 of 1..100 is 99.01");
    require(std::abs(summary.maxMs - 100.0) < 1e-9, "max is exact");
    require(std::abs(summary.meanMs - 50.5) < 1e-9, "mean is exact");

    // Beyond capacity: bounded storage, exact max/mean, reservoir estimation
    // flagged, and percentiles still within the observed range.
    ravequalification::TimingRecorder bounded(64);
    double runningSum = 0.0;
    for (int index = 0; index < 6400; ++index)
    {
        const auto value = 1.0 + static_cast<double>(index % 97) * 0.5;
        bounded.add(value);
        runningSum += value;
    }
    const auto boundedSummary = bounded.summarize();
    require(boundedSummary.count == 6400, "reservoir counts every sample");
    require(boundedSummary.storedSamples == 64, "storage stays bounded at capacity");
    require(boundedSummary.reservoirEstimated, "beyond capacity is reservoir-estimated");
    require(std::abs(boundedSummary.maxMs - (1.0 + 96 * 0.5)) < 1e-9, "reservoir max is exact");
    require(std::abs(boundedSummary.meanMs - runningSum / 6400.0) < 1e-9,
            "reservoir mean is exact over all samples");
    require(boundedSummary.p50Ms >= 1.0 && boundedSummary.p50Ms <= 49.0,
            "reservoir p50 stays in the plausible range");
}

void testDeterministicNoise()
{
    ravequalification::DeterministicNoise first(12345, 0.25f);
    ravequalification::DeterministicNoise second(12345, 0.25f);
    for (int index = 0; index < 1000; ++index)
    {
        const auto left = first.nextSample();
        const auto right = second.nextSample();
        require(std::memcmp(&left, &right, sizeof(float)) == 0,
                "noise is deterministic per seed");
    }

    ravequalification::DeterministicNoise bounded(12345, 0.25f);
    for (int index = 0; index < 10000; ++index)
    {
        const auto sample = bounded.nextSample();
        require(std::isfinite(sample), "noise samples are finite");
        require(sample >= -0.25f && sample <= 0.25f, "noise respects the peak amplitude");
    }
}

void testJsonWriterEscaping()
{
    std::string sink;
    ravequalification::JsonWriter json(sink);
    json.beginObject();
    json.key("quote");
    json.value("he said \"hi\"\\done\n\t\b\f\r");
    json.key("control");
    json.value(std::string("\x01\x1f"));
    json.key("number");
    json.value(42);
    json.key("unsignedBig");
    json.valueUnsigned(18446744073709551615ull);
    json.key("flag");
    json.value(true);
    json.key("list");
    json.beginArray();
    json.value(1.5, 3);
    json.value("two");
    json.endArray();
    json.key("nested");
    json.beginObject();
    json.key("inner");
    json.value("value");
    json.endObject();
    json.endObject();

    require(sink == "{\"quote\":\"he said \\\"hi\\\"\\\\done\\n\\t\\b\\f\\r\","
                    "\"control\":\"\\u0001\\u001f\",\"number\":42,"
                    "\"unsignedBig\":18446744073709551615,\"flag\":true,"
                    "\"list\":[1.500,\"two\"],\"nested\":{\"inner\":\"value\"}}",
            "JSON writer output is exact including escapes");
}

void testCompleteCallbackCycleSamples()
{
    using ravequalification::completeCallbackCycleSamples;

    require(completeCallbackCycleSamples(0.001, 48000.0, { 2048 }) == 2048,
            "short fixed-size segment rounds up to one complete callback");
    require(completeCallbackCycleSamples(0.2 / 7.0, 48000.0,
                                         { 64, 128, 256, 512, 1024, 2048 })
                == 4032,
            "short alternating segment contains one complete callback cycle");
    require(completeCallbackCycleSamples(0.001, 48000.0, { 64 }, 4096 + 2048)
                == 6144,
            "alignment segment covers transport latency plus verification window");
    require(completeCallbackCycleSamples(0.01, 48000.0, {}) == 0,
            "empty callback cycle is rejected safely");
}

void testOptionParsing()
{
    const char* missing[] { "rave_qualification" };
    auto result = ravequalification::parseOptions(1, missing);
    require(!result.ok && result.error.find("--model") != std::string::npos,
            "missing --model is rejected with an actionable message");

    const char* absentFile[] { "rave_qualification", "--model",
                               "/path/that/does/not/exist.ts" };
    result = ravequalification::parseOptions(3, absentFile);
    require(result.ok, "a syntactically valid absent path still parses (existence checked later)");

    const char* badHash[] { "rave_qualification", "--model", "m.ts", "--expect-sha256", "abc" };
    result = ravequalification::parseOptions(5, badHash);
    require(!result.ok && result.error.find("64 hexadecimal") != std::string::npos,
            "short digest rejected");

    const char* badRate[] { "rave_qualification", "--model", "m.ts", "--sample-rate", "500000" };
    result = ravequalification::parseOptions(5, badRate);
    require(!result.ok && result.error.find("409600") != std::string::npos,
            "sample rate above the transport fade limit is rejected");

    const char* badSizes[] { "rave_qualification", "--model", "m.ts",
                             "--callback-sizes", "64,0,128" };
    result = ravequalification::parseOptions(5, badSizes);
    require(!result.ok && result.error.find("callback") != std::string::npos,
            "zero callback size rejected");

    const char* unknown[] { "rave_qualification", "--nonsense" };
    result = ravequalification::parseOptions(2, unknown);
    require(!result.ok && result.error.find("--nonsense") != std::string::npos,
            "unknown arguments rejected");

    const char* smoke[] { "rave_qualification", "--mode", "smoke", "--model", "m.ts" };
    result = ravequalification::parseOptions(5, smoke);
    require(result.ok, "smoke options parse");
    require(result.options.mode == ravequalification::Mode::Smoke, "smoke mode recorded");
    require(std::abs(result.options.durationSeconds - 0.2) < 1e-9,
            "smoke default per-segment duration is short");
    require(result.options.inferenceFrames == 128, "smoke default inference frames are small");
    require(result.options.callbackSizes.size() == 6, "smoke default covers the callback matrix");

    const char* soak[] { "rave_qualification", "--mode", "soak", "--model", "m.ts" };
    result = ravequalification::parseOptions(5, soak);
    require(result.ok, "soak options parse");
    require(std::abs(result.options.durationSeconds - 7200.0) < 1e-9,
            "soak default duration is the required 2 hours");
    require(result.options.callbackSizes == std::vector<std::size_t> { 2048 },
            "soak default is the contractual 2048-sample callback");

    const char* explicitSizes[] { "rave_qualification", "--model", "m.ts",
                                  "--callback-sizes", "512,64,512,128" };
    result = ravequalification::parseOptions(5, explicitSizes);
    require(result.ok, "explicit callback sizes parse");
    require((result.options.callbackSizes == std::vector<std::size_t> { 64, 128, 512 }),
            "callback sizes are sorted and de-duplicated");
}

void testUtcFormatting()
{
    // 2026-02-14T00:00:00Z is 1771027200 epoch seconds.
    require(ravequalification::formatUtcIso8601(1771027200) == "2026-02-14T00:00:00Z",
            "UTC ISO-8601 formatting is exact");
}
} // namespace

int main()
{
    testSha256KnownVectors();
    testTimingRecorder();
    testDeterministicNoise();
    testJsonWriterEscaping();
    testCompleteCallbackCycleSamples();
    testOptionParsing();
    testUtcFormatting();
    std::cout << "rave_qualification_support_tests passed\n";
    return EXIT_SUCCESS;
}
