#pragma once

// Shared support logic for the RAVE-05 offline qualification harness
// (tools/qualification/RaveQualificationMain.cpp). Pure standard-library code
// with no JUCE, LibTorch, or engine dependencies, so it can be unit-tested
// independently of the model runtime.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <ios>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ravequalification
{
// ============================================================================
// SHA-256 (FIPS 180-4), used to hash-gate the local model fixture before the
// backend is allowed to load it. TorchScript models are executable untrusted
// input, so verification happens before torch::jit::load.
// ============================================================================
class Sha256
{
public:
    void update(const std::uint8_t* data, std::size_t size)
    {
        messageLength += size;
        while (size > 0)
        {
            if (bufferUsed == 0 && size >= blockSize)
            {
                transform(data);
                data += blockSize;
                size -= blockSize;
                continue;
            }
            const auto take = std::min(size, static_cast<std::size_t>(blockSize) - bufferUsed);
            std::memcpy(buffer.data() + bufferUsed, data, take);
            bufferUsed += take;
            data += take;
            size -= take;
            if (bufferUsed == blockSize)
            {
                transform(buffer.data());
                bufferUsed = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> finish()
    {
        const auto bits = messageLength * 8;
        const std::uint8_t one = 0x80;
        updateNoLength(&one, 1);
        const std::uint8_t zero = 0x00;
        while (bufferUsed != 56)
            updateNoLength(&zero, 1);
        std::array<std::uint8_t, 8> length {};
        for (std::size_t index = 0; index < 8; ++index)
            length[index] = static_cast<std::uint8_t>(bits >> (56 - 8 * index));
        updateNoLength(length.data(), length.size());

        std::array<std::uint8_t, 32> digest {};
        for (std::size_t index = 0; index < 8; ++index)
        {
            digest[4 * index + 0] = static_cast<std::uint8_t>(state[index] >> 24);
            digest[4 * index + 1] = static_cast<std::uint8_t>(state[index] >> 16);
            digest[4 * index + 2] = static_cast<std::uint8_t>(state[index] >> 8);
            digest[4 * index + 3] = static_cast<std::uint8_t>(state[index]);
        }
        return digest;
    }

    static std::string hex(const std::array<std::uint8_t, 32>& digest)
    {
        static constexpr char digits[] = "0123456789abcdef";
        std::string out;
        out.reserve(digest.size() * 2);
        for (const auto byte : digest)
        {
            out.push_back(digits[byte >> 4]);
            out.push_back(digits[byte & 0x0F]);
        }
        return out;
    }

private:
    void updateNoLength(const std::uint8_t* data, std::size_t size)
    {
        // Identical to update() but without accumulating messageLength, used
        // internally by finish() so padding does not change the length.
        while (size > 0)
        {
            if (bufferUsed == 0 && size >= blockSize)
            {
                transform(data);
                data += blockSize;
                size -= blockSize;
                continue;
            }
            const auto take = std::min(size, static_cast<std::size_t>(blockSize) - bufferUsed);
            std::memcpy(buffer.data() + bufferUsed, data, take);
            bufferUsed += take;
            data += take;
            size -= take;
            if (bufferUsed == blockSize)
            {
                transform(buffer.data());
                bufferUsed = 0;
            }
        }
    }

    static std::uint32_t rotr(std::uint32_t value, unsigned count) noexcept
    {
        return (value >> count) | (value << (32 - count));
    }

    void transform(const std::uint8_t* block)
    {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        std::array<std::uint32_t, 64> w {};
        for (std::size_t index = 0; index < 16; ++index)
            w[index] = (static_cast<std::uint32_t>(block[4 * index]) << 24)
                | (static_cast<std::uint32_t>(block[4 * index + 1]) << 16)
                | (static_cast<std::uint32_t>(block[4 * index + 2]) << 8)
                | static_cast<std::uint32_t>(block[4 * index + 3]);
        for (std::size_t index = 16; index < 64; ++index)
        {
            const auto s0 = rotr(w[index - 15], 7) ^ rotr(w[index - 15], 18) ^ (w[index - 15] >> 3);
            const auto s1 = rotr(w[index - 2], 17) ^ rotr(w[index - 2], 19) ^ (w[index - 2] >> 10);
            w[index] = w[index - 16] + s0 + w[index - 7] + s1;
        }

        auto a = state[0];
        auto b = state[1];
        auto c = state[2];
        auto d = state[3];
        auto e = state[4];
        auto f = state[5];
        auto g = state[6];
        auto h = state[7];

        for (std::size_t index = 0; index < 64; ++index)
        {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto ch = (e & f) ^ (~e & g);
            const auto temp1 = h + s1 + ch + k[index] + w[index];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    static constexpr std::size_t blockSize = 64;
    std::array<std::uint32_t, 8> state {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    std::array<std::uint8_t, blockSize> buffer {};
    std::size_t bufferUsed = 0;
    std::uint64_t messageLength = 0;
};

// Streaming SHA-256 of a file; returns false with a message on I/O failure.
inline bool sha256HexOfFile(const std::string& path,
                            std::string& hexOut,
                            std::uintmax_t& sizeOut,
                            std::string& error)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        error = "cannot open model file for reading: " + path;
        return false;
    }
    file.seekg(0, std::ios::end);
    const auto end = file.tellg();
    if (end < 0)
    {
        error = "cannot determine model file size: " + path;
        return false;
    }
    sizeOut = static_cast<std::uintmax_t>(end);
    file.seekg(0, std::ios::beg);

    Sha256 hash;
    std::vector<char> chunk(1024 * 1024);
    while (file)
    {
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto read = file.gcount();
        if (read > 0)
            hash.update(reinterpret_cast<const std::uint8_t*>(chunk.data()),
                        static_cast<std::size_t>(read));
    }
    if (file.bad())
    {
        error = "I/O error while reading model file: " + path;
        return false;
    }
    hexOut = Sha256::hex(hash.finish());
    return true;
}

inline bool isValidSha256Hex(std::string_view value)
{
    if (value.size() != 64)
        return false;
    return std::all_of(value.begin(), value.end(), [](const char character) {
        const auto lowered = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
        return (lowered >= '0' && lowered <= '9') || (lowered >= 'a' && lowered <= 'f');
    });
}

// ============================================================================
// Bounded-memory timing statistics. Durations are stored raw until the
// capacity is reached; beyond that a fixed-seed reservoir sample (Vitter R)
// keeps memory bounded while percentile estimates stay stable. The maximum
// and mean are always exact over every observed sample.
// ============================================================================
struct TimingSummary
{
    std::uint64_t count = 0;
    double p50Ms = 0.0;
    double p95Ms = 0.0;
    double p99Ms = 0.0;
    double maxMs = 0.0;
    double meanMs = 0.0;
    std::size_t storedSamples = 0;
    bool reservoirEstimated = false;
};

class TimingRecorder
{
public:
    explicit TimingRecorder(const std::size_t capacitySamples)
        : capacity(std::max<std::size_t>(capacitySamples, 1)), random(0x5EED1234u)
    {
        samples.reserve(this->capacity);
    }

    void add(double milliseconds)
    {
        ++seen;
        totalMs += static_cast<long double>(milliseconds);
        maxMs = std::max(maxMs, milliseconds);
        if (samples.size() < capacity)
        {
            samples.push_back(milliseconds);
            return;
        }
        // Vitter's Algorithm R: keep item with probability capacity/seen.
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        if (uniform(random) * static_cast<double>(seen) < static_cast<double>(capacity))
        {
            std::uniform_int_distribution<std::size_t> slot(0, capacity - 1);
            samples[slot(random)] = milliseconds;
        }
    }

    [[nodiscard]] TimingSummary summarize() const
    {
        TimingSummary summary;
        summary.count = seen;
        summary.maxMs = maxMs;
        summary.meanMs = seen > 0 ? static_cast<double>(totalMs / static_cast<long double>(seen)) : 0.0;
        summary.storedSamples = samples.size();
        summary.reservoirEstimated = seen > samples.size();
        if (!samples.empty())
        {
            auto sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            summary.p50Ms = percentileOfSorted(sorted, 0.50);
            summary.p95Ms = percentileOfSorted(sorted, 0.95);
            summary.p99Ms = percentileOfSorted(sorted, 0.99);
        }
        return summary;
    }

private:
    // Linear-interpolated percentile (the numpy "linear" convention) over an
    // ascending-sorted sample.
    static double percentileOfSorted(const std::vector<double>& sorted, const double quantile)
    {
        const auto count = sorted.size();
        if (count == 1)
            return sorted.front();
        const auto position = quantile * static_cast<double>(count - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, count - 1);
        const auto fraction = position - static_cast<double>(lower);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }

    std::size_t capacity;
    std::vector<double> samples;
    std::mt19937_64 random;
    std::uint64_t seen = 0;
    long double totalMs = 0.0L;
    double maxMs = 0.0;
};

// ============================================================================
// Deterministic representative signal: a self-contained 64-bit LCG so the
// stimulus is identical on every platform and standard library.
// ============================================================================
class DeterministicNoise
{
public:
    explicit DeterministicNoise(const std::uint64_t seed, const float peakAmplitude = 0.25f)
        : state(seed ? seed : 0x9E3779B97F4A7C15ull), peak(peakAmplitude)
    {
    }

    float nextSample()
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const auto centered = static_cast<double>(state >> 33) / 2147483648.0 - 1.0;
        return static_cast<float>(centered * static_cast<double>(peak));
    }

    void fill(float* destination, std::size_t count)
    {
        for (std::size_t index = 0; index < count; ++index)
            destination[index] = nextSample();
    }

private:
    std::uint64_t state;
    float peak;
};

// ============================================================================
// Minimal deterministic JSON writer with centralized string escaping.
// ============================================================================
class JsonWriter
{
public:
    explicit JsonWriter(std::string& sink) : out(sink) {}

    void beginObject()
    {
        separate();
        out += '{';
        stack.push_back(false);
        afterKey = false;
    }

    void endObject()
    {
        out += '}';
        stack.pop_back();
    }

    void beginArray()
    {
        separate();
        out += '[';
        stack.push_back(false);
        afterKey = false;
    }

    void endArray()
    {
        out += ']';
        stack.pop_back();
    }

    void key(std::string_view name)
    {
        comma();
        writeString(name);
        out += ':';
        afterKey = true;
    }

    void value(std::string_view text)
    {
        separate();
        writeString(text);
    }

    void value(const char* text) { value(std::string_view { text != nullptr ? text : "" }); }

    void value(bool flag)
    {
        separate();
        out += flag ? "true" : "false";
    }

    void value(double number, int precision = 6)
    {
        separate();
        if (!std::isfinite(number))
        {
            out += "null";
            return;
        }
        char buffer[64];
        const auto written = std::snprintf(buffer, sizeof(buffer), "%.*f", precision, number);
        if (written > 0)
            out.append(buffer, static_cast<std::size_t>(written));
    }

    template <typename Integer, typename = std::enable_if_t<std::is_integral_v<Integer>>>
    void value(Integer number)
    {
        separate();
        char buffer[32];
        const auto written = std::snprintf(buffer, sizeof(buffer), "%lld",
                                           static_cast<long long>(number));
        if (written > 0)
            out.append(buffer, static_cast<std::size_t>(written));
    }

    void valueUnsigned(std::uint64_t number)
    {
        separate();
        char buffer[32];
        const auto written = std::snprintf(buffer, sizeof(buffer), "%llu",
                                           static_cast<unsigned long long>(number));
        if (written > 0)
            out.append(buffer, static_cast<std::size_t>(written));
    }

private:
    void comma()
    {
        if (stack.empty())
            return;
        if (stack.back())
            out += ',';
        stack.back() = true;
    }

    void separate()
    {
        if (afterKey)
        {
            afterKey = false;
            return;
        }
        comma();
    }

    void writeString(std::string_view text)
    {
        out += '"';
        for (const char character : text)
        {
            const auto unsignedCharacter = static_cast<unsigned char>(character);
            switch (character)
            {
                case '"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                default:
                    if (unsignedCharacter < 0x20)
                    {
                        char escape[7];
                        std::snprintf(escape, sizeof(escape), "\\u%04x", unsignedCharacter);
                        out += escape;
                    }
                    else
                    {
                        out += character;
                    }
            }
        }
        out += '"';
    }

    std::string& out;
    std::vector<bool> stack;
    bool afterKey = false;
};

// ============================================================================
// Command-line options (fail-closed validation happens in parseOptions).
// ============================================================================
enum class Mode { Benchmark, Smoke, Soak };

struct Options
{
    Mode mode = Mode::Benchmark;
    std::string modelPath;
    std::string expectedSha256; // empty = record-only
    double sampleRate = 48000.0;
    std::vector<std::size_t> callbackSizes { 64, 128, 256, 512, 1024, 2048 };
    double durationSeconds = 60.0; // benchmark: total paced audio; soak: soak length
    int inferenceFrames = 512;
    int channels = 2;
    std::string jsonPath;   // empty = JSON goes to stdout after the summary
    std::string sourceDir;  // empty = compiled-in source directory
    std::uint64_t signalSeed = 0x20260214u;
};

struct ParseResult
{
    bool ok = false;
    std::string error;
    std::string usage;
    Options options;
};

// Round a requested paced segment up to complete callback cycles. This keeps
// every timing sample attributable to the callback size named in the report
// and guarantees at least one complete cycle even for short smoke runs.
[[nodiscard]] inline std::uint64_t completeCallbackCycleSamples(
    const double pacedSeconds,
    const double sampleRate,
    const std::vector<std::size_t>& callbackCycle,
    const std::uint64_t minimumSamples = 0)
{
    std::uint64_t cycleSamples = 0;
    for (const auto callbackSize : callbackCycle)
        cycleSamples += static_cast<std::uint64_t>(callbackSize);

    if (cycleSamples == 0)
        return 0;

    const auto requestedSamples = static_cast<std::uint64_t>(
        std::ceil(std::max(0.0, pacedSeconds) * std::max(0.0, sampleRate)));
    const auto requiredSamples = std::max({ std::uint64_t { 1 },
                                            requestedSamples,
                                            minimumSamples });
    const auto cycles = (requiredSamples + cycleSamples - 1) / cycleSamples;
    return cycles * cycleSamples;
}

[[nodiscard]] std::string defaultUsageText();

[[nodiscard]] inline ParseResult parseOptions(const int argc, const char* const* argv)
{
    ParseResult result;
    result.usage = defaultUsageText();
    Options options;

    std::vector<std::size_t> callbackSizes;
    bool durationSet = false;
    bool callbackSizesSet = false;
    auto fail = [&result](std::string message) {
        result.error = std::move(message);
        return result;
    };

    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        auto requiresValue = [&]() -> std::optional<std::string> {
            if (index + 1 >= argc)
                return {};
            return std::string(argv[index + 1]);
        };

        if (argument == "--mode")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--mode requires a value (benchmark|smoke|soak)");
            ++index;
            if (*value == "benchmark")
                options.mode = Mode::Benchmark;
            else if (*value == "smoke")
                options.mode = Mode::Smoke;
            else if (*value == "soak")
                options.mode = Mode::Soak;
            else
                return fail("unknown --mode '" + *value + "' (expected benchmark, smoke, or soak)");
        }
        else if (argument == "--model")
        {
            auto value = requiresValue();
            if (!value || value->empty())
                return fail("--model requires a file path");
            options.modelPath = *value;
            ++index;
        }
        else if (argument == "--expect-sha256")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--expect-sha256 requires a 64-character hex digest");
            if (!isValidSha256Hex(*value))
                return fail("--expect-sha256 must be exactly 64 hexadecimal characters");
            options.expectedSha256 = *value;
            ++index;
        }
        else if (argument == "--sample-rate")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--sample-rate requires a value");
            char* end = nullptr;
            const auto rate = std::strtod(value->c_str(), &end);
            if (end == value->c_str() || *end != '\0' || !std::isfinite(rate))
                return fail("--sample-rate must be a finite number");
            options.sampleRate = rate;
            ++index;
        }
        else if (argument == "--callback-sizes")
        {
            auto value = requiresValue();
            if (!value || value->empty())
                return fail("--callback-sizes requires a comma-separated list");
            std::size_t start = 0;
            while (start <= value->size())
            {
                const auto commaPosition = value->find(',', start);
                const auto token = value->substr(
                    start, commaPosition == std::string::npos ? std::string::npos : commaPosition - start);
                if (token.empty())
                    return fail("--callback-sizes contains an empty entry");
                char* end = nullptr;
                const auto parsed = std::strtoul(token.c_str(), &end, 10);
                if (end != token.c_str() + token.size() || parsed == 0 || parsed > 65536)
                    return fail("--callback-sizes entries must be integers in [1, 65536]: '"
                                + token + "'");
                callbackSizes.push_back(static_cast<std::size_t>(parsed));
                if (commaPosition == std::string::npos)
                    break;
                start = commaPosition + 1;
            }
            callbackSizesSet = true;
            ++index;
        }
        else if (argument == "--duration-seconds")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--duration-seconds requires a value");
            char* end = nullptr;
            const auto duration = std::strtod(value->c_str(), &end);
            if (end == value->c_str() || *end != '\0' || !std::isfinite(duration) || duration <= 0.0)
                return fail("--duration-seconds must be a positive finite number");
            options.durationSeconds = duration;
            durationSet = true;
            ++index;
        }
        else if (argument == "--inference-frames")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--inference-frames requires a value");
            char* end = nullptr;
            const auto frames = std::strtoul(value->c_str(), &end, 10);
            if (end != value->c_str() + value->size() || frames == 0 || frames > 100000)
                return fail("--inference-frames must be an integer in [1, 100000]");
            options.inferenceFrames = static_cast<int>(frames);
            ++index;
        }
        else if (argument == "--channels")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--channels requires a value (1 or 2)");
            if (*value != "1" && *value != "2")
                return fail("--channels must be 1 or 2");
            options.channels = std::stoi(*value);
            ++index;
        }
        else if (argument == "--json")
        {
            auto value = requiresValue();
            if (!value || value->empty())
                return fail("--json requires a file path");
            options.jsonPath = *value;
            ++index;
        }
        else if (argument == "--source-dir")
        {
            auto value = requiresValue();
            if (!value || value->empty())
                return fail("--source-dir requires a directory path");
            options.sourceDir = *value;
            ++index;
        }
        else if (argument == "--signal-seed")
        {
            auto value = requiresValue();
            if (!value)
                return fail("--signal-seed requires an unsigned integer");
            char* end = nullptr;
            const auto seed = std::strtoull(value->c_str(), &end, 0);
            if (end != value->c_str() + value->size())
                return fail("--signal-seed must be an unsigned integer");
            options.signalSeed = seed;
            ++index;
        }
        else if (argument == "--help" || argument == "-h")
        {
            result.ok = true;
            result.options = options;
            return result; // caller checks for --help before validation
        }
        else
        {
            return fail("unknown argument: " + std::string(argument));
        }
    }

    if (options.modelPath.empty())
        return fail("--model is required (path to the local, hash-verified TorchScript fixture)");

    if (callbackSizesSet)
    {
        std::sort(callbackSizes.begin(), callbackSizes.end());
        callbackSizes.erase(std::unique(callbackSizes.begin(), callbackSizes.end()), callbackSizes.end());
        options.callbackSizes = std::move(callbackSizes);
    }

    // The engine transport is fixed by contract; reflect that here instead of
    // inventing configuration.
    constexpr double minimumPreparedSampleRate = 0.5;
    constexpr double maximumPreparedSampleRate = 409600.0; // 5 ms fade <= 2048 samples
    if (!(options.sampleRate >= minimumPreparedSampleRate)
        || options.sampleRate > maximumPreparedSampleRate)
        return fail("--sample-rate must be in (0, 409600] Hz; the engine's 5 ms transport fade "
                    "cannot exceed the fixed 2048-sample inference quantum");

    if (options.mode == Mode::Soak)
    {
        if (!callbackSizesSet)
            options.callbackSizes = { 2048 }; // the contractual soak configuration
        if (!durationSet)
            options.durationSeconds = 7200.0; // the required 2-hour soak length
        options.inferenceFrames = 512;
    }
    else if (options.mode == Mode::Smoke)
    {
        if (!callbackSizesSet)
            options.callbackSizes = { 64, 128, 256, 512, 1024, 2048 };
        if (!durationSet)
            options.durationSeconds = 0.2; // paced seconds per callback-size segment
        options.inferenceFrames = 128;
    }

    result.ok = true;
    result.options = std::move(options);
    return result;
}

[[nodiscard]] inline std::string defaultUsageText()
{
    return
        "usage: rave_qualification --model <path.ts> [options]\n"
        "\n"
        "Offline RAVE-05 qualification harness: benchmarks the TorchScript backend and the\n"
        "shared RaveAudioEngine processing path outside any realtime host. The model fixture\n"
        "is local and hash-gated; it is never copied into or committed to the repository.\n"
        "\n"
        "required:\n"
        "  --model <path>            TorchScript model file (verify its SHA-256 first)\n"
        "\n"
        "options:\n"
        "  --mode <benchmark|smoke|soak>  run mode (default benchmark)\n"
        "  --expect-sha256 <hex>     refuse to load unless the file hash matches\n"
        "  --sample-rate <hz>        device configuration to qualify (default 48000)\n"
        "  --callback-sizes <list>   comma-separated callback sizes (default 64,128,256,512,1024,2048;\n"
        "                            soak default 2048)\n"
        "  --duration-seconds <s>    total paced audio seconds across the callback matrix\n"
        "                            (default 60; smoke 0.2); soak: total soak length (default 7200)\n"
        "  --inference-frames <n>    direct inference benchmark frames (default 512; smoke 128)\n"
        "  --channels <1|2>          engine output channels (default 2, the stereo host layout)\n"
        "  --json <path>             write the machine-readable report to a file (otherwise it is\n"
        "                            printed to stdout after the human summary)\n"
        "  --source-dir <path>       git source directory for revision capture\n"
        "                            (default: the compiled build tree)\n"
        "  --signal-seed <n>         deterministic stimulus seed (default 0x20260214)\n"
        "\n"
        "exit codes: 0 pass, 1 threshold failure, 2 usage/input error,\n"
        "            3 model hash mismatch, 4 model load/qualification failure, 5 unexpected error\n";
}

// Format seconds as a stable UTC "YYYY-MM-DDTHH:MM:SSZ" string.
[[nodiscard]] inline std::string formatUtcIso8601(std::time_t epochSeconds)
{
    std::tm utc {};
#if defined(_WIN32)
    gmtime_s(&utc, &epochSeconds);
#else
    gmtime_r(&epochSeconds, &utc);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}
} // namespace ravequalification
