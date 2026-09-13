#pragma once

// Force-included by the plugin test target before any JUCE header so the
// JUCE_LOG_CURRENT_ASSERTION hook is defined first. Any JUCE assertion that
// fires in test-compiled code records a failure instead of only logging, and
// main() turns a non-zero count into a failing test. Assertions raised inside
// the plugin target's own compilation units are additionally caught by the
// recall test wrapper, which scans the child output for JUCE assertion lines.

#include <atomic>
#include <cstdio>

namespace rave_test
{
inline std::atomic<int>& juceAssertionCount() noexcept
{
    static std::atomic<int> count { 0 };
    return count;
}

inline void recordJuceAssertion(const char* const file, const int line) noexcept
{
    juceAssertionCount().fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr, "TEST-CAPTURED JUCE assertion at %s:%d\n", file, line);
}
} // namespace rave_test

#ifndef JUCE_LOG_CURRENT_ASSERTION
#define JUCE_LOG_CURRENT_ASSERTION ::rave_test::recordJuceAssertion(__FILE__, __LINE__);
#endif
