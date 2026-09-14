#pragma once

#include <cstdint>
#include <optional>
#include <utility>

namespace rave
{
// Message-thread request gate used to reject stale asynchronous model results.
// It has no realtime callers: callbacks only update StandaloneSessionState.
class LatestRequestGeneration final
{
public:
    [[nodiscard]] std::uint64_t begin() noexcept { return ++current; }

    [[nodiscard]] bool isCurrent(const std::uint64_t generation) const noexcept
    {
        return generation == current;
    }

private:
    std::uint64_t current = 0;
};

// Retains only the newest deferred or queued request. The request payload is
// deliberately opaque so the helper stays independent of JUCE and testable.
template <typename Request> class LatestRequestSlot final
{
public:
    void replace(Request request, const std::uint64_t generation)
    {
        entry = Entry{std::move(request), generation};
    }

    [[nodiscard]] bool hasValue() const noexcept { return entry.has_value(); }

    [[nodiscard]] std::optional<Request> takeCurrent(const LatestRequestGeneration& generations)
    {
        if (!entry.has_value())
            return std::nullopt;

        auto result = std::move(entry);
        entry.reset();
        if (!generations.isCurrent(result->generation))
            return std::nullopt;

        return std::move(result->request);
    }

private:
    struct Entry
    {
        Request request;
        std::uint64_t generation = 0;
    };

    std::optional<Entry> entry;
};
} // namespace rave
