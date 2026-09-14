#pragma once

#include <cstdint>

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
}
