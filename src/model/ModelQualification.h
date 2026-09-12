#pragma once

#include "model/ModelBackend.h"

#include <string>

namespace rave
{
// Validates a loaded candidate model off the audio thread at the intended
// runtime configuration before it may be reported active:
//
//   1. reject runtime configurations the model cannot run at (for example an
//      exported sample rate mismatch),
//   2. prepare and reset the model,
//   3. warm it up with a deterministic low-level signal on the exact runtime
//      processing path (latent encode/decode when available, plain forward
//      otherwise),
//   4. require correctly sized, finite output for every warm-up block.
//
// Returns an empty string when the candidate qualifies, otherwise an actionable
// diagnostic. Never throws: backend exceptions are captured as failures.
[[nodiscard]] std::string qualifyModelBackend(ModelBackend& backend,
                                              const ModelRuntimeConfiguration& configuration);
} // namespace rave
