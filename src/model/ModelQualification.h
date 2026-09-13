#pragma once

#include "model/ModelBackend.h"

#include <string>

namespace rave
{
// Validates a runtime configuration against a model without running audio:
// rejects unset configurations and configurations the model cannot run at
// (for example an exported sample rate mismatch). Returns an empty string when
// the configuration is acceptable, otherwise an actionable diagnostic.
[[nodiscard]] std::string checkModelConfiguration(ModelBackend& backend,
                                                  const ModelRuntimeConfiguration& configuration);

// Validates a loaded candidate model off the audio thread at the intended
// runtime configuration before it may be reported active:
//
//   1. reject runtime configurations the model cannot run at (see above),
//   2. prepare the model,
//   3. reset it, requiring the model's reset to succeed,
//   4. warm it up with a deterministic low-level signal on the exact runtime
//      processing path (latent encode/decode when available, plain forward
//      otherwise),
//   5. require correctly sized, finite output for every warm-up block.
//
// Returns an empty string when the candidate qualifies, otherwise an actionable
// diagnostic. Never throws: backend exceptions are captured as failures.
[[nodiscard]] std::string qualifyModelBackend(ModelBackend& backend,
                                              const ModelRuntimeConfiguration& configuration);
} // namespace rave
