# RAVE Performance Instrument Roadmap

## Goal

Deliver a dependable, continuously playable one-model RAVE transform instrument across standalone, VST3, and Audio Unit targets. The milestone includes live input, model-driven latent controls, MIDI learn, dry/wet processing, reliable state recall, bounded overload behavior, measured latency, and validation with real exported RAVE models.

Generate/prior operation, elaborate modulation, effects, expanded routing, multiple models, and arbitrary graph editing remain outside this milestone.

## Current baseline

The repository currently contains a shared JUCE audio engine, standalone application, VST3/AU effect targets, background TorchScript loading, RAVE metadata probing, asynchronous inference, dry/wet mixing, latent controls, eight stable plugin macros, plugin MIDI learn, and plugin state recall.

Verified locally on the current working tree:

- LibTorch build and CTest: 7/7 passing.
- LibTorch-disabled build and CTest: 5/5 passing.
- VST3 and AU bundles build and pass strict ad-hoc signature verification.
- A generated RAVE-like TorchScript fixture verifies plugin model-path and latent recall.

The plugin/recall work after commit `d7c63b9f5394dcfc8e0968905ba3e88865e6afb0` is currently uncommitted. Existing synthetic tests do not qualify real RAVE exports or DAW behavior.

## Known priority gaps

1. `RaveAudioEngine` can switch abruptly between current dry audio and delayed wet/dry output; it does not yet expose a fixed timeline or click-free recovery.
2. Variable callback sizes can cause valid wet blocks to be discarded, and plugin latency is not reported.
3. Models are loaded but not warmed up and qualified at the active sample rate before activation.
4. A saved model path that is already missing is silently skipped during state restore instead of producing the documented visible failure.
5. `InferenceWorker::prepare` resets latent controls, so device/host reprepare can lose direct latent values.
6. Standalone still lacks device selection, MIDI learn, and versioned preset save/load.
7. State parsing and backend output validation need stronger bounds, migration, finite-value, and failure tests.
8. Current tests use synthetic models and do not establish real streaming RAVE performance, host compatibility, or soak reliability.

## Phases

### RAVE-01 — Qualification contract

Define what “dependable instrument” means before optimizing implementation.

- [ ] Record the source revision, compiler, JUCE/LibTorch versions, Mac hardware, and build configuration used for qualification.
- [ ] Select permitted real streaming RAVE exports and record identity, provenance, sample rate, channel count, latent count, and reset requirements.
- [ ] Define supported devices, DAW hosts, sample rates, callback sizes, and mono/stereo routing behavior.
- [ ] Define measurable callback-time, inference-time, end-to-end latency, overload-recovery, and soak thresholds.
- [ ] Preserve existing plugin parameter IDs: `dryWet` and `macro1` through `macro8`.
- [ ] Confirm the LibTorch-enabled test suite is actually registered with `ctest -N`.

**Verification:** full current build/test matrix and a checked qualification specification.

**Delivery boundary:** qualification documentation only; implementation and publication require separate authority.

### RAVE-02 — Model activation and lifecycle reliability

Depends on: RAVE-01 verified.

- [ ] Warm up and validate candidate models off the audio callback at the intended runtime configuration before reporting them active.
- [ ] Confirm streaming/reset behavior with selected real RAVE exports.
- [ ] Reject unsupported sample rates clearly before activation, or add measured resampling with explicit latency.
- [ ] Keep the previous working model active when candidate qualification fails.
- [ ] Reject malformed metadata, non-finite output, unexpected shapes, and backend exceptions with bounded dry fallback.
- [ ] Audit model replacement, prepare/release, editor closure, and shutdown for synchronization and blocking risks.
- [ ] Document that an in-process worker cannot contain native LibTorch crashes or forcibly cancel stalled native inference.

**Focused checks:**

```sh
ctest --test-dir build -R 'rave_(model_loader|torch_backend|inference_worker)_tests' --output-on-failure
```

Add warm-up failure, sample-rate mismatch, malformed metadata, stateful reset, non-finite output, throwing backend, and repeated replacement cases.

### RAVE-03 — Stable timing, fallback, and latency

Depends on: RAVE-01 and RAVE-02 verified.

- [ ] Establish a fixed output timeline independent of worker completion timing and callback partitioning.
- [ ] Keep the dry path consistently aligned with processed output at every dry/wet value.
- [ ] Report VST3/AU latency accurately to the host.
- [ ] Define behavior for variable, zero-length, and oversized callbacks.
- [ ] Implement smoothed dry/wet transitions plus controlled overload fallback and recovery fades.
- [ ] Drop stale output deterministically and recover from queue saturation without replaying increasingly old audio.
- [ ] Expose readiness, latency, deadline misses, queue drops, processing errors, and basic meters without blocking audio.
- [ ] Audit callback allocations, synchronization primitives, notifications, and worst-case execution time.

**Focused checks:**

```sh
ctest --test-dir build -R 'rave_(engine|inference_worker|audio_engine|plugin)_tests' --output-on-failure
```

Add impulse-latency, alternating callback-size, delayed-worker, queue-saturation, failure/recovery, and timing-continuity tests at 0%, intermediate, and 100% wet.

### RAVE-04 — Reliable controls, presets, and standalone parity

Depends on: RAVE-02 verified. Final audio acceptance also depends on RAVE-03.

- [ ] Preserve all latent values across prepare/reprepare, sample-rate changes, editor closure, and device changes.
- [ ] Correct missing-model restore status and prevent older pending loads from overriding newer state.
- [ ] Define macro authority over latent dimensions 1–8 and remove duplicate/ambiguous serialized sources.
- [ ] Bound serialized latent counts and validate state versions; retain tested migration from the initial float-only plugin state.
- [ ] Add explicit model relinking when a saved same-machine path is unavailable.
- [ ] Add standalone MIDI input selection, learn, clear/reassign behavior, and saved mappings.
- [ ] Add versioned standalone preset save/load for model identity, every latent value, dry/wet, and MIDI assignments.
- [ ] Add standalone audio-device/input selection sufficient for reproducible performance.
- [ ] Test recall with more than eight latent dimensions and process audio after restoration.

**Focused checks:**

```sh
ctest --test-dir build -R 'rave_plugin(_model_recall)?_tests' --output-on-failure
```

Add shared-state/standalone tests for restore-before-prepare, reprepare, missing files, superseded loads, unsupported versions, and larger latent vectors.

### RAVE-05 — Real-model and host qualification

Depends on: RAVE-03 and RAVE-04 verified.

- [ ] Benchmark selected real RAVE models on the target Mac under representative concurrent audio load.
- [ ] Record callback and inference timing distributions, latency, CPU, memory, queue drops, deadline misses, and processing errors against RAVE-01 thresholds.
- [ ] Run the agreed soak duration, including overload and recovery.
- [ ] Verify standalone device changes, MIDI disconnect/reconnect, preset reopen, and model failure recovery.
- [ ] Validate VST3 and AU scanning and operation in selected hosts, including editor-closed recall, automation, bypass, transport changes, repeated prepare/release, and multiple instances.
- [ ] Test offline rendering and either support deterministic behavior or document the limitation.
- [ ] Repeat the full LibTorch/no-LibTorch matrix and bundle verification on the final source state.
- [ ] Update README claims to match measured behavior and retained limitations.

**Qualification commands:**

```sh
cmake --build build --parallel
ctest --test-dir build -N
ctest --test-dir build --output-on-failure

cmake -S . -B build-no-torch -G Ninja \
  -DRAVE_USE_SYSTEM_JUCE=OFF -DRAVE_ENABLE_LIBTORCH=OFF
cmake --build build-no-torch --parallel
ctest --test-dir build-no-torch --output-on-failure

codesign --verify --strict --verbose=2 "<VST3 bundle>"
codesign --verify --strict --verbose=2 "<AU bundle>"
auval -v aumf RvPI RvAI
```

Host installation, registration, signing identities, notarization, and release remain separate authorized delivery operations.

## Deferred backlog

After the dependable transform milestone:

- Generate/prior mode.
- Latent scaling, freeze, smoothing, LFOs, envelopes, XY morphing, and named positions.
- User-defined macro mappings across multiple latent dimensions.
- Optional file player, pre/post effects, and output matrix.
- Portable content-addressed model libraries.
- ONNX evaluation only after streaming-state and output equivalence are proven.
- Optional process-isolated inference for unattended installations.

Explicitly deferred: multiple simultaneous models and arbitrary graph editing.
