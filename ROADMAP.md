# RAVE Performance Instrument Roadmap

## Goal

Deliver a dependable, continuously playable one-model RAVE transform instrument across standalone, VST3, and Audio Unit targets. The milestone includes live input, model-driven latent controls, MIDI learn, dry/wet processing, reliable state recall, bounded overload behavior, measured latency, and validation with real exported RAVE models.

Generate/prior operation, elaborate modulation, effects, expanded routing, multiple models, and arbitrary graph editing remain outside this milestone.

## Current baseline

The repository currently contains a shared JUCE audio engine, standalone application, VST3/AU effect targets, background TorchScript loading with off-thread candidate qualification, transactional model activation and rollback, RAVE metadata probing, fixed 2048-sample asynchronous inference framing on a 4096-sample host-reported transport, aligned dry/wet mixing with deadline fallback and recovery fades, bounded runtime telemetry, latent controls, eight stable plugin macros, plugin MIDI learn, and plugin state recall.

Latest local qualification evidence:

- Current RAVE-06 LibTorch-disabled build and CTest: 8/8 passing, including deterministic icon assets.
- The preceding RAVE-04 LibTorch qualification passed 9/9, including plugin model-path and latent recall with the hash-gated real fixture; the post-icon LibTorch matrix remains a RAVE-05 check.
- Current VST3 and AU bundles build, contain the generated icon resources, and pass strict ad-hoc signature verification.

Deterministic tests qualify lifecycle behavior, exact transport timing, variable callback partitions, deadline commitment, fallback/recovery fades, queue saturation, stale-result rejection, telemetry epochs, and the SHA-256-verified local RAVE fixture. DAW hosts, long-duration performance, and perceptual/intrinsic model latency remain unqualified.

## Known priority gaps

1. Original project-owned icon generation and packaging are complete in RAVE-06; official RAVE/IRCAM artwork is intentionally out of scope.
2. Real DAW compatibility, soak reliability, intrinsic model latency, target-machine performance, and final human icon observation remain RAVE-05 qualification work. Hardware, DAW, soak, and visual observation are not claimed by deterministic source checks.

## Phases

### RAVE-01 — Qualification contract

Define what “dependable instrument” means before optimizing implementation.

- [x] Record the source revision, compiler, JUCE/LibTorch versions, Mac hardware, and build configuration used for qualification.
- [x] Select permitted real streaming RAVE exports and record identity, provenance, sample rate, channel count, latent count, and reset requirements.
- [x] Define supported devices, DAW hosts, sample rates, callback sizes, and mono/stereo routing behavior.
- [x] Define measurable callback-time, inference-time, end-to-end latency, overload-recovery, and soak thresholds.
- [x] Preserve existing plugin parameter IDs: `dryWet` and `macro1` through `macro8`.
- [x] Confirm the LibTorch-enabled test suite is actually registered with `ctest -N`.

**Verification:** full current build/test matrix and a checked qualification specification.

**Delivery boundary:** qualification documentation only; implementation and publication require separate authority.

### RAVE-02 — Model activation and lifecycle reliability

Depends on: RAVE-01 verified.

- [x] Warm up and validate candidate models off the audio callback at the intended runtime configuration before reporting them active.
- [x] Confirm streaming/reset behavior with the SHA-256-verified `birds_pluma_b2048_r48000_z12.ts` local qualification fixture at 48 kHz/2048 samples.
- [x] Reject unsupported sample rates clearly before activation, or add measured resampling with explicit latency.
- [x] Keep the previous working model active when candidate qualification fails.
- [x] Reject malformed metadata, non-finite output, unexpected shapes, and backend exceptions with bounded dry fallback.
- [x] Audit model replacement, prepare/release, editor closure, and shutdown for synchronization and blocking risks.
- [x] Document that an in-process worker cannot contain native LibTorch crashes or forcibly cancel stalled native inference.

**Focused checks:**

```sh
ctest --test-dir build -R 'rave_(model_loader|torch_backend|inference_worker)_tests' --output-on-failure
```

Add warm-up failure, sample-rate mismatch, malformed metadata, stateful reset, non-finite output, throwing backend, and repeated replacement cases.

### RAVE-03 — Stable timing, fallback, and latency

Depends on: RAVE-01 and RAVE-02 verified.

- [x] Establish a fixed output timeline independent of worker completion timing and callback partitioning.
- [x] Keep the dry path consistently aligned with processed output at every dry/wet value.
- [x] Report VST3/AU latency accurately to the host.
- [x] Define behavior for variable, zero-length, and oversized callbacks.
- [x] Implement smoothed dry/wet transitions plus controlled overload fallback and recovery fades.
- [x] Drop stale output deterministically and recover from queue saturation without replaying increasingly old audio.
- [x] Expose readiness, latency, deadline misses, queue drops, processing errors, and alignment errors without blocking audio; retain peak metering for final qualification.
- [x] Audit callback allocations, synchronization primitives, notifications, and worst-case bounded work.

**Focused checks:**

```sh
ctest --test-dir build -R 'rave_(engine|inference_worker|audio_engine|plugin)_tests' --output-on-failure
```

Add impulse-latency, alternating callback-size, delayed-worker, queue-saturation, failure/recovery, and timing-continuity tests at 0%, intermediate, and 100% wet.

### RAVE-04 — Reliable controls, presets, and standalone parity

Depends on: RAVE-02 verified. Final audio acceptance also depends on RAVE-03.

- [x] Preserve standalone session latent values across device callback reprepare, controls, and model activation; plugin lifecycle coverage remains in the existing engine tests.
- [x] Surface missing standalone/plugin model paths as relink-required and reject stale standalone loader results with a monotonic request generation.
- [x] Define host macro parameters as authoritative for latent dimensions 1–8; plugin schema v2 serializes only dynamic latents from index 8 onward.
- [x] Bound standalone/plugin serialized latent counts, validate standalone state versions, and migrate the initial float-only plugin state format.
- [x] Add explicit model relinking when a saved same-machine path is unavailable.
- [x] Add standalone MIDI input identity and one-selected-input callback wiring, plus learn, clear/reassign behavior for dry/wet and every scrolling latent row.
- [x] Add bounded versioned standalone `.ravepreset` Save/Load state for model identity, every latent value, dry/wet, MIDI assignments/input, and audio setup.
- [x] Add visible standalone AudioDeviceSelectorComponent input/output configuration and message-thread device application.
- [x] Test standalone state recall with 12 latents, MIDI/device identities, deterministic latest-request gating, and preset-owned control retention through pending/missing/relink activation; real-device and DAW observation remain RAVE-05.

**Focused checks:**

```sh
ctest --test-dir build -R 'rave_plugin(_model_recall)?_tests' --output-on-failure
```

Add shared-state/standalone tests for restore-before-prepare, reprepare, missing files, superseded loads, unsupported versions, and larger latent vectors.

### RAVE-06 — Original project-owned instrument icon

- [x] Create and record provenance for a restrained original geometric icon; do not copy, embed, or imply endorsement by RAVE, IRCAM, or a third party.
- [x] Keep one canonical SVG source and deterministically generate square transparent 16–1024 px derivatives, including representative small/HiDPI and large/HiDPI sizes, without distortion.
- [x] Display the same embedded SVG identity in the standalone and plugin editors with an accessible icon title while retaining the existing control layouts.
- [x] Use JUCE's `ICON_BIG`/`ICON_SMALL` CMake facilities for standalone, VST3, and AU metadata; package JUCE-generated `Icon.icns` and the inspectable canonical SVG where macOS bundles support resources.
- [x] Document project authorship, MIT artwork license, source, deterministic generation, hashes, and the user-authorized substitution for the rejected official mark.

**Focused checks:** run `python3 assets/icon/generate_icon.py --check`; configure a fresh no-LibTorch JUCE build, build standalone/VST3/AU, run CTest, inspect dimensions/alpha/hashes and each bundle's `Icon.icns`, `rave-instrument-icon.svg`, and `Info.plist`, then run strict VST3/AU codesign verification. Human visual confirmation at intended host/system scale remains deferred to RAVE-05.

### RAVE-05 — Real-model and host qualification

Depends on: RAVE-03, RAVE-04, and RAVE-06 verified.

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
