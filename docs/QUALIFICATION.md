# RAVE Instrument Qualification Contract

## RAVE-04 core state checks

The standalone state test exercises a 12-latent bounded `.ravepreset` round trip, MIDI learn/clear/unique reassignment, dry/wet and latent CC ranges, selected MIDI input identity, audio input/output device identity, and deterministic latest-request gating. The standalone UI wires these state operations through message-thread snapshots, AudioDeviceSelectorComponent, one selected MIDI callback, visible relinking, and scrolling learn/clear rows. Plug-in state schema 2 is capped at 1 MiB and 4096 latents; dimensions 0–7 come only from the stable host macros while dynamic state begins at dimension 8. Missing saved paths remain visible and relinkable. Real-device and DAW qualification are intentionally not claimed here.

## Purpose

This document defines the measurable acceptance contract for the dependable one-model Transform milestone. It does not claim that the thresholds currently pass. Results must identify the exact source revision, model file hash, hardware, host, sample rate, buffer size, and test duration.

## Baseline environment

Recorded on 2026-09-12.

| Item | Qualification baseline |
| --- | --- |
| Hardware | Macmini8,1; Intel Core i7-8700B 3.20 GHz; 32 GiB RAM |
| OS | macOS 14.8.8 (23J620), Darwin x86_64 |
| Compiler | Apple Clang 16.0.0 (clang-1600.0.26.3) |
| CMake / Ninja | CMake 4.2.3; Ninja 1.13.2 |
| JUCE | 7.0.9 |
| LibTorch | PyTorch/LibTorch 2.2.2, x86_64 |
| Standalone | JUCE CoreAudio device backend |
| AU host | Logic Pro 11.2.2 |
| VST3/AU host | REAPER 7.79 |
| Additional host | Ableton Live 12.4.2 |

The baseline source before qualification work is commit `10da07e57d5537b96106d7694611300e8d9a7211`. Later reports must use their actual tested revision instead.

## Development model fixtures

Third-party model files are local qualification inputs, not repository assets. Do not commit or redistribute them.

Primary candidate:

- Repository: `Intelligent-Instruments-Lab/rave-models`
- Repository revision observed: `c25a03d625840c40cd3a48779ed72f2a1947d7b4`
- File: `birds_pluma_b2048_r48000_z12.ts`
- Declared properties from filename: 2048 frame, 48 kHz, 12 latent dimensions
- Size: 42,105,901 bytes
- LFS SHA-256: `a12ad61a2b0b5ee2329a72993bd94386a571600b37dd23feaa0a404940468d68`

Secondary candidates:

- `guitar_iil_b2048_r48000_z16.ts` for a larger 48 kHz model.
  - Size: 163,881,670 bytes
  - LFS SHA-256: `02458214e23890d6818504319a5b9903eabfe87a524491f6524f453e7f3dbcf0`
- A 44.1 kHz model such as `voice_jvs_b2048_r44100_z16.ts` to verify explicit sample-rate compatibility or rejection.

The model repository declares CC BY-NC 4.0. These files are approved here only as local, non-commercial development fixtures. Commercial use, bundling, or redistribution requires a separate licensing decision. The hosting service marks TorchScript files with a generic model-code security warning while also reporting no antivirus detections for the named primary/48 kHz secondary artifacts. Treat every model as executable untrusted input: verify the exact hash and provenance before loading, and never load an unreviewed user-supplied model during qualification.

## Supported milestone behavior

### Audio and model contract

- One active model.
- Transform mode only: live audio to `forward`, or `encode` → latent offsets → `decode`.
- Mono RAVE model input and output.
- Mono host layout: mono dry and mono processed output.
- Stereo host layout: stereo dry path; channel 1 feeds the mono model; processed mono is duplicated into both wet outputs.
- Primary sample rate: 48,000 Hz.
- A model whose exported sample rate differs from the active device/host rate must be rejected before activation until measured resampling exists.
- Required streaming model methods and metadata are `forward`; optional latent operation requires compatible `encode`, `decode`, `forward_params`, `encode_params`, and `decode_params`. A declared sample rate is read from the `get_sample_rate` method or, when it is absent, the `sampling_rate` integer buffer used by nn~/RAVE exports. Present metadata must be strictly valid: malformed buffers (non-tensor, fractional or wrong dtype, wrong rank, non-positive values) and invalid declared rates reject the candidate instead of degrading silently. The `sampling_rate` buffer must be a scalar or a single-element 1-D buffer; other ranks or shapes (for example a rank-2 singleton) are rejected.

### Model activation and lifecycle

- A candidate model is loaded, prepared, reset, and warmed up on the background
  loader thread at the intended device sample rate and maximum block size
  before it may be reported active; the audio callback never performs
  qualification work.
- A candidate whose declared sample rate differs from the active configuration
  is rejected before activation with an actionable message that names both
  rates; incompatible rates are never activated silently.
- Warm-up must produce correctly sized, finite output on the exact runtime
  processing path, and the model's reset must succeed. Malformed metadata
  buffers, unexpected output shapes, reset failures, processing failures, and
  thrown backend exceptions fail qualification.
- Activation rechecks every candidate against the exact known host configuration under lifecycle ownership, so a prepare or sample-rate change after background qualification cannot install an incompatible candidate over the previous usable model. One lifecycle ownership mutex protects every non-audio lifecycle mutation (prepare, release, replacement) and the configuration snapshot itself; the audio callback never takes it and stays lock-free, reading only single atomic copies published under the mutex. Model loading and queued restores are deferred until a real host/device configuration exists; defaults are never invented. The known configuration survives release and callback detachment. Deterministic lifecycle tests instrument the prepare-side lock-acquisition boundary and coordinate with a latch so a concurrent prepare is provably pending on serialization before activation commits.
- Activation reports success only when the candidate's checked reset and worker start succeeded under the exact snapshot — whenever a configuration is known, activation prepares, checked-resets, and starts the candidate, even after release or a previous rollback failure. If the candidate's reset refuses at activation time (for example it reset cleanly during qualification but not afterwards), the candidate is rejected, the previous backend and all of its latent control values are restored, and its restart is verified before retention is claimed.
- Loaded and usable are tracked separately. Every prepare verifies the installed backend against the new exact configuration before starting; an incompatible reprepare or failing reset keeps the model installed but reports it as not usable and renders bounded dry pass-through until a compatible configuration returns. The engine retains a thread-safe lifecycle diagnostic plus revision counter for these transitions — including release/device stop — and the plugin and standalone status surfaces render the shared lifecycle presenter's truthful text (failure diagnostic, active claim, installed-but-not-running, or no model), never a stale active claim while silent. Backend prepare, checked reset, and worker thread-start failures are contained, recorded, and never escape host or device prepare paths. When a rollback abandons a backend, the engine latent count and worker latent storage are cleared together with the installed flag, so no model means zero latents, and the plugin drops the serialized model identity so no dead path is restored. The standalone app derives its immediate post-reattach status from the current engine state through the same shared presenter and synchronizes the observed lifecycle revision.
- A failed qualification or activation reports its truthful outcome through the visible status surface: the previous usable model retained (with all latent control values restored), no previous model (bounded dry pass-through), or a rollback that could not restart the previous model (bounded dry pass-through, previous model not claimed as retained, with a later healthy replacement still activatable).
- Non-finite runtime output from an already active model is dropped as a
  processing error so the engine falls back to dry audio instead of emitting it.
- Repeated model replacement, prepare/release cycles, editor closure, and
  shutdown are bounded: loader and inference threads are joined, never leaked.
- An in-process inference worker cannot contain a native LibTorch crash and
  cannot forcibly cancel a stalled native inference call; both remain
  process-level risks documented for this milestone. An optional separate
  inference process stays a deferred option for unattended installations.

### Callback and block matrix

Exercise these callback sizes at 48 kHz where the device/host supports them:

- 64, 128, 256, 512, 1024, and 2048 samples.
- Alternating callback partitions with equal total samples.
- Zero-length callbacks if supplied by the adapter test harness.
- Offline/non-realtime host rendering as a separately reported mode.

A supported configuration must not silently discard valid model frames merely because callback partitions vary.

### Stable host parameters

These IDs are compatibility commitments for the milestone:

- `dryWet`
- `macro1` through `macro8`

Their normalized meaning, ordering, and version must not change. Additional model-dependent latents remain UI/session state, not dynamically created host parameters.

## Acceptance thresholds

### Real-time callback

For each supported configuration:

- No heap allocation, mutex acquisition, model execution, file access, or thread join in the audio callback.
- Callback wall time p99 ≤ 25% of the device callback period.
- Callback maximum ≤ 75% of the device callback period during the normal-load run.
- Zero host/device xruns caused by the plugin during the normal-load run.
- No non-finite samples at the output.

### Inference and queues

For the primary model at its supported processing frame:

- Inference wall time p99 ≤ 90% of the model-frame duration.
- No input drops, output drops, alignment errors, or processing errors during the normal-load run.
- Queue depth and age remain bounded by the documented capacity.
- Stale results are never replayed after overload recovery.

### Latency

- Measured wet-path impulse latency is deterministic for a fixed model/configuration within ±1 sample after warm-up.
- Reported plugin latency equals measured processing latency within ±1 sample.
- Dry audio is delayed by the same processing latency whenever a model is active, including 0% wet, so automation does not change timing.
- Target total processing latency for the primary 48 kHz model is ≤ 4096 samples (85.34 ms), excluding hardware input/output latency. A model exceeding this limit is unsupported for the milestone unless the contract is deliberately revised with measured evidence.

### Transitions and overload

- Dry/wet and fallback gain transitions are ramped over 5–20 ms with no single-sample discontinuity greater than 0.1 full scale for bounded test signals.
- On a missed deadline, output moves to the defined aligned fallback without waiting in the callback.
- Recovery begins only with a current, sequence-valid output block and completes within 500 ms after inference returns below deadline.
- Repeated overload does not increase latency without bound.
- Loading or qualifying a failed candidate leaves the previous playable model active.

### State and controls

- Plugin state round-trips dry/wet, macros, MIDI mappings, active model reference, and every latent value.
- State parsing rejects unsupported versions and bounds model path length and latent count before allocation.
- More-than-eight-dimensional models retain direct latent values across editor close/reopen and prepare/reprepare.
- Missing model files produce a visible relinkable state while audio remains safe.
- Standalone presets provide equivalent model, latent, dry/wet, and MIDI recall.

### Soak and recovery

- Automated engine soak: minimum 2 hours with the primary model and representative signal at 48 kHz/2048 samples.
- Per-host interactive soak: minimum 30 minutes in Logic, REAPER, and Ableton for each supported plugin format available in that host.
- During soak: no crash, deadlock, unbounded memory growth, non-finite output, or unexplained counter increase.
- Exercise model replacement, editor closure, transport stop/start, device or sample-rate reprepare, MIDI disconnect/reconnect, and controlled synthetic overload.

## Verification matrix

### Automated baseline

```sh
cmake --build build --parallel
ctest --test-dir build -N
ctest --test-dir build --output-on-failure

cmake -S . -B build-no-torch -G Ninja \
  -DRAVE_USE_SYSTEM_JUCE=OFF -DRAVE_ENABLE_LIBTORCH=OFF
cmake --build build-no-torch --parallel
ctest --test-dir build-no-torch -N
ctest --test-dir build-no-torch --output-on-failure
```

Expected registration at this baseline:

- LibTorch build: 7 tests, including `rave_torch_backend_tests` and `rave_plugin_model_recall_tests`.
- LibTorch-disabled build: 5 tests; LibTorch-specific tests absent.

### Plugin packaging and hosts

```sh
codesign --verify --deep --strict "<VST3 bundle>"
codesign --verify --deep --strict "<AU bundle>"
auval -v aumf RvPI RvAI
```

Also record:

- VST3 scan and processing in REAPER and Ableton.
- AU scan and processing in Logic and REAPER.
- Automation read/write for `dryWet` and all eight macros.
- Session reopen with the editor closed before save.
- Bypass, transport changes, repeated prepare/release, multiple instances, and offline render behavior.

Host installation, AU registration, distribution signing, notarization, and release are distinct delivery operations. A bundle signature check alone is not host qualification.

## Evidence format

Every benchmark or host report must include:

- Full Git commit SHA and clean/dirty status.
- Build type and CMake options.
- Model source URL, repository revision, exact filename, byte size, and SHA-256.
- Hardware, OS, host, audio device, sample rate, callback size, channel layout, and duration.
- Callback/inference count, p50/p95/p99/max timing, measured/reported latency, queue/drop/error counters, xruns, CPU, and memory.
- Pass/fail against each applicable threshold.
- Known omissions and whether the run used synthetic or real audio.

## RAVE-03 fixed-frame qualification

The backend qualification contract is exactly 48 kHz / 2048 samples. When
`RAVE_TEST_MODEL_PATH` is set, the existing SHA-256-gated backend smoke runs
that exact frame size; the fixture remains local and must not be committed.
The engine transport is independently fixed at 4096 samples and is not a
measurement of a model's intrinsic receptive-field latency. Host qualification
must confirm the processor continues to report 4096 through model activation,
overload, release, and reprepare.
