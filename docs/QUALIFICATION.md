# RAVE Instrument Qualification Contract

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
- Required streaming model methods and metadata are `forward`; optional latent operation requires compatible `encode`, `decode`, `forward_params`, `encode_params`, and `decode_params`.

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
