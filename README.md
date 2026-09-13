# RAVE Performance Instrument

## Direction

Build a native RAVE performance instrument in C++ with JUCE, using LibTorch initially to run existing RAVE models. The application should have a shared audio engine that can later target standalone, VST3, and AU formats.

The instrument should let a performer load a model, route audio through it, explore its latent space, assign modulation and MIDI, and save the complete setup.

### Fixed realtime transport

Audio is accumulated into 2048-sample inference frames and rendered on a
model-independent 4096-sample timeline. The plugin reports 4096 samples in all
model and overload states. Missing results use exactly aligned delayed dry;
dry/wet and availability transitions are independently faded over 5 ms. This
contractual transport latency does not measure intrinsic model receptive-field
or perceptual latency.

Runtime counters reset on prepare. Deadline misses count frames absent at their
playback boundary; queue drops combine worker input/output enqueue failures;
late results arrived after their boundary; processing/reset errors count
backend failures; alignment errors count malformed, duplicate, impossible
future results, and invalid layouts. Lifecycle status separately remains the
authoritative model-usability state.

Reference projects:

- [IRCAM RAVE](https://github.com/acids-ircam/RAVE)
- [Neutone models](https://neutone.jp/fx/models)
- [JUCE](https://juce.com/)
- [nn~ documentation](https://github.com/acids-ircam/nn_tilde)
- [CPAL](https://github.com/RustAudio/cpal)
- [egui](https://github.com/emilk/egui)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

## Proposed stack

| Layer | Recommendation | Rationale |
| --- | --- | --- |
| Application and plugin framework | JUCE + C++20 | Native audio, MIDI, UI, standalone, VST3, and AU targets |
| Initial model runtime | LibTorch, CPU inference | Closest compatibility with exported RAVE TorchScript models |
| Interface | Custom JUCE components | Faders, knobs, XY pads, meters, and modulation displays |
| Conventional DSP | JUCE DSP plus custom processors | Filtering, gain, envelopes, saturation, delay, and mixing |
| Routing | Small explicit processing graph | Performance-oriented flexibility without a patching language |
| State | Versioned presets and sessions | Model identity, controls, routing, MIDI, and modulation assignments |
| Build | CMake | Shared engine with standalone and plugin targets |

## Playable modes

- **Transform:** incoming audio → encoder → latent manipulation → decoder → output.
- **Generate:** recorded or generated latent trajectories → decoder → output, with a prior where supported.

Latent controls should support per-dimension offsets and scaling, freeze, smoothing, LFOs, envelopes, MIDI assignments, and an XY pad for morphing between saved positions within the same model.

Latent dimensions are model-dependent rather than fixed synth parameters. The interface should encourage exploration, naming useful positions, and building macro controls from combinations of dimensions. Movement through latent space may be more musically useful than a stationary latent vector.

## Initial routing

Start with input selection, an optional file player, one model, dry/wet mixing, pre/post effects, and an output matrix. Defer multiple models and arbitrary graph editing.

## Audio architecture

- Real-time audio callback with bounded DSP and preallocated audio queues.
- Dedicated inference worker for model execution.
- Background loading and warm-up before model activation.
- Non-blocking UI exchange for controls and metering.
- The audio callback must never wait for inference.
- Define a controlled-fade fallback and visible overload indicator for missed inference deadlines.
- An in-process inference worker cannot contain a native LibTorch crash and cannot forcibly cancel a stalled native inference call; both are documented process-level risks for this milestone.
- Measure model, buffering, resampling, and device latency; compensate the dry path and report plugin latency accurately.
- Consider an optional separate inference process later if unattended installations become important.

## Runtime strategy

Keep inference behind a replaceable backend interface. Begin with LibTorch and CPU inference. Investigate ONNX Runtime for selected models only after proving streaming-state compatibility and output equivalence; existing `.ts` model loading should not be assumed.

## Current implementation

The repository currently provides:

- A JUCE standalone application plus VST3 and Audio Unit effect targets, all
  sharing the same `RaveAudioEngine` processing path.
- Host-visible dry/wet plus eight stable macro parameters mapped to latent
  dimensions 1–8, with remaining dimensions available as direct UI controls.
- MIDI CC learn for dry/wet and each fixed macro; assignments and parameter
  values are included in plugin state.
- Plugin session recall for the model path and every latent offset. Model reload
  remains asynchronous and works without opening the editor; missing files fail
  visibly instead of blocking audio.
- A custom plugin editor with background TorchScript model selection and a
  scrollable model-driven latent control surface.
- Preallocated, bounded SPSC audio-block queues feeding a dedicated inference
  worker; audio submission and retrieval never wait for model execution.
- Sequence-tagged dry and wet blocks so asynchronous results are mixed with
  their corresponding input rather than the current callback block.
- Live-engine integration with dry fallback plus missed-deadline, alignment,
  queue-overload, and processing-error counters for later UI metering.
- An optional LibTorch backend that loads generic `forward(audio)` TorchScript
  models and probes nn~/RAVE `forward_params`, `encode_params`, `decode_params`,
  and `get_sample_rate` metadata.
- RAVE latent offsets implemented as `encode` → per-dimension broadcast offsets
  → `decode`; controls are atomically snapshotted by the inference worker.
- A scrollable latent-fader surface generated from each model's reported latent
  dimension count, with double-click reset to zero.
- A native model chooser backed by off-thread loading and qualification: every
  candidate is loaded, prepared, reset, and warmed up at the intended sample
  rate and block size before activation. Incompatible sample rates, malformed
  metadata, unexpected output shapes, failing resets, non-finite warm-up
  output, processing failures, and thrown backend exceptions fail
  qualification, keep the previous model active, and surface an actionable
  status message.
- Transactional engine activation rechecks each candidate against the exact known host configuration under lifecycle ownership and — whenever a configuration is known — prepares, checked-resets, and starts it before success, even after release or a failed rollback. Loaded and usable are tracked separately: incompatible reprepares or failing resets keep the model installed but report truthful bounded dry pass-through, the engine retains a lifecycle diagnostic and revision that the shared presenter used by both the plugin and standalone status surfaces renders for every transition — including release/device stop rendering installed-but-not-running —, abandoning a backend after a failed rollback clears all latent state and the serialized model identity, and the standalone derives its post-reattach status from the live engine state; the previous backend's restart is verified before claiming retention, and a later healthy replacement remains activatable. The configuration survives release and callback detachment; model loading and state restores wait for a real host configuration, and repeated replacement, prepare/release cycles, editor closure, and shutdown remain bounded.
- A native model chooser that safely detaches and restarts the audio callback
  around backend replacement, so no loading or preparation work ever runs in
  the callback.
- CTest coverage for queue behavior and, when Python Torch and LibTorch are
  configured, end-to-end TorchScript loading, metadata, and latent inference.

Latent dimensions beyond the eight macros remain UI controls rather than dynamic
host parameters because DAWs expect a stable parameter list. Saved model paths
must remain available on the same machine. Smooth fallback fades, portable model
identity resolution, standalone preset/MIDI learn, and reported host latency are
not implemented yet.

## Building

A JUCE CMake package can be supplied by the system, or JUCE 7.0.9 can be fetched:

```sh
cmake -S . -B build -G Ninja -DRAVE_USE_SYSTEM_JUCE=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To enable LibTorch, include its CMake prefix. For a Python PyTorch installation:

```sh
cmake -S . -B build -G Ninja -DRAVE_USE_SYSTEM_JUCE=OFF \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Use `-DRAVE_ENABLE_LIBTORCH=OFF` for an engine/UI-only build. On macOS the
build produces:

- `build/rave_instrument_artefacts/.../RAVE Performance Instrument.app`
- `build/rave_plugin_artefacts/.../VST3/RAVE Performance Instrument.vst3`
- `build/rave_plugin_artefacts/.../AU/RAVE Performance Instrument.component`

Debug plugin bundles are ad-hoc signed after JUCE generates VST3 metadata so
local hosts can validate the complete bundle. Distribution signing and
notarization remain release steps.

## First milestone

Run one model continuously in a standalone app with:

- Live input
- Latent faders
- MIDI learn
- Dry/wet control
- Reliable preset recall

Benchmark on the Mac under realistic audio loads before adding the elaborate interface. The first milestone is successful only if the system behaves like a dependable musical instrument.
