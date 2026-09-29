# RAVE Performance Instrument

## Direction

Build a native RAVE performance instrument in C++ with JUCE, using LibTorch initially to run existing RAVE models. The application should have a shared audio engine that can later target standalone, VST3, and AU formats.

The instrument should let a performer load a model, route audio through it, explore its latent space, assign modulation and MIDI, and save the complete setup.

RAVE-04 adds bounded standalone `.ravepreset` sessions (model path, all latents, dry/wet, MIDI mapping/input, and audio identities), visible standalone audio/MIDI selection, MIDI learn/clear for every latent, and missing-model relinking. RAVE-06 adds a project-owned original latent-portal icon to both editors and supported macOS bundle metadata/resources; it intentionally does not use the official RAVE/IRCAM mark. Real-device and DAW observation remain RAVE-05 work.

### Fixed realtime transport

Audio is accumulated into 2048-sample inference frames and rendered on a
model-independent 4096-sample timeline. The plugin reports 4096 samples in all
model and overload states. Each frame commits wet or aligned delayed dry once, 5 ms before playback;
exact-deadline results are accepted and later results cannot upgrade it.
Wet-to-dry fades during that pre-playback window, while dry-to-wet fades over
the first 5 ms of playback. Release invalidates readiness and all timeline
history before any detached callback can render it. Dry/wet target smoothing
remains independent. This
contractual transport latency does not measure intrinsic model receptive-field
or perceptual latency.

Runtime counters reset on prepare. Deadline misses count eligible frames absent
at their pre-playback commitment deadline; queue drops combine worker input/output enqueue failures;
late results arrived after their boundary; processing/reset errors count
backend failures; alignment errors count malformed, duplicate, impossible
future results, and invalid layouts; duplicate identity takes precedence even
after commitment, while a first post-deadline result is late. Each prepare
starts a fresh counter epoch. Rates must be finite and positive, with a 5 ms
fade no longer than the 2048-sample quantum. Lifecycle status separately
remains the authoritative model-usability state.

Reference projects:

- [IRCAM RAVE](https://github.com/acids-ircam/RAVE)
- [Neutone models](https://neutone.jp/fx/models)
- [JUCE](https://juce.com/)
- [nn~ documentation](https://github.com/acids-ircam/nn_tilde)
- [CPAL](https://github.com/RustAudio/cpal)
- [egui](https://github.com/emilk/egui)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

## Proposed stack

| Layer                            | Recommendation                  | Rationale                                                           |
| -------------------------------- | ------------------------------- | ------------------------------------------------------------------- |
| Application and plugin framework | JUCE + C++20                    | Native audio, MIDI, UI, standalone, VST3, and AU targets            |
| Initial model runtime            | LibTorch, CPU inference         | Closest compatibility with exported RAVE TorchScript models         |
| Interface                        | Custom JUCE components          | Faders, knobs, XY pads, meters, and modulation displays             |
| Conventional DSP                 | JUCE DSP plus custom processors | Filtering, gain, envelopes, saturation, delay, and mixing           |
| Routing                          | Small explicit processing graph | Performance-oriented flexibility without a patching language        |
| State                            | Versioned presets and sessions  | Model identity, controls, routing, MIDI, and modulation assignments |
| Build                            | CMake                           | Shared engine with standalone and plugin targets                    |

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
- Standalone `.ravepreset` Save/Load controls with fail-closed versioned state,
  message-thread device application, one selected MIDI callback, scrolling
  arbitrary latent MIDI learn/clear controls, and an explicit missing-model
  Relink action. Latest standalone model/preset/relink requests are generation
  gated so stale loader results cannot activate.
- CTest coverage for queue behavior and, when Python Torch and LibTorch are
  configured, end-to-end TorchScript loading, metadata, and latent inference.

Latent dimensions beyond the eight macros remain UI controls rather than dynamic
host parameters because DAWs expect a stable parameter list. Presets retain a
missing same-machine model path and require explicit relinking; portable model
identity resolution remains deferred.

## Building

Use the native build wrapper: `./build.sh --test` on Linux/macOS or `.\build.ps1 --test --icon-assets-only` on Windows. `VERSION` controls all artifact versions. Builds remain local; staging and installation are separate explicit actions.

See [native build and artifact procedures](docs/BUILDING.md) for prerequisites, packaging, runtime relocation, installation destinations and formatting checks.

### Icon assets

The original project-owned icon's canonical SVG, MIT artwork license,
provenance, deterministic PNG generation, and bundle-inspection procedure are
in [`docs/ICON_ASSETS.md`](docs/ICON_ASSETS.md). Verify checked-in derivatives
without desktop automation with:

```sh
python3 assets/icon/generate_icon.py --check
```

### RAVE-05 qualification harness

LibTorch builds also provide `rave_qualification`, an offline command that
benchmarks the TorchScript backend and the shared `RaveAudioEngine` processing
path with a hash-gated local model fixture (the fixture stays in the ignored
`.qualification-models/` directory and is never committed):

```sh
cmake -S . -B build-qual -G Ninja \
  -DCMAKE_DISABLE_FIND_PACKAGE_JUCE=TRUE \
  -DRAVE_ENABLE_LIBTORCH=ON \
  -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build-qual --parallel

/usr/bin/shasum -a 256 .qualification-models/birds_pluma_b2048_r48000_z12.ts
# must print a12ad61a2b0b5ee2329a72993bd94386a571600b37dd23feaa0a404940468d68

./build-qual/rave_qualification --mode smoke \
  --model .qualification-models/birds_pluma_b2048_r48000_z12.ts \
  --expect-sha256 a12ad61a2b0b5ee2329a72993bd94386a571600b37dd23feaa0a404940468d68 \
  --json evidence/rave-qualification-smoke.json
```

The harness measures direct inference and paced real-time callback timing over
the 64–2048 callback matrix plus alternating partitions, using complete
callbacks even in smoke mode, checks finite output, the fixed 4096-sample
reported latency, a post-latency delayed-dry verification window, and
zero-counter runtime telemetry, exercises deterministic overload/recovery
semantics through the engine's test seam, and offers `--mode soak` with a
default 7200-second (2-hour) bounded-memory run. Thresholds are explicit and
fail-closed; timing thresholds require a quiet machine, and every report
records the 1-minute load average, the exact git revision and dirty state, the
model hash, and the hardware/configuration identity.

Limitations: this is an offline, non-realtime harness. It never observes audio
devices, hosts, or DAWs, so host/device xruns, device I/O latency, wet-path
impulse and perceptual latency, DAW scanning/automation/recall, real-device
standalone behavior, and interactive soaks remain unverified by it and require
the manual checklist in [`docs/QUALIFICATION.md`](docs/QUALIFICATION.md). The
harness is not registered with CTest, so the long soak can never run inside
ordinary testing. See
[`docs/RAVE05_QUALIFICATION_REPORT_TEMPLATE.md`](docs/RAVE05_QUALIFICATION_REPORT_TEMPLATE.md)
for the evidence template.

## First milestone

Run one model continuously in a standalone app with:

- Live input
- Latent faders
- MIDI learn
- Dry/wet control
- Reliable preset recall

Benchmark on the Mac under realistic audio loads before adding the elaborate interface. The first milestone is successful only if the system behaves like a dependable musical instrument.

## Compact performance interface

The standalone app and plugin share a dark interface. The standalone opens at
700 × 480; the plugin at 680 × 460. The plugin shows up to 16 latent controls as
compact sliders within that default window. Larger latent spaces and the optional
MIDI mapping controls can scroll independently of the dry/wet mix. Numeric values
remain editable, mix is displayed as a percentage, and double-click resets a knob.

Use **MIDI** to reveal learning assignments. In the standalone, **Audio** opens a
scrollable panel for audio devices and MIDI input selection. Preset load/save and
missing-model relinking remain on the main toolbar. Full status messages are also
available as tooltips. The redesign preserves host parameter IDs, MIDI mappings,
preset compatibility and the existing model lifecycle.

The plugin tests include in-process layout and interaction checks for both interfaces.
Set `RAVE_UI_PREVIEW_DIR` to an absolute local directory when running them to save
component renders without opening or automating desktop windows.

### Starter models and input-free synthesis

Packaged builds include Birds (Pluma), Voice (VocalSet), and two Freesound Loops exports.
Start with **Freesound Loops Lite at 44.1 kHz** on modest CPUs. The other three
models require **48 kHz** and more CPU. Select them from **Load model**.
To synthesize without an audio source, enable **Generate without audio input**,
raise **Dry / Wet**, and move the latent knobs. **Motion depth** and **Motion rate**
add slow independent latent modulation; zero depth holds the chosen latent vector.

The decoder produces audio directly; no separately trained prior is required.
Generator settings are saved in presets/host state and default to off.
Birds and Voice are non-commercial models; see the
[starter pack licenses and credits](assets/models/README.md).
Build instructions include the pinned model fetch step.

Project-owned plugin code is **GPL-3.0-or-later**. See [COPYING.md](COPYING.md) for
third-party terms and the distinction between plugin and model licensing.
