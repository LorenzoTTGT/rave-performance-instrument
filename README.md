# RAVE Performance Instrument

## Direction

Build a native RAVE performance instrument in C++ with JUCE, using LibTorch initially to run existing RAVE models. The application should have a shared audio engine that can later target standalone, VST3, and AU formats.

The instrument should let a performer load a model, route audio through it, explore its latent space, assign modulation and MIDI, and save the complete setup.

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
- Measure model, buffering, resampling, and device latency; compensate the dry path and report plugin latency accurately.
- Consider an optional separate inference process later if unattended installations become important, since a worker thread does not contain native runtime crashes.

## Runtime strategy

Keep inference behind a replaceable backend interface. Begin with LibTorch and CPU inference. Investigate ONNX Runtime for selected models only after proving streaming-state compatibility and output equivalence; existing `.ts` model loading should not be assumed.

## Current implementation

The repository currently provides:

- A JUCE standalone application with live audio pass-through and dry/wet UI.
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
- A native model chooser backed by off-thread loading; activating a loaded model
  safely detaches and restarts the audio callback around backend replacement.
- CTest coverage for queue behavior and, when Python Torch and LibTorch are
  configured, end-to-end TorchScript loading, metadata, and latent inference.

Smooth fallback fades, MIDI learn, preset recall, and reported host latency are
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

Use `-DRAVE_ENABLE_LIBTORCH=OFF` for an engine/UI-only build.

## First milestone

Run one model continuously in a standalone app with:

- Live input
- Latent faders
- MIDI learn
- Dry/wet control
- Reliable preset recall

Benchmark on the Mac under realistic audio loads before adding the elaborate interface. The first milestone is successful only if the system behaves like a dependable musical instrument.
