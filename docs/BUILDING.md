# Native builds and artifacts

`VERSION` is the single source for CMake, plugin metadata and the standalone app. Advance it for each delivered implementation: patch for fixes/build changes, minor for compatible features, major for incompatible changes. Keep the `RvAI` / `RvPI` plugin codes and `org.raveperformance.instrument.plugin` bundle ID stable so saved sessions continue to resolve the plugin.

## Requirements

Build natively on each operating system, with Python 3.12, CMake 3.22 or newer, Git and a C++20 compiler. Windows uses Visual Studio 2022 Build Tools (Desktop C++ workload and Windows SDK); macOS uses Xcode command-line tools; Linux requires GCC, ALSA, X11, Xrandr, Xinerama, Xcursor and FreeType development packages. JUCE 7.0.9 is fetched automatically; `--juce-source PATH` uses an existing checkout for offline builds.

Install CPU PyTorch in the selected Python environment before building. Tested versions are 2.14.0 on Linux/Windows and 2.2.2 on Intel macOS. macOS builds currently qualify Intel x86_64 only; Apple Silicon and universal packages require separate qualification. A requested Torch build fails if Torch is missing. `--without-torch` explicitly builds the limited backend-free variant.

Install repository tooling with `python -m pip install -r requirements-dev.txt` and `npm ci`. Linux packaging requires the pinned `patchelf`; icon regeneration requires `rsvg-convert` (librsvg). On Windows use `--icon-assets-only` to verify committed icon assets; Linux/macOS perform full regeneration. The reduced Windows check is reported explicitly.

## Build and test

```sh
./build.sh --test
./build.sh --test --stage build/staging/0.3.1-linux-x86_64
```

On Windows, activate the Python environment and run:

```powershell
.\build.ps1 --test --icon-assets-only --stage build/staging/0.3.1-windows-x64
```

On macOS use `./build.sh` with the same arguments as Linux and an appropriate staging name. `PYTHON` selects the shell wrapper's interpreter; `RAVE_PYTHON` selects the PowerShell interpreter. Activate an environment that places CMake on PATH. `--config Debug`, `--jobs 3`, and `--build-dir PATH` are supported; defaults are Release, three jobs, and `build/<version>-<system>-<architecture>-<configuration>`.

Builds stay in the repository. Staging is opt-in and always runs the full CTest suite first. It requires a fresh destination and never installs a plugin. Source identity is captured during configuration; staging refuses changed source inputs. Do not edit sources while a build is running. Direct CMake builds remain available, but use the wrapper for the tested build/test/stage sequence.

## Package and relocation

Packages contain VST3 and standalone artifacts, plus AU on macOS, runtime dependencies, notices, and `build-identity.json`. The manifest records version, source revision/dirty state/content hash, compiler, architecture, Torch version, runtime origins, and file checksums. These are local integrity/provenance checks, not a cryptographic publisher signature.

Staging copies runtime libraries privately, rewrites Linux/macOS runtime paths, and scans a relocated VST3 package with development library environment variables removed. Linux additionally checks dependency resolution. Windows uses a small VST3 entry DLL to load its private Torch implementation with `LoadLibraryExW` scoped to the implementation directory and system libraries; it does not modify the DAW's PATH. The Microsoft Visual C++ runtime remains a Windows prerequisite. macOS bundles receive ad-hoc signatures and strict verification after relocation edits; Developer ID signing/notarization is a separate distribution step.

```sh
python scripts/artifacts.py verify build/staging/0.3.1-linux-x86_64
```

Relocation scans verify loading and metadata. They do not replace real DAW/audio/model performance qualification documented in [QUALIFICATION.md](QUALIFICATION.md). These packages do not claim new model qualification or production signing. Review third-party licensing before redistribution.

## Explicit installation

Close hosts using RAVE before replacing it. Installation first verifies the complete package and copies each artifact to a temporary sibling. Existing installations are preserved unless `--replace` is provided; a failed replacement rolls back, retaining backups if restoration itself fails. Other plugins are never removed.

```sh
# Exercise installation without touching host plugin directories:
python scripts/artifacts.py install build/staging/0.3.1-linux-x86_64 --install-root build/install-check

# Explicitly install the verified package into native discovery directories:
python scripts/artifacts.py install build/staging/0.3.1-linux-x86_64
# Add --replace only to replace the same RAVE artifact.
```

| Platform | VST3                            | AU                                    | Standalone                                            |
| -------- | ------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| Linux    | `~/.vst3`                       | —                                     | `~/.local/lib/rave-performance-instrument/<version>`  |
| macOS    | `~/Library/Audio/Plug-Ins/VST3` | `~/Library/Audio/Plug-Ins/Components` | `~/Applications`                                      |
| Windows  | `%CommonProgramFiles%\VST3`     | —                                     | `%LOCALAPPDATA%\Programs\RAVE Performance Instrument` |

Windows VST3 installation may require an elevated shell. macOS uses standard user discovery directories; system-wide deployment is a separate deliberate operation. Retain the staged package and its identity manifest as the installation record. Never install the raw CMake output, whose dependencies still point at development locations.

## Formatting and CI

Prettier is pinned locally. `npm run format` / `npm run format:check` operate on changed supported files relative to `HEAD` (or `FORMAT_BASE`), excluding dependencies and generated files. Husky/lint-staged formats staged supported files while preserving partially staged work. Python uses pinned Black; CMake uses cmake-format; C++ uses clang-format. Format changed C++ ranges only until a separate formatting baseline is reviewed. Shell and PowerShell wrappers currently have syntax checks but no configured formatter; those formats remain a formatting setup gap.

CI checks changed-file formatting and builds/tests/stages Linux, Windows and Intel macOS packages, then exercises installation into a disposable directory. CI artifact upload retains tar archives preserving executable permissions; it does not publish a release or install on a host. A successful build alone is not evidence that DAW qualification or signing has passed.

## Factory models and generator

Before staging a Torch-enabled package, run `python scripts/models.py fetch`.
This downloads four pinned pretrained exports (about 380 MB) and verifies their
SHA-256 hashes. Stage fails if a model is missing or modified. Every standalone,
VST3 and AU artifact receives its own `Models` resource folder with the manifest,
upstream model cards and license notices, so installation and relocation retain
the library without depending on a developer checkout. Weights stay outside Git.

In **Load model**, choose Freesound Loops Lite and configure the device/DAW at **44.1 kHz** for a lighter starting point. Birds, Voice and full Freesound Loops require **48 kHz** and more CPU. Enable **Generate without audio input**, turn Dry / Wet toward
100%, then adjust latent knobs, Motion depth and Motion rate. Depth zero gives a
stationary latent vector. Generation calls the decoder directly, bypassing the
encoder; it is not a trained autoregressive prior, oscillator or waveshaper.
Independent slow sine modulation moves the latent dimensions. Generated output
is soft-limited to 0.25 full scale; failed frames fall back to silence. Generation
starts off and is saved with host state / standalone presets.

Birds and VocalSet retain **CC BY-NC 4.0** terms. Freesound Loops carries its
publisher's MIT declaration. See [model notices](../assets/models/README.md).

Staging also runs the native decoder smoke test for every factory model before
copying artifacts. This checks loading, finite output and non-silent latent-only
generation; reported timings are measurements, not real-time certification.
The plugin advertises an infinite tail because Generate can sustain indefinitely;
choose an explicit render duration when exporting generated audio from a DAW.

On Windows the wrapper limits per-project compiler parallelism to one while
`--jobs` bounds concurrent build projects, preventing nested `/MP` compilation
from multiplying the requested worker budget. See [Microsoft's build parallelism
guidance](https://devblogs.microsoft.com/visualstudio/tuning-c-build-parallelism-in-vs2010/).
