# RAVE-05 qualification report template

Fill one copy of this template per qualification session and store it (with the
raw JSON evidence files) outside the repository tree. The automated slice is
produced by `rave_qualification` (see
[`docs/QUALIFICATION.md`](QUALIFICATION.md)); everything in part 3 is manual
work that the offline harness cannot perform. Never claim an item that was not
observed. Never copy the model fixture into the repository or into evidence
archives that leave the licensed development machine.

## 1. Session identity

| Item | Value |
| --- | --- |
| Date | `<YYYY-MM-DD, timezone>` |
| Git commit (from harness JSON `revision.commit`) | `<full SHA>` |
| Dirty state (`revision.dirty`, `dirtyEntries`) | `<clean | dirty — N entries>` |
| Build type / compiler (`environment.buildType`, `cxxCompiler`) | `<...>` |
| LibTorch / JUCE versions | `<...>` |
| Hardware (`environment.cpuModel`, `cpuArchitecture`, `logicalCores`, `memoryMb`) | `<...>` |
| OS (`environment.operatingSystem`) | `<...>` |
| Model file / size / SHA-256 (`model.path`, `sizeBytes`, `sha256`) | `<...>` |
| Hash gate (`model.hashMatchedExpected`) | `<true — expected digest recorded>` |
| Audio interface (manual sessions only) | `<device, sample rate, buffer size>` |

## 2. Automated offline results (`rave_qualification`)

Attach the harness JSON evidence files:

- Smoke: `rave-qualification-smoke-<yyyymmddThhmmssZ>.json`
- Benchmark: `rave-qualification-benchmark-<yyyymmddThhmmssZ>.json`
- Soak: `rave-qualification-soak-2h-<yyyymmddThhmmssZ>.json` + `-progress.log`

Record from each report:

| Field | Smoke | Benchmark | Soak (2 h) |
| --- | --- | --- | --- |
| Verdict (`verdict.status`) | | | |
| 1-minute load average at report | | | |
| Inference p50/p95/p99/max ms (production path) | | | |
| Callback p99/max ms per size (64…2048) | | | |
| Delayed-dry alignment mismatches / compared | | | |
| Reported latency constant at 4096 samples | | | |
| Finite output (inference + rendered) | | | |
| Telemetry counters all zero | | | |
| Overload/recovery seam probes all passed | | | |
| Peak RSS MB (start → end for soak) | | | |
| Pacing lags | | | |

Threshold interpretation: every `measured` threshold is fail-closed — a `fail`
verdict means the requirement is not met and must be investigated or the
contract revised with measured evidence. `unverified` items below are not
satisfied by any automated run.

## 3. Manual host qualification (not automated)

For each host/format, record pass/fail plus a screenshot-free observation note.
Required hosts: Logic Pro (AU), REAPER (VST3 + AU), Ableton Live (VST3).

| Item | Logic (AU) | REAPER (VST3) | REAPER (AU) | Ableton (VST3) |
| --- | --- | --- | --- | --- |
| Plug-in scans and loads | | | | |
| Editor opens; latent surface renders | | | | |
| Audio processes with the qualified model | | | | |
| Latency reported/compensated (4096 samples) | | | | |
| `dryWet` + `macro1`–`macro8` automation read/write | | | | |
| Bypass and transport start/stop | | | | |
| Editor-closed save/reopen recall | | | | |
| Multiple instances (≥2) | | | | |
| Offline/bounce render behavior | | | | |
| Sample-rate / buffer-size change reprepare | | | | |
| Model replacement and failure recovery visible in status | | | | |
| ≥30-minute interactive soak, no xruns/crashes | | | | |

## 4. Manual standalone qualification

| Item | Result / note |
| --- | --- |
| Real input/output device selection and change | |
| MIDI device disconnect/reconnect | |
| MIDI learn/clear for `dryWet` and each latent | |
| `.ravepreset` save/reopen (model + latents + mappings) | |
| Missing-model relink flow | |
| ≥30-minute interactive soak with representative input | |

## 5. Soak observations (2-hour automated + interactive)

- Automated soak: verdict, telemetry deltas over time (from progress log), peak
  RSS trajectory, any deadline misses/queue drops with wall-clock timestamps.
- Interactive soaks per host: duration, xruns observed, CPU/memory observations,
  controlled synthetic overload and recovery behavior, editor open/close and
  transport churn.

## 6. Residual limitations to restate in the release record

- An in-process inference worker cannot contain a native LibTorch crash and
  cannot forcibly cancel a stalled native inference call.
- Measured latency is the contractual 4096-sample transport latency; intrinsic
  receptive-field and perceptual latency are human observations.
- Results identify the exact revision, model hash, and hardware; they do not
  transfer to other machines without a repeat run.
