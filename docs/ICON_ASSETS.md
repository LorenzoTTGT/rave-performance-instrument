# Original icon assets (RAVE-06)

The RAVE Performance Instrument uses a project-owned original icon, not the
official RAVE or IRCAM mark. The user explicitly authorized this substitution
after rejecting the official mark. Provenance, the artwork-only MIT license,
and the non-affiliation statement are in
[`assets/icon/PROVENANCE.md`](../assets/icon/PROVENANCE.md) and
[`assets/icon/LICENSE.md`](../assets/icon/LICENSE.md).

## Source and generation

The single canonical source is:

```text
assets/icon/rave-instrument-icon.svg
```

Its generated square RGBA derivatives are 16, 32, 64, 128, 256, 512, and 1024
px PNGs under `assets/icon/`. The 16/32 pair is the representative small and
2x-small output; 512/1024 are representative large and 2x-large output. The
artwork retains a transparent outer margin at every size and is never
stretched because each output is rendered from the square SVG view box at an
explicit square size.

Generation requires Python 3 and `rsvg-convert` (librsvg), both used only by
the checked-in asset tool:

```sh
python3 assets/icon/generate_icon.py --write
python3 assets/icon/generate_icon.py --check
```

`--write` renders every derivative, validates its dimensions/RGBA format, and
updates `assets/icon/SHA256SUMS`. `--check` verifies the checked-in dimensions
and SHA-256 values (including transparent and opaque alpha samples), re-renders in a temporary directory, and compares the bytes. Review a renderer upgrade deliberately before running `--write` because
it may change antialiasing bytes.

## UI and packaging

CMake uses inspected JUCE `ICON_BIG`/`ICON_SMALL` target facilities with the
1024 px and 32 px derivatives. JUCE's macOS bundle path creates an `Icon.icns`
and records it in each generated `Info.plist`; this applies to the standalone
application and VST3/AU wrapper targets. The canonical SVG is additionally
embedded as JUCE BinaryData for the standalone and plugin editors and copied
as `rave-instrument-icon.svg` to each supported macOS bundle's `Contents/Resources`.
The visual uses an accessible title, “RAVE Performance Instrument original
icon”, while the surrounding product title remains readable text.

After the configured no-LibTorch build, inspect all generated resources and
metadata without desktop automation:

```sh
find build -path '*RAVE Performance Instrument.app/Contents/Resources/Icon.icns' -o \
  -path '*RAVE Performance Instrument.vst3/Contents/Resources/Icon.icns' -o \
  -path '*RAVE Performance Instrument.component/Contents/Resources/Icon.icns'
find build -name Info.plist -print -exec plutil -extract CFBundleIconFile raw -o - {} \;
codesign --verify --strict --verbose=2 "<VST3 bundle>"
codesign --verify --strict --verbose=2 "<AU bundle>"
```

The deterministic checks establish dimensions, alpha-capable PNG format,
hashes, and bundle placement. A human should still visually confirm contrast
and recognizability in the standalone app and a host editor at the intended
system scaling.
