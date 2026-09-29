# Source and third-party licensing

Project-owned RAVE Performance Instrument source code is licensed under
**GPL-3.0-or-later**; see [LICENSE](LICENSE). Distribution of the plugin must follow
that license, including its corresponding-source requirements.

This project is an independent instrument using RAVE-compatible model exports.
RAVE was developed by the ACIDS team at IRCAM; see the
[original project](https://github.com/acids-ircam/RAVE). We do not claim to have
invented RAVE or trained the bundled third-party models.

The source license does not replace third-party terms:

- JUCE 7.0.9 is used under its GPLv3 option. See the upstream JUCE license.
- PyTorch/LibTorch retains its own license and dependency notices, included in
  staged packages.
- Original icon artwork retains its [MIT license](assets/icon/LICENSE.md).
- Model weights and upstream model cards retain their
  [individual licenses and attribution](assets/models/README.md), including
  CC BY-NC 4.0 for Birds and VocalSet. They are separate data assets, not GPL
  licensed weights. The starter pack includes non-commercial assets.

Open-source code does not automatically make bundled data commercially usable.
