# Factory starter models

This pack contains unmodified pretrained exports. Birds, Voice and full Freesound Loops run at **48 kHz**. **Freesound Loops Lite runs at 44.1 kHz** and is the recommended starting point on modest CPUs. All use 2048-sample inference frames. Match the standalone audio device or DAW session to the selected model rate. No training is needed.

| Model | Creator / attribution | Model license |
| --- | --- | --- |
| Freesound Loops Lite | Tangible Music Lab; optimized FSL10K export | MIT, as declared by its publisher |
| Birds — Pluma | Intelligent Instruments Lab; curated by Giacomo Lepri for Pluma | CC BY-NC 4.0 |
| Voice — VocalSet | Intelligent Instruments Lab; trained on VocalSet | CC BY-NC 4.0 |
| Freesound Loops | Tangible Music Lab; trained on FSL10K | MIT, as declared by its publisher |

**Birds and VocalSet are provided for non-commercial use.** Preserve attribution and license notices when sharing. The plugin source license is separate from the model licenses. This pack does not grant rights to the original training recordings or promise that all generated outputs are unrestricted.

Sources: [IIL models](https://huggingface.co/Intelligent-Instruments-Lab/rave-models), [Tangible Music Lab models](https://huggingface.co/Tangible-Music-Lab/RAVE_models). The adjacent upstream model cards preserve dataset descriptions and credits. `manifest.json` pins revisions, download URLs, sizes and SHA-256 hashes; downloaded `.ts` weights are ignored by Git and included in staged native artifacts.

Fetch with `python scripts/models.py fetch`; verify with `python scripts/models.py verify`. Model weights are never silently downloaded by the plugin at runtime.

The three 48 kHz models are CPU intensive. Preliminary measurements on the development PC exceeded real-time deadlines for the larger exports; bundling is not a performance guarantee. Use Lite first, and monitor the host for dropouts.
