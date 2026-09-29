---
license: mit
---
# RAVE Models by Tangible Music Lab

This is a collection of RAVE (Realtime Audio Variational autoEncoder) models trained by the Tangible Music Lab for audio generation and transformation. The aim of this repository is to provide musicians with pre-trained models for building embedded RAVE models on the Raspberry Pi platform or similar, for physical hardware and tangible interface development for sound and music experimentation. These models enable real-time audio manipulation and generation while being optimized for resource-constrained environments, making them ideal for interactive musical instruments and sound installations.


## Model Description

- **Developed by:** Tangible Music Lab
- **Model type:** RAVE (Realtime Audio Variational autoEncoder) 
- **License:** MIT

## Model Sources

- **Repository:** https://huggingface.co/Tangible-Music-Lab/rave_models
- **Training Code:** https://github.com/victor-shepardson/RAVE


## Direct Use

These models are designed for real-time audio generation and transformation. They can be used with:
- nn~
- NN.ar
- rave-supercollider

### Models

#### tam_freesoundloop10k_default_b2048_r48000_z16.ts
- Dataset: Freesound Loop Dataset
- Model: RAVE v3 with default configuration 
- Latent dimensions: 16
- Sample rate: 48kHz

#### tam_freesoundloop10k_raspi_b2048_r44100_z16.ts
- Dataset: Freesound Loop Dataset
- Model: Modified RAVE v3, optimized for Raspberry Pi 5
- Latent dimensions: 16
- Sample rate: 44.1kHz
- Special features: Scaled down for real-time performance on RPi 5

## Features
- All models are exported for streaming inference
- Compatible with nn~, NN.ar, and rave-supercollider
- Models focus on encoder-decoder architecture without prior networks
- Training checkpoints provided for transfer learning 
- For training, use the Intelligent Instruments Lab RAVE fork: https://github.com/victor-shepardson/RAVE

## Training Details

### Training Data

The models were trained on the [Freesound Loop Dataset (FSL10K)](https://zenodo.org/records/3967852), a comprehensive collection of musical loops curated for machine learning applications. The dataset consists of 9,455 loops from Freesound.org.
All sounds in the dataset are licensed under various Creative Commons licenses.

### Training Procedure

Training checkpoints are provided for both models to enable transfer learning on custom datasets.

## Citation

```bibtex
@misc {tangible_music_lab_2025,
   author       = { {Tangible Music Lab} },
   title        = { RAVE Models },
   year         = 2025,
   url          = { https://huggingface.co/Tangible-Music-Lab/rave_models },
   publisher    = { Hugging Face }
}