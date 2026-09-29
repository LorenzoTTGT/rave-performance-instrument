---
license: cc-by-nc-4.0
---

# RAVE Models

This is a collection of [RAVE](https://github.com/acids-ircam/RAVE) models trained by the [Intelligent Instruments Lab](https://iil.is) for various projects. 

For a full description see our blog post at: https://iil.is/news/ravemodels, and for more about RAVE, see the original [paper](https://arxiv.org/abs/2111.05011) from IRCAM.

Most of these models are encoder-decoder only, no prior, and all use the `--causal` mode and are exported for streaming inference with [nn~](https://github.com/acids-ircam/nn_tilde), [NN.ar](https://github.com/elgiano/nn.ar) or [rave-supercollider](https://github.com/victor-shepardson/rave-supercollider).

In the `checkpoints/` directory are some complete checkpoints which can be used with our [fork of RAVE](https://github.com/victor-shepardson/RAVE) to speed up training by transfer learning.

Citation:

```
@misc {intelligent_instruments_lab_2023,
	author       = { {Intelligent Instruments Lab} },
	title        = { rave-models (Revision ad15daf) },
	year         = 2023,
	url          = { https://huggingface.co/Intelligent-Instruments-Lab/rave-models },
	doi          = { 10.57967/hf/1235 },
	publisher    = { Hugging Face }
}
```

## Musical Instruments

### guitar_iil_b2048_r48000_z16.ts

Dataset: [IILGuitarTimbre](https://github.com/Intelligent-Instruments-Lab/IILGuitarTimbre), a timbre-oriented collection of plucking, strumming, striking, scraping and more recorded dry from an electric guitar.

Model: modified RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### sax_soprano_franziskaschroeder_b2048_r48000_z20.ts

Dataset: Soprano sax improvisation by [Franziska Schroeder](https://improvisationai.wordpress.com/).

Model: modified RAVE v1, 48kHz, block size 2048, 20 latent dimensions.

### organ_archive_b2048_r48000_z16.ts  

Dataset: various recordings of organ music sourced from archive.org. Small amounts of voice and other instruments were included, and vinyl record noises are prominent.

Model: modified RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### organ_bach_b2048_sr48000_z16.ts

Dataset: various recordings of J.S. Bach music for church organ.

Model: modified RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### mrp_strengjavera_b2048_r44100_z16.ts

Dataset: [magnetic resonator piano](https://andrewmcpherson.org/project/mrp) controlled by [artificial life](https://github.com/Intelligent-Instruments-Lab/iil-python-tools/tree/ja-dev/tolvera), as part of generative installation Strengjavera by Jack Armitage premiered at AIMC 2023. See [paper](https://aimc2023.pubpub.org/pub/83k6upv8) and [Zenodo](https://zenodo.org/records/8329855) for citation.

Model: RAVE v3, 44.1kHz, block size 2048, 16 latent dimensions.

## Voice

### voice_vocalset_b2048_r48000_z16.ts

Dataset: [VocalSet](https://zenodo.org/record/1193957) singing voice dataset.

Model: modified RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### voice_hifitts_b2048_r48000_z16.ts

Dataset: [Hi-Fi TTS](https://www.openslr.org/109/) audiobooks dataset.

Model: modified RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### voice_jvs_b2048_r44100_z16.ts

Dataset: [Hi-Fi TTS](https://www.openslr.org/109/) speaker 9017 (John Van Stan).

Model: RAVE v3, 44.1kHz, block size 2048, 16 latent dimensions.

### voice_vctk_b2048_r44100_z22.ts

Dataset: [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443) multispeaker read speech dataset.

Model: RAVE v3, 44.1kHz, block size 2048, 22 latent dimensions.

### voice_multivoice_b2048_r48000_z11.ts

Dataset: combination of speaking and singing voice datasets: [CSTR VCTK Corpus](https://datashare.ed.ac.uk/handle/10283/3443), [VocalSet](https://zenodo.org/record/1193957), [Children's Song Dataset](https://zenodo.org/records/4785016), [NUS-48E](https://ieeexplore.ieee.org/document/6694316/), [attHACK](https://arxiv.org/abs/2004.04410).

Model: RAVE v3 with spectral discriminator, 48kHz, block size 2048, 11 latent dimensions.

## Birds

### birds_motherbird_b2048_r48000_z16.ts

This model of bird sounds was curated by Manuel Cherep, Jessica Shand and Jack Armitage for their piece Motherbird, performed at TENOR 2023 in Boston, May 2023.

Dataset: bird sounds.

Model: RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### birds_pluma_b2048_r48000_z12.ts

This model of bird sounds was curated by Giacomo Lepri for his instrument *[Pluma](http://www.giacomolepri.com/pluma)*

Dataset: bird sounds.

Model: modified RAVE v1, 48kHz, block size 2048, 12 latent dimensions.

## *Pond Brain* Marine Sounds

These models of marine sounds were trained for [Jenna Sutela](https://jennasutela.com/)'s *Pond Brain* installations at [Copenhagen Contemporary](https://copenhagencontemporary.org/en/yet-it-moves-read-online/) and the [Helsinki Biennial](https://helsinkibiennaali.fi/en/artist/jenna-sutela/)

Caution: these decoders sometimes produce a loud chirp on first initialization.

### water_pondbrain_b2048_r48000_z16.ts

Dataset: water recordings from freesound.org.
<details>
<summary>list of freesound users</summary>
inspectorj, inchadney, aesqe, vonfleisch, javetakami, atomediadesign, kolezan, zabuhailo, zaziesound, repdac3, al_sub, lgarrett, uzbazur, lydmakeren, frenkfurth, edo333, boredtoinsanity, owl, kaydinhamby, tliedes, ilmari_freesound, manoslindos, l3ardoc, alexbuk, s-light
</details>

Model: modified RAVE v1, 48kHz, block size 2048, 16 latent dimensions.

### humpbacks_pondbrain_b2048_r48000_z20.ts

Dataset: humpback whale recordings from the [Watkins database](https://cis.whoi.edu/science/B/whalesounds/index.cfm), [MBARI](https://freesound.org/people/MBARI_MARS/), and BBC.

Model: modified RAVE v1, 48kHz, block size 2048, 20 latent dimensions.

### marinemammals_pondbrain_b2048_r48000_z20.ts

Dataset: various marine mammal sounds from [NOAA](https://www.fisheries.noaa.gov/national/science-data/sounds-ocean-mammals), the [Watkins database](https://cis.whoi.edu/science/B/whalesounds/index.cfm), freesound users `felixblume` and `geraldfiebig`, and sound effects databases.

Model: modified RAVE v1, 48kHz, block size 2048, 20 latent dimensions.


## *Thales* magnets_b2048_r48000_z8.ts

Dataset: One hour recording of magnets of different dimensions hitting each other or scratching wooden and metallic surfaces. Used for [Thales](https://iil.is/pdf/2023_nime_privato_et_al_thales.pdf), a musical instrument based on magnets 

Model: RAVE v1, 48Khz, block size 2048, 8 latent dimensions.


## *Crozzoli's Music* crozzoli_bigensemblesmusic_18d.ts

Dataset: Six recordings of long contemporary compositions for electronic and acoustic big ensembles.

Model: RAVE v3, 48Khz, block size 2048, 18 latent dimensions.


## *Aulus-les-Bains Dawn Chorus @ CAMP* birds_dawnchorus_b2048_r48000_z8.ts

Dataset: ~230 minutes of dawn chorus recorded by Gregory White at Aulus-les-Bains as part of a residency at CAMPfr.com.

Model: RAVE v3, 48Khz, block size 2048, 8 latent dimensions.



