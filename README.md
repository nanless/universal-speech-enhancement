______________________________________________________________________

<div align="center">

# Universal Speech Enhancement

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Features

- Speech signal quality enhancement under all distortion conditions
- one model for all speech enhancement tasks: noise suppression, dereverberation, equalization, packet loss concealment, bandwidth extension, declipping, and others
- GAN-based and score diffusion-based approaches for training and inference
- easy-to-use interface for training and inference
- 24kHz sampling rate pipeline

## Description

This repository contains the code for training and inference code for the universal speech enhancement model using GAN-based and score diffusion-based approaches. 

Universal speech enhancement aims to improve speech signals recorded under various adverse conditions and distortions, including noise, reverberation, clipping, equalization (EQ) distortion, packet loss, codec loss, bandwidth limitations, and other forms of degradation. A comprehensive universal speech enhancement system integrates multiple techniques such as noise suppression, dereverberation, equalization, packet loss concealment, bandwidth extension, declipping, and other enhancement methods to produce speech signals that closely approximate studio-quality audio.

The model is trained on the clean speech from [EARS dataset](https://github.com/facebookresearch/ears_dataset.git) and noise signal from [DNS5 dataset](https://github.com/microsoft/DNS-Challenge.git). The sampling rate used in the project is 24 kHz. The input audio will be resampled to 24 kHz before processing, and the output audio is still 24 kHz.

## Demo Page

https://nanless.github.io/universal-speech-enhancement-demo
It may take a few minutes to load the audio and image files in the demo page.

## Installation

#### Pip

```bash
# install pytorch (I use rocm for GPU training, but you can use CUDA if you have it)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
# install requirements
pip install -r requirements.txt
```

## How to run

### Training

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=SGMSE_Large
```

### Inference

Predict with trained model

Download the pretrained SGMSE model: https://huggingface.co/nanless/universal-speech-enhancement-SGMSE/blob/main/use_SGMSE.ckpt

Download the pretrain LSGAN model: https://huggingface.co/nanless/universal-speech-enhancement-LSGAN/blob/main/use_LSGAN.ckpt


```bash
python src/predict.py data.data_folder=<path/to/test/folder> data.target_folder=<path/to/output/folder> model=SGMSE_Large ckpt_path=<path/to/trained/model>
```

## References

This repository is developed based on the following repositories.

https://github.com/sp-uhh/sgmse

https://github.com/facebookresearch/ears_dataset

https://github.com/microsoft/DNS-Challenge

https://github.com/ashleve/lightning-hydra-template

