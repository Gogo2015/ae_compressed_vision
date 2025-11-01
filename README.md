# Quantized Autoencoder for Video Compression

This repository contains the implementation and results from my research on a **quantized convolutional autoencoder** for video compression. The model compresses short grayscale video sequences into discrete latent codes using a tunable quantization codebook, enabling control over bitrate and reconstruction fidelity. This project explores the classic rate–distortion trade-off in neural video compression.

📄 [Read the full IEEE-style technical paper (AutoencoderPaper.pdf)](./AutoencoderPaper.pdf)
*Author: Gaurav Mitra (University of Texas at Austin)*

## Overview

- Implements a 3D convolutional autoencoder with quantized latent space
- Enables lossy video compression through discrete codebooks of varying size
- Trains and evaluates on the Moving MNIST dataset
- Includes visualizations of reconstructed frames and a rate–distortion curve
- Designed for research on learned video codecs and real-time compression

## Core Features

- **Encoder**: 3D CNN that downsamples spatial resolution
- **Decoder**: Transposed convolutions reconstruct frames from latent code
- **Quantization**: Latent features are mapped to the nearest codebook entry
- **Control**: Bitrate is adjusted by selecting codebook size ( K = {2, 4, 8, 16, 32, 64, 128})

## Getting Started

```bash
git clone https://github.com/Gogo2015/quantized-autoencoder-video-compression.git
cd quantized-autoencoder-video-compression
pip install -r requirements.txt
```

### Train a model:
```bash
python train.py --codebook_size 32
```

### Visualize reconstructions:
```bash
python visualize.py --model_path models/codebook_32.pth
```

## Results Summary

- Clear rate–distortion tradeoff: larger codebooks improve reconstruction
- Visual results included for each codebook size
- Quantitative MSE vs. bits-per-pixel trend shows diminishing returns at higher K
- Limitations: early stopping criteria caused undertraining in large-K models

## In Progress(Hiatus): IP-Frame Autoencoder Extension

I was extending this work to support **real-time video compression** using an IP-frame strategy:
- **I-frames**: Encoded using the existing quantized autoencoder
- **P-frames**: Predicted from prior latent states using temporal interpolation
- Targeted toward low-latency applications like streaming, robotics, or embedded systems

Updates will be pushed to the `ip_frames/` folder as development progresses.

## Paper

For technical details, architecture diagrams, and rate–distortion graphs, refer to the report below:

📄 [`AutoencoderPaper.pdf`](./AutoencoderPaper.pdf) — IEEE-formatted research paper

## References

- [1] A. Habibian *et al.*, “Video Compression with Rate–Distortion Autoencoders,” *ICCV*, 2019.  
- [2] D. P. Kingma and M. Welling, “Auto-Encoding Variational Bayes,” *ICLR*, 2014.

## Acknowledgment

This project was conducted under the guidance of Dr. Takashi Tanaka and Ronnie Ogden at the University of Texas at Austin. I’d also like to thank Pranad Muttoju for his contributions as well.
