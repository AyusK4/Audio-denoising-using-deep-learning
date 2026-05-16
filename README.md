# Audio Denoiser (DCUNet)

A web app that removes background noise from speech recordings using a pretrained DCUNet model. The backend is a Flask API that runs the denoising model, and the frontend is a Vite + React UI.

This project reduces the heavy dependence on clean speech data typically required by deep learning denoisers. It shows that deep speech denoising networks can be trained using only noisy speech samples, and that training with noisy targets can outperform conventional clean-target training in complex noise distributions and low $\text{SNR}$ (high-noise environments). The approach is demonstrated on both real-world and synthetic noises using a 20-layer Deep Complex U-Net. The goal is to encourage audio data collection even when recordings are not perfectly clean, which can lower data collection costs and expand speech denoising for low-resource languages.

![UI Screenshot](docs/UI.png)

## Project structure

- backend/ - Flask API, model code, metrics, and runtime folders
- Frontend/ - React UI (Vite)
- notebooks/ - training and exploration notebooks
- docs/ - screenshots and documentation assets
- environment.yml - conda environment for the backend

## Prerequisites

- Conda (Miniconda/Anaconda)
- Node.js 18+ and npm
- ffmpeg installed and available on PATH

## Setup and run (local)

### 1) Backend

```powershell
cd "C:\Users\HP\Desktop\Audio denoiser\Audio-denoising-using-deep-learning"
conda env create -f environment.yml
conda activate audio_denoiser1
python backend\server.py
```

The API runs on http://localhost:5000 and exposes POST /denoise.

### 2) Frontend

```powershell
cd "C:\Users\HP\Desktop\Audio denoiser\Audio-denoising-using-deep-learning\Frontend"
npm install
npm run dev
```

Open the URL shown by Vite (usually http://localhost:5173).

## Model weights

The backend expects pretrained weights at:

```
backend/Pretrained_Weights/Noise2Noise/mixed.pth
```

These weights are intentionally ignored by git. Train the model and then , place them manually before running.

## Datasets

We use two standard datasets:

- UrbanSound8K (real-world noise samples)
- Voice Bank + DEMAND (speech samples)

Please download them from:

- https://urbansounddataset.weebly.com/urbansound8k.html
- https://datashare.ed.ac.uk/handle/10283/2791

## Runtime outputs

The backend writes temporary files to:

```
backend/Samples/
```

This folder is ignored by git /you can create it.

## Notes

- For best results, record or upload WAV/WebM audio with clear speech and minimal clipping.
