# Virtual Photo Booth (BASNet)

A lightweight virtual photo studio that performs background removal using BASNet and composes portraits with themed frames and backdrops for quick sharing and demos.  
This repo contains an updated version of the earlier VPB prototype and a BASNet-powered pipeline for higher-quality cutouts.

## What this does

- Uses BASNet (Boundary-Aware Segmentation Network) to extract the foreground with crisp boundaries suitable for virtual backgrounds and frames.  
- Provides a simple run script to segment, composite, and preview outputs.  
- Includes sample outputs from a campus event to demonstrate background replacement and framing.

## Repository layout

- `test_test.py` — main entry point that runs the BASNet-powered experience (updated version).  
- `merged.py` — combined utilities for the current pipeline.  
- `test.py` — legacy prototype retained for reference.  
- `data_loader.py` — helper to prepare assets before first run.  
- `saved_model/basnet/basnet.pth` — pre-trained BASNet weights (place here after download).  
- `Images/` — example outputs shown below on this page.

## Setup

1) Create and activate a Python environment (version of choice).  
2) Install project dependencies as used by the scripts.  
3) Download the BASNet weight file `basnet.pth` and place it at:

