# 512D
# Project Setup Guide

This guide explains how to install dependencies on a Linux system with **CUDA 12.6**.  
**Minimum Requirements**:  
- NVIDIA GPU with Compute Capability ≥ 3.5
- NVIDIA Drivers ≥ 535
- Linux (Tested on Ubuntu 22.04)

---

## Step 1: Install System Dependencies

### 1.1 Verify CUDA 12.6 Installation
```bash
nvidia-smi
```
### Verify CUDA version (should show 12.6)
```bash
nvcc --version
```
### If CUDA 12.6 is not installed:

[Follow NVIDIA\'s CUDA 12.6 installation guide](https://developer.nvidia.com/cuda-12-6-0-download-archive)

### 1.2 Install System Libraries
```bash
sudo apt update && sudo apt install -y \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0
```
## Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv myenv
```

```bash
# Activate environment
source myenv/bin/activate
```

## Step 3: Install Python Dependencies
```bash
# Install from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```
### If you see cuda related errors:
```bash
# Reinstall PyTorch with explicit CUDA 12.6 support
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
  --index-url https://download.pytorch.org/whl/cu126
```

## Step 4: Verify Installation
### 4.1 Check GPU Accessibility
```bash 
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```
### 4.2 Resolve Dependency Conflicts
```bash
# Check for package conflicts
pip check

# If conflicts exist, manually resolve them (example):
# pip install "numpy>=1.21,<1.24"  # If a package requires older numpy
```

