# Installation Guide

This guide provides detailed instructions for installing Visualizr on different
platforms and environments.

## Table of Contents

- [Quick Installation](#quick-installation)
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
    - [Using uv (Recommended)](#using-uv-recommended)
    - [Using pip](#using-pip)
    - [Using Docker](#using-docker)
    - [From Source](#from-source)
- [GPU Setup](#gpu-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Quick Installation

For most users, the fastest way to get started is using `uv`:

```bash
uvx --python 3.10 visualizr
```

For Docker users:

```bash
curl -o compose.yaml https://raw.githubusercontent.com/alphaspheredotai/visualizr/main/compose.yaml
docker compose up
```

## System Requirements

### Minimum Requirements

- **Python**: 3.10 (an exact version required)
- **Operating System**:
    - Linux (Ubuntu 18.04+, CentOS 7+)
    - Windows 10/11
    - macOS 10.15+
- **Memory**: 8 GB RAM minimum
- **Storage**: 5 GB free space for models and dependencies

### Recommended Requirements

- **GPU**: NVIDIA GPU with CUDA 11.8 support
- **Memory**: 16 GB RAM or more
- **Storage**: 10 GB+ free space
- **Python**: 3.10 with virtual environment

> **Important**: Visualizr needs python 3.10 only (it does not support other versions).

### GPU Requirements (Optional but Recommended)

- **NVIDIA GPU**: GTX 1060 or better
- **CUDA**: Version 11.8
- **VRAM**: 6 GB+ for optimal performance
- **Compute Capability**: 6.0 or higher

> **Note**: Visualizr can run on CPU-only systems, but GPU acceleration significantly
> improves performance.

## Installation Methods

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a modern Python package manager that provides fast,
reliable dependency resolution.

#### 1. Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

#### 2. Run Visualizr

```bash
# Run directly without installation
uvx --python 3.10 visualizr

# Or install globally
uv tool install --python 3.10 visualizr
visualizr
```

#### 3. Install in a Project

```bash
# Create new project
uv init my-visualizr-project
cd my-visualizr-project

# Add visualizr
uv add visualizr

# Run
uv run visualizr
```

### Using pip

If you prefer using pip, ensure you have Python 3.10 installed:

#### 1. Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### 2. Install Visualizr

```bash
pip install visualizr
```

#### 3. Run Application

```bash
visualizr
```

### Using Docker

Docker provides an isolated environment with all dependencies pre-configured.

#### 1. Using Docker Compose (Recommended)

```bash
# Download compose file
curl -o compose.yaml https://raw.githubusercontent.com/alphaspheredotai/visualizr/main/compose.yaml

# Start the application
docker compose up

# Run in background
docker compose up -d
```

#### 2. Using Docker Run

```bash
# Pull and run the image
docker run -p 7860:7860 visualizr:latest

# With GPU support (requires nvidia-docker)
docker run --gpus all -p 7860:7860 visualizr:latest
```

#### 3. Build from Source

```bash
# Clone repository
git clone https://github.com/AlphaSphereDotAI/visualizr.git
cd visualizr

# Build image
docker build -t visualizr .

# Run
docker run -p 7860:7860 visualizr
```

### From Source

For development or customization, you can install from a source:

#### 1. Clone Repository

```bash
git clone https://github.com/AlphaSphereDotAI/visualizr.git
cd visualizr
```

#### 2. Install Dependencies

Using uv (recommended):

```bash
# Install with development dependencies
uv sync --dev

# Install without development dependencies  
uv sync
```

Using pip:

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install in development mode
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

#### 3. Run Application

```bash
# Using uv
uv run visualizr

# Using pip
python -m visualizr
```

## GPU Setup

### CUDA Installation

For GPU acceleration, install CUDA 11.8:

#### Linux (Ubuntu)

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit 11.8
sudo apt-get -y install cuda-toolkit-11-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
source ~/.bashrc
```

#### Windows

1. Download CUDA 11.8
   from [NVIDIA's website](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Run the installer and follow instructions
3. Verify installation: `nvcc --version`

### Verify GPU Setup

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU device
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Model Download

Visualizr automatically downloads required models on the first run. Models are cached
in:

- **Linux/macOS**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

### Manual Model Download

```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download taocode/anitalker_ckpts

# Using Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('taocode/anitalker_ckpts')"
```

## Verification

After installation, verify everything works correctly:

### 1. Test Installation

```bash
# Check version
visualizr --version

# Test import
python -c "import visualizr; print('Installation successful!')"
```

### 2. Run Test Generation

```bash
# Start web interface
visualizr

# Open browser and navigate to http://localhost:7860
# Upload a test image and audio file
# Generate a test video
```

### 3. API Test

```bash
# Test API endpoints
curl http://localhost:7860/api/docs
```

## Directory Structure

After installation, Visualizr creates the following directories:

```
~/
├── ckpts/              # Model checkpoints
├── assets/             # Input assets
│   ├── image/          # Input images
│   ├── audio/          # Input audio files  
│   └── video/          # Input video files
├── results/            # Output videos
├── logs/               # Application logs
└── .cache/             # Cached models
```

## Environment Variables

Configure Visualizr using environment variables:

```bash
# Model settings
export VISUALIZR_DEVICE=cuda        # or 'cpu'
export VISUALIZR_BATCH_SIZE=1       # Batch size
export VISUALIZR_MODEL_REPO=taocode/anitalker_ckpts

# Directory settings
export VISUALIZR_RESULTS_DIR=./results
export VISUALIZR_CKPTS_DIR=./ckpts
export VISUALIZR_ASSETS_DIR=./assets

# Server settings
export VISUALIZR_HOST=0.0.0.0
export VISUALIZR_PORT=7860
```

## Troubleshooting

### Common Issues

#### Permission Errors

```bash
# On Linux/macOS, ensure proper permissions
sudo chown -R $USER:$USER ~/.cache/huggingface/

# On Windows, run as administrator or check file permissions
```

#### CUDA Errors

```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"
```

#### Memory Issues

```bash
# Reduce batch size
export VISUALIZR_BATCH_SIZE=1

# Use CPU if GPU memory is insufficient
export VISUALIZR_DEVICE=cpu
```

#### Model Download Issues

```bash
# Clear model cache
rm -rf ~/.cache/huggingface/hub/

# Manual download
huggingface-cli download taocode/anitalker_ckpts
```

### Getting Help

If you encounter issues:

1. **Check logs**: Look in the `logs/` directory for error messages
2. **Verify system requirements**: Ensure all requirements are met
3. **Update dependencies**: Try updating to the latest versions
4. **Community support**: Visit
   our [GitHub Issues](https://github.com/AlphaSphereDotAI/visualizr/issues)

## Next Steps

After a successful installation:

1. **[Quick Start Guide](quickstart.md)** - Create your first video
2. **[User Guide](user-guide.md)** - Learn all features
3. **[Configuration](configuration.md)** - Customize settings
4. **[API Reference](api-reference.md)** - Integrate with your applications

---

*Installation complete! Ready to [create your first talking avatar](quickstart.md)?*
