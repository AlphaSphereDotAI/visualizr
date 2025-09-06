# Troubleshooting Guide

Comprehensive troubleshooting guide for common issues and solutions when using
Visualizr.

## Table of Contents

- [Quick Diagnostics](#quick-diagnostics)
- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Problems](#performance-problems)
- [Model Issues](#model-issues)
- [Hardware Problems](#hardware-problems)
- [Network & API Issues](#network--api-issues)
- [File & Storage Issues](#file--storage-issues)
- [Advanced Debugging](#advanced-debugging)

## Quick Diagnostics

### System Check Script

Run this script to quickly diagnose common issues:

```bash
#!/bin/bash
# visualizr-diagnostic.sh

echo "=== Visualizr Diagnostic Tool ==="
echo

# Python version
echo "Python Version:"
python --version
python3 --version 2>/dev/null || echo "python3 not found"
python3.10 --version 2>/dev/null || echo "python3.10 not found"
echo

# GPU Check
echo "GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
else
    echo "NVIDIA drivers not found"
fi
echo

# CUDA Check
echo "CUDA Status:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "CUDA not found"
fi
echo

# PyTorch GPU Check
echo "PyTorch GPU Status:"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not available"
echo

# Visualizr Installation
echo "Visualizr Status:"
python -c "import visualizr; print('Visualizr installed successfully')" 2>/dev/null || echo "Visualizr not installed"
echo

# Memory Check
echo "System Memory:"
free -h 2>/dev/null || echo "Memory info not available"
echo

# Disk Space
echo "Disk Space:"
df -h . 2>/dev/null || echo "Disk info not available"
echo

# Network Test
echo "Network Test:"
curl -s --max-time 5 https://huggingface.co > /dev/null && echo "HuggingFace Hub accessible" || echo "HuggingFace Hub connection failed"
echo

echo "=== Diagnostic Complete ==="
```

### Quick Health Check

```python
#!/usr/bin/env python3
"""Quick health check for Visualizr"""

import sys
import torch
import subprocess
from pathlib import Path

def health_check():
    """Perform comprehensive health check"""
    
    issues = []
    
    # Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python {sys.version} < required 3.10")
    
    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        issues.append("PyTorch not installed")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ CUDA not available (CPU mode)")
    
    # Visualizr
    try:
        import visualizr
        print("✓ Visualizr installed")
    except ImportError:
        issues.append("Visualizr not installed")
    
    # Dependencies
    required_packages = [
        'gradio', 'transformers', 'librosa', 
        'moviepy', 'numpy', 'huggingface_hub'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            issues.append(f"Missing package: {package}")
    
    # Directories
    dirs_to_check = ['ckpts', 'assets', 'results']
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ Directory {dir_name} exists")
        else:
            print(f"⚠ Directory {dir_name} missing (will be created)")
    
    # Report issues
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✅ All checks passed!")
        return True

if __name__ == "__main__":
    health_check()
```

## Installation Issues

### Python Version Problems

**Problem**: Wrong Python version or Python not found

```bash
# Error examples
python: command not found
Python 3.9.x but 3.10.x required
```

**Solutions**:

```bash
# Install Python 3.10 on Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Install Python 3.10 on macOS with Homebrew
brew install python@3.10

# Install Python 3.10 on Windows
# Download from python.org or use Windows Store

# Verify installation
python3.10 --version
```

### uv Installation Issues

**Problem**: uv not found or installation fails

```bash
# Error examples
uv: command not found
curl: command not found
```

**Solutions**:

```bash
# Install uv on Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Install uv on Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv

# Verify installation
uv --version
```

### Package Dependency Conflicts

**Problem**: Conflicting package versions

```
ERROR: pip's dependency resolver does not currently have a way to solve this
```

**Solutions**:

```bash
# Clear pip cache
pip cache purge

# Use uv for better dependency resolution
uv pip install visualizr

# Create fresh virtual environment
python3.10 -m venv fresh_env
source fresh_env/bin/activate  # Windows: fresh_env\Scripts\activate
pip install visualizr

# Force reinstall
pip install --force-reinstall --no-deps visualizr
```

### Permission Issues

**Problem**: Permission denied during installation

```bash
# Error examples
Permission denied: '/usr/local/lib/python3.10'
[Errno 13] Permission denied
```

**Solutions**:

```bash
# Use virtual environment (recommended)
python3.10 -m venv venv
source venv/bin/activate
pip install visualizr

# User installation
pip install --user visualizr

# Fix permissions (Linux/macOS)
sudo chown -R $USER:$USER ~/.local/
sudo chown -R $USER:$USER ~/.cache/
```

## Runtime Errors

### CUDA Out of Memory

**Problem**: GPU memory exhausted

```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions**:

```python
# Solution 1: Reduce settings
export VISUALIZR_STEP_T=25          # Fewer steps
export VISUALIZR_FACE_SR=false      # Disable super-resolution
export VISUALIZR_BATCH_SIZE=1       # Minimum batch size

# Solution 2: Clear GPU cache
import torch
torch.cuda.empty_cache()

# Solution 3: Use CPU fallback
export VISUALIZR_DEVICE=cpu

# Solution 4: Enable memory management
export VISUALIZR_MEMORY_FRACTION=0.7  # Use 70% of GPU memory
```

```python
# Python code to clear memory
def clear_gpu_memory():
    """Clear GPU memory and cache"""
    import torch
    import gc
    
    # Clear Python garbage
    gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Model Loading Failures

**Problem**: Models fail to download or load

```
OSError: Unable to load weights from checkpoint file
FileNotFoundError: Model file not found
```

**Solutions**:

```bash
# Solution 1: Manual model download
python -c "
from huggingface_hub import snapshot_download
snapshot_download('taocode/anitalker_ckpts', local_dir='./ckpts')
"

# Solution 2: Check network connectivity
curl -I https://huggingface.co/taocode/anitalker_ckpts

# Solution 3: Clear model cache
rm -rf ~/.cache/huggingface/
rm -rf ./ckpts/

# Solution 4: Set custom model path
export VISUALIZR_CKPTS_DIR=/path/to/models

# Solution 5: Check disk space
df -h
```

### Import Errors

**Problem**: Python modules not found

```python
ModuleNotFoundError: No module named 'visualizr'
ImportError: No module named 'torch'
```

**Solutions**:

```bash
# Verify installation
pip list | grep visualizr
pip show visualizr

# Reinstall
pip uninstall visualizr
pip install visualizr

# Check Python path
python -c "import sys; print(sys.path)"

# Virtual environment check
which python
pip --version
```

### Port Already in Use

**Problem**: Default port 7860 is occupied

```
OSError: [Errno 98] Address already in use
```

**Solutions**:

```bash
# Solution 1: Use different port
export VISUALIZR_PORT=8080
visualizr

# Solution 2: Find and kill process using port
lsof -ti:7860 | xargs kill -9

# Solution 3: Windows equivalent
netstat -ano | findstr :7860
taskkill /PID <PID> /F

# Solution 4: Docker conflict check
docker ps | grep 7860
```

## Performance Problems

### Slow Generation Speed

**Problem**: Video generation takes too long

**Diagnostic**:

```python
# Check current settings
python -c "
from visualizr.settings import Settings
settings = Settings()
print(f'Device: {settings.model.device}')
print(f'Inference: {settings.model.infer_type}')  
print(f'Steps: {settings.model.step_t}')
print(f'Super-res: {settings.model.face_sr}')
"
```

**Solutions**:

```bash
# Speed optimizations
export VISUALIZR_INFER_TYPE=mfcc_full_control  # Fastest
export VISUALIZR_STEP_T=25                     # Fewer steps
export VISUALIZR_FACE_SR=false                 # No super-resolution
export VISUALIZR_DEVICE=cuda                   # Use GPU

# Monitor GPU usage
nvidia-smi -l 1

# Check system resources
htop
```

### Memory Leaks

**Problem**: Memory usage increases over time

**Diagnostic**:

```python
# Memory monitoring script
import psutil
import torch
import time

def monitor_memory():
    """Monitor system and GPU memory"""
    while True:
        # System memory
        mem = psutil.virtual_memory()
        print(f"System RAM: {mem.percent}% ({mem.used/1e9:.1f}GB/{mem.total/1e9:.1f}GB)")
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_mem:.1f}GB/{gpu_total:.1f}GB")
        
        time.sleep(5)

monitor_memory()
```

**Solutions**:

```python
# Implement memory cleanup
def cleanup_after_generation():
    """Clean up after each generation"""
    import gc
    import torch
    
    # Clear variables
    del result, video_256, video_512
    
    # Python garbage collection
    gc.collect()
    
    # GPU cache cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

### High CPU Usage

**Problem**: Excessive CPU usage during generation

**Solutions**:

```bash
# Check process usage
top -p $(pgrep -f visualizr)

# Limit CPU cores
taskset -c 0-3 python -m visualizr  # Use only cores 0-3

# Adjust worker processes
export VISUALIZR_NUM_WORKERS=2

# Docker CPU limits
docker run --cpus="4.0" visualizr:latest
```

## Model Issues

### Poor Quality Results

**Problem**: Generated videos have poor quality or artifacts

**Diagnostic Checklist**:

- Input image quality and resolution
- Audio quality and format
- Model settings and parameters
- Hardware capabilities

**Solutions**:

```bash
# Quality improvements
export VISUALIZR_INFER_TYPE=hubert_full_control  # Best quality
export VISUALIZR_STEP_T=100                      # More steps
export VISUALIZR_FACE_SR=true                    # Super-resolution
export VISUALIZR_SEED=42                         # Consistent results

# Input validation
python -c "
from PIL import Image
img = Image.open('input.jpg')
print(f'Image size: {img.size}')
print(f'Image mode: {img.mode}')
print(f'Image format: {img.format}')
"
```

### Lip-sync Issues

**Problem**: Audio and lip movements are not synchronized

**Solutions**:

```bash
# Try different audio processing
export VISUALIZR_INFER_TYPE=hubert_audio_only  # Better for natural speech
export VISUALIZR_INFER_TYPE=mfcc_full_control  # Better for precise control

# Check audio format
ffprobe input.wav

# Convert audio format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### Model Download Failures

**Problem**: Models fail to download from HuggingFace

**Solutions**:

```bash
# Check HuggingFace connectivity
curl -I https://huggingface.co

# Use different mirror/proxy
export HF_ENDPOINT=https://hf-mirror.com

# Manual download
git lfs install
git clone https://huggingface.co/taocode/anitalker_ckpts ./ckpts

# Verify downloads
ls -la ./ckpts/
du -sh ./ckpts/
```

## Hardware Problems

### NVIDIA Driver Issues

**Problem**: CUDA not available despite having NVIDIA GPU

**Diagnostic**:

```bash
# Check driver status
nvidia-smi
lsmod | grep nvidia

# Check CUDA installation
nvcc --version
cat /usr/local/cuda/version.txt
```

**Solutions**:

```bash
# Ubuntu/Debian: Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-470  # or latest version
sudo reboot

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Verify installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### GPU Memory Issues

**Problem**: GPU has insufficient memory

**Solutions**:

```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

# Free GPU memory
sudo nvidia-smi --gpu-reset

# Kill GPU processes
sudo fuser -v /dev/nvidia*
sudo kill -9 <PID>

# Use memory-efficient settings
export VISUALIZR_MEMORY_FRACTION=0.5
export VISUALIZR_BATCH_SIZE=1
```

### CPU/Memory Limitations

**Problem**: Insufficient system resources

**Solutions**:

```bash
# Check system resources
free -h
nproc
lscpu

# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Close other applications
pkill -f chrome
pkill -f firefox
```

## Network & API Issues

### Connection Timeouts

**Problem**: Requests timeout during generation

**Solutions**:

```bash
# Increase timeouts
export VISUALIZR_TIMEOUT=600  # 10 minutes

# nginx configuration
location /api/generate_video {
    proxy_read_timeout 600s;
    proxy_connect_timeout 60s;
}

# Client timeout handling
curl --max-time 600 http://localhost:7860/api/generate_video
```

### API Rate Limiting

**Problem**: Too many requests error

**Solutions**:

```bash
# Check rate limits
curl -I http://localhost:7860/api/generate_video

# Implement client-side rate limiting
python -c "
import time
import requests

def rate_limited_request():
    for i in range(5):
        try:
            response = requests.post('http://localhost:7860/api/generate_video')
            return response
        except requests.exceptions.RequestException:
            time.sleep(2 ** i)  # Exponential backoff
    raise Exception('Max retries exceeded')
"
```

### Firewall Issues

**Problem**: Cannot access Visualizr from other machines

**Solutions**:

```bash
# Check if port is open
netstat -tlnp | grep 7860
ss -tlnp | grep 7860

# Open firewall port (Ubuntu)
sudo ufw allow 7860/tcp

# Open firewall port (CentOS/RHEL)
sudo firewall-cmd --permanent --add-port=7860/tcp
sudo firewall-cmd --reload

# Test connectivity
telnet server_ip 7860
curl http://server_ip:7860/heartbeat
```

## File & Storage Issues

### Disk Space Problems

**Problem**: Insufficient disk space for models or results

**Solutions**:

```bash
# Check disk usage
df -h
du -sh ./ckpts/ ./results/ ./assets/

# Clean up old results
find ./results -name "*.mp4" -mtime +7 -delete

# Move models to larger disk
mkdir /mnt/storage/visualizr_models
ln -s /mnt/storage/visualizr_models ./ckpts

# Configure custom paths
export VISUALIZR_RESULTS_DIR=/mnt/storage/results
export VISUALIZR_CKPTS_DIR=/mnt/storage/models
```

### File Permission Issues

**Problem**: Cannot read/write files

**Solutions**:

```bash
# Fix permissions
sudo chown -R $USER:$USER ./ckpts/ ./results/ ./assets/
chmod -R 755 ./ckpts/ ./results/ ./assets/

# Docker permission issues
docker run --user $(id -u):$(id -g) visualizr:latest

# SELinux issues (CentOS/RHEL)
sudo setsebool -P container_manage_cgroup on
sudo semanage fcontext -a -t container_file_t "/path/to/visualizr(/.*)?"
sudo restorecon -R /path/to/visualizr
```

### Corrupted Files

**Problem**: Model files or results are corrupted

**Solutions**:

```bash
# Verify file integrity
md5sum ./ckpts/stage1.ckpt
sha256sum ./results/output.mp4

# Re-download models
rm -rf ./ckpts/
python -c "from huggingface_hub import snapshot_download; snapshot_download('taocode/anitalker_ckpts', local_dir='./ckpts')"

# Repair video files
ffmpeg -i corrupted.mp4 -c copy repaired.mp4
```

## Advanced Debugging

### Logging Configuration

Enable detailed logging for debugging:

```python
# debug_logging.py
import logging
import sys

def setup_debug_logging():
    """Enable comprehensive debug logging"""
    
    # Root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler('visualizr_debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific loggers
    loggers = [
        'visualizr',
        'torch',
        'transformers',
        'gradio',
        'huggingface_hub'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    # Disable noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

# Usage
setup_debug_logging()
```

### Performance Profiling

Profile code to find bottlenecks:

```python
# profile_visualizr.py
import cProfile
import pstats
from visualizr.app.runner import app

def profile_generation():
    """Profile video generation performance"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run generation
    result = app.generate_video(
        infer_type='hubert_audio_only',
        image_path='test_image.jpg',
        audio_path='test_audio.wav',
        face_sr=False,
        # ... other parameters
    )
    
    profiler.disable()
    
    # Save and display results
    profiler.dump_stats('generation_profile.prof')
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('tottime')
    stats.print_stats(20)  # Top 20 functions

# Usage
profile_generation()
```

### Memory Profiling

Track memory usage:

```python
# memory_profile.py
from memory_profiler import profile
import torch

@profile
def memory_intensive_generation():
    """Profile memory usage during generation"""
    
    # Your generation code here
    from visualizr.app.runner import app
    
    result = app.generate_video(
        # ... parameters
    )
    
    return result

# Run with: python -m memory_profiler memory_profile.py
```

### Network Debugging

Debug API and network issues:

```python
# network_debug.py
import requests
import time
import json

def debug_api_request():
    """Debug API requests with detailed logging"""
    
    url = "http://localhost:7860/api/generate_video"
    
    # Request with debugging
    session = requests.Session()
    
    # Enable request/response logging
    import logging
    import http.client as http_client
    
    http_client.HTTPConnection.debuglevel = 1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True
    
    # Make request
    try:
        response = session.post(url, files={
            'image_path': ('test.jpg', open('test.jpg', 'rb')),
            'audio_path': ('test.wav', open('test.wav', 'rb'))
        })
        
        print(f"Status: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Content: {response.text[:1000]}")
        
    except Exception as e:
        print(f"Request failed: {e}")

debug_api_request()
```

### Container Debugging

Debug Docker containers:

```bash
# Container inspection
docker inspect visualizr_container

# Container logs
docker logs -f visualizr_container
docker logs --tail 100 visualizr_container

# Execute commands in container
docker exec -it visualizr_container bash
docker exec -it visualizr_container python -c "import torch; print(torch.cuda.is_available())"

# Debug container networking
docker exec visualizr_container netstat -tlnp
docker exec visualizr_container curl http://localhost:7860/heartbeat

# Resource usage
docker stats visualizr_container
```

### Environment Debugging

Debug environment and configuration:

```python
# env_debug.py
import os
import sys
from pathlib import Path

def debug_environment():
    """Debug environment configuration"""
    
    print("=== Environment Debug ===")
    
    # Python environment
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path: {sys.path}")
    
    # Working directory
    print(f"CWD: {os.getcwd()}")
    
    # Environment variables
    visualizr_vars = {k: v for k, v in os.environ.items() if 'VISUALIZR' in k}
    print(f"Visualizr env vars: {visualizr_vars}")
    
    # Directory structure
    dirs_to_check = ['ckpts', 'assets', 'results', 'logs']
    for dir_name in dirs_to_check:
        path = Path(dir_name)
        print(f"{dir_name}: exists={path.exists()}, is_dir={path.is_dir()}")
        if path.exists():
            try:
                files = list(path.iterdir())[:5]  # First 5 files
                print(f"  Contents: {[f.name for f in files]}")
            except PermissionError:
                print(f"  Permission denied")
    
    # Package versions
    packages = ['torch', 'gradio', 'transformers', 'visualizr']
    for package in packages:
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: not installed")

debug_environment()
```

## Getting Help

### Community Support

1. **GitHub Issues
   **: [Report bugs and get help](https://github.com/AlphaSphereDotAI/visualizr/issues)
2. **Discussions
   **: [Community discussions](https://github.com/AlphaSphereDotAI/visualizr/discussions)
3. **Documentation
   **: [Official documentation](https://github.com/AlphaSphereDotAI/visualizr/docs)

### Bug Report Template

When reporting issues, include:

```markdown
## Bug Report

**Environment**:
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- Visualizr: [e.g., 0.0.5]
- GPU: [e.g., RTX 3080, 10GB VRAM]

**Command/Code**:
```bash
[Command that caused the issue]
```

**Expected Behavior**:
[What you expected to happen]

**Actual Behavior**:
[What actually happened]

**Error Message**:

```
[Full error message and traceback]
```

**Additional Context**:
[Any other relevant information]

**Diagnostic Output**:
[Output from diagnostic scripts above]

```

### Professional Support

For enterprise users requiring dedicated support:
- Contact: [AlphaSphere.AI](https://alphasphere.ai)
- Email: support@alphasphere.ai
- Consulting: Custom implementation and optimization services

---

*Need more help? Check our [User Guide](user-guide.md) or [Configuration Guide](configuration.md) for detailed information.*
