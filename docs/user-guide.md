# User Guide

This comprehensive guide covers all features and capabilities of Visualizr, including
the web interface, command-line tools, and advanced usage patterns.

## Table of Contents

- [Overview](#overview)
- [Web Interface](#web-interface)
- [Command Line Interface](#command-line-interface)
- [Parameter Guide](#parameter-guide)
- [Inference Types](#inference-types)
- [File Formats](#file-formats)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)
- [Performance Optimization](#performance-optimization)

## Overview

Visualizr provides multiple ways to generate talking avatar videos:

1. **Web Interface**: Interactive GUI for easy use
2. **Command Line**: Programmatic control and automation
3. **Python API**: Direct integration in applications
4. **REST API**: HTTP-based integration

## Web Interface

The web interface is the most user-friendly way to generate videos. It provides three
main tabs for different workflows.

### Starting the Web Interface

```bash
# Start Visualizr
visualizr

# Custom port and host
VISUALIZR_PORT=8080 VISUALIZR_HOST=0.0.0.0 visualizr

# Open browser to http://localhost:7860
```

### Tab 1: Generate from Paths

Upload image and audio files directly for one-off generations.

#### Input Fields

**Reference Image**

- **Purpose**: Portrait image to animate
- **Formats**: JPG, JPEG, PNG
- **Recommended**: 512x512px or higher, front-facing portraits
- **Click**: Upload button to select file

**Input Audio**

- **Purpose**: Speech audio for lip-sync
- **Formats**: WAV, MP3, M4A
- **Recommended**: Clear speech, 16kHz sample rate
- **Features**: Download button available after upload

#### Process

1. **Upload Files**: Click upload buttons for image and audio
2. **Configure Settings**: Use Configuration tab if needed
3. **Generate**: Click "Generate" button
4. **Monitor Progress**: Watch for completion messages
5. **Download**: Save generated videos

### Tab 2: Generate from Name

Select from pre-loaded character images for repeated use.

#### Character Management

**Adding Characters**:

```bash
# Place images in assets/image/ directory
cp portrait1.jpg assets/image/character1.jpg
cp portrait2.png assets/image/character2.png

# Restart Visualizr to refresh character list
```

**Supported Extensions**: `.jpg`, `.jpeg`, `.png`

#### Usage Process

1. **Select Character**: Choose from dropdown list
2. **Upload Audio**: Provide speech audio file
3. **Configure**: Adjust settings in Configuration tab
4. **Generate**: Click "Generate" button
5. **Review Output**: Check generated videos

### Tab 3: Configuration

Control all generation parameters for fine-tuning output.

#### Core Settings

**Inference Type**

- **Options**: See [Inference Types](#inference-types) section
- **Default**: `hubert_audio_only`
- **Impact**: Affects quality, speed, and control options

**Seed**

- **Range**: Any integer
- **Default**: 0
- **Purpose**: Reproducible results
- **Usage**: Keep same seed for consistent character across videos

**Face Super-Resolution**

- **Options**: Enabled/Disabled
- **Default**: Disabled
- **Impact**: Generates 512x512 enhanced video
- **Note**: Requires additional processing time and VRAM

#### Pose Controls

**Pose Yaw** (Left/Right Rotation)

- **Range**: -1.0 to 1.0
- **Default**: 0.0
- **Effect**: -1 = left turn, +1 = right turn
- **Usage**: Simulate looking left or right

**Pose Pitch** (Up/Down Tilt)

- **Range**: -1.0 to 1.0
- **Default**: 0.0
- **Effect**: -1 = look down, +1 = look up
- **Usage**: Vary vertical gaze direction

**Pose Roll** (Side Tilt)

- **Range**: -1.0 to 1.0
- **Default**: 0.0
- **Effect**: -1 = tilt left, +1 = tilt right
- **Usage**: Add natural head movements

#### Face Positioning

**Face Location**

- **Range**: 0.0 to 1.0
- **Default**: 0.5
- **Effect**: Vertical position in frame
- **Usage**: 0 = bottom, 0.5 = center, 1 = top

**Face Scale**

- **Range**: 0.0 to 1.0
- **Default**: 0.5
- **Effect**: Size of face in frame
- **Usage**: Larger values = bigger face

#### Quality Settings

**Step T** (Generation Steps)

- **Range**: 1 to 200+
- **Default**: 50
- **Impact**: More steps = higher quality, longer processing
- **Recommendations**:
    - Fast preview: 25 steps
    - Standard quality: 50 steps
    - High quality: 100+ steps

## Command Line Interface

For automation, scripting, and programmatic use.

### Basic Usage

```bash
# Direct execution
visualizr

# Python module
python -m visualizr

# Custom settings via environment variables
VISUALIZR_PORT=8080 python -m visualizr
```

### Environment Variables

Configure Visualizr behavior:

```bash
# Server Settings
export VISUALIZR_HOST=0.0.0.0          # Bind address
export VISUALIZR_PORT=7860             # Port number

# Model Settings  
export VISUALIZR_DEVICE=cuda           # 'cuda' or 'cpu'
export VISUALIZR_INFER_TYPE=hubert_audio_only
export VISUALIZR_FACE_SR=false         # Enable super-resolution

# Directory Settings
export VISUALIZR_RESULTS_DIR=./results
export VISUALIZR_CKPTS_DIR=./ckpts
export VISUALIZR_ASSETS_DIR=./assets

# Quality Settings
export VISUALIZR_STEP_T=50             # Generation steps
export VISUALIZR_SEED=42               # Random seed
```

### Python API Usage

Direct integration in Python applications:

```python
from visualizr.app.runner import app
from pathlib import Path

# Basic generation
result = app.generate_video(
    infer_type='hubert_audio_only',
    image_path='assets/image/portrait.jpg',
    audio_path='assets/audio/speech.wav',
    face_sr=False,
    pose_yaw=0.0,
    pose_pitch=0.0,
    pose_roll=0.0,
    face_location=0.5,
    face_scale=0.5,
    step_t=50,
    seed=0
)

# Extract results
video_256, video_512, message = result
print(f"Generated video: {video_256.value}")
```

### Batch Processing Example

```python
import os
from pathlib import Path
from visualizr.app.runner import app

# Process multiple files
image_dir = Path("assets/image")
audio_dir = Path("assets/audio")

for image_file in image_dir.glob("*.jpg"):
    for audio_file in audio_dir.glob("*.wav"):
        print(f"Processing {image_file.name} + {audio_file.name}")
        
        result = app.generate_video(
            infer_type='hubert_audio_only',
            image_path=str(image_file),
            audio_path=str(audio_file),
            face_sr=True,  # High quality
            step_t=100,    # High quality
            seed=42,       # Consistent
            # ... other parameters
        )
        
        if result[0]:  # Check if generation succeeded
            print(f"Success: {result[2].value}")
        else:
            print(f"Failed: {result[2].value}")
```

## Parameter Guide

### Complete Parameter Reference

| Parameter       | Type     | Range/Options                 | Default             | Description             |
|-----------------|----------|-------------------------------|---------------------|-------------------------|
| `infer_type`    | str      | See [types](#inference-types) | `hubert_audio_only` | Audio processing model  |
| `image_path`    | str/Path | Valid file path               | Required            | Input portrait image    |
| `audio_path`    | str/Path | Valid file path               | Required            | Input speech audio      |
| `face_sr`       | bool     | true/false                    | false               | Enable super-resolution |
| `pose_yaw`      | float    | -1.0 to 1.0                   | 0.0                 | Left/right rotation     |
| `pose_pitch`    | float    | -1.0 to 1.0                   | 0.0                 | Up/down tilt            |
| `pose_roll`     | float    | -1.0 to 1.0                   | 0.0                 | Side tilt               |
| `face_location` | float    | 0.0 to 1.0                    | 0.5                 | Vertical position       |
| `face_scale`    | float    | 0.0 to 1.0                    | 0.5                 | Face size               |
| `step_t`        | int      | 1+                            | 50                  | Generation steps        |
| `seed`          | int      | Any integer                   | 0                   | Random seed             |

### Parameter Combinations

#### Preset Configurations

**Quick Preview**:

```python
{
    'infer_type': 'mfcc_full_control',
    'face_sr': False,
    'step_t': 25,
    # ... other params
}
```

**Balanced Quality**:

```python
{
    'infer_type': 'hubert_audio_only',
    'face_sr': False,
    'step_t': 50,
    # ... other params
}
```

**Maximum Quality**:

```python
{
    'infer_type': 'hubert_full_control',
    'face_sr': True,
    'step_t': 100,
    # ... other params
}
```

**Consistent Character Series**:

```python
{
    'seed': 42,           # Same seed
    'face_location': 0.5, # Same positioning
    'face_scale': 0.5,    # Same scale
    # ... vary audio_path only
}
```

## Inference Types

Choose the right model for your specific needs:

### MFCC-Based Models

Use Mel-frequency cepstral coefficients for audio processing.

#### `mfcc_full_control`

- **Audio Processing**: MFCC features with delta coefficients
- **Control**: Full pose and face parameter control
- **Quality**: Good with precise parameter tuning
- **Speed**: Fastest processing (4x audio speed)
- **Best For**: When you need maximum control over output
- **Frame Rate**: 25fps video, 100Hz audio processing

#### `mfcc_pose_only`

- **Audio Processing**: MFCC features
- **Control**: Pose parameters only
- **Quality**: Basic lip-sync with pose control
- **Speed**: Fast processing
- **Best For**: Simple animations with head movements
- **Limitations**: Less natural facial expressions

### HuBERT-Based Models

Use HuBERT (Hidden-Unit BERT) for more natural audio processing.

#### `hubert_audio_only` ⭐ **Recommended**

- **Audio Processing**: HuBERT hidden states
- **Control**: Audio-driven animation only
- **Quality**: High lip-sync accuracy
- **Speed**: Fast to medium processing
- **Best For**: Most general use cases
- **Frame Rate**: 25fps video, 50Hz audio processing
- **Natural**: Best balance of quality and speed

#### `hubert_pose_only`

- **Audio Processing**: HuBERT features
- **Control**: Pose parameters with HuBERT
- **Quality**: Good pose control with natural audio
- **Speed**: Medium processing
- **Best For**: When pose control is needed with quality audio

#### `hubert_full_control`

- **Audio Processing**: Full HuBERT feature extraction
- **Control**: Complete pose and face parameter control
- **Quality**: Highest quality output
- **Speed**: Slowest processing
- **Best For**: Maximum quality applications
- **Requirements**: More VRAM and processing time

### Selection Guide

**Choose based on priority:**

🏃 **Speed Priority**: `mfcc_full_control` → `mfcc_pose_only`

🎯 **Quality Priority**: `hubert_full_control` → `hubert_audio_only`

🎚️ **Control Priority**: `mfcc_full_control` → `hubert_full_control`

⚖️ **Balanced**: `hubert_audio_only` (recommended for most users)

## File Formats

### Supported Input Formats

#### Images

- **Formats**: JPEG (.jpg, .jpeg), PNG (.png)
- **Resolution**: 256x256 minimum, 512x512+ recommended
- **Color**: RGB color images
- **Aspect**: Portrait orientation preferred
- **Quality**: High-quality, well-lit portraits work best

#### Audio

- **Formats**: WAV (.wav), MP3 (.mp3), M4A (.m4a)
- **Sample Rate**: 16kHz recommended, 22kHz/44kHz supported
- **Channels**: Mono preferred, stereo supported
- **Duration**: Any length (longer audio = longer video)
- **Quality**: Clear speech with minimal background noise

### Output Formats

#### Generated Videos

- **Format**: MP4 (H.264 codec)
- **Resolution**: 256x256 (standard) or 512x512 (with super-resolution)
- **Frame Rate**: 25 FPS
- **Audio**: Synchronized with input audio
- **Location**: `results/` directory by default

#### Naming Convention

```
results/
├── [image_name]-[audio_name].mp4        # 256x256 version
└── [image_name]-[audio_name]_SR.mp4     # 512x512 version (if enabled)
```

## Advanced Usage

### Custom Directory Structure

```bash
# Set up custom directories
export VISUALIZR_ASSETS_DIR=/path/to/assets
export VISUALIZR_RESULTS_DIR=/path/to/results
export VISUALIZR_CKPTS_DIR=/path/to/models

# Create structure
mkdir -p $VISUALIZR_ASSETS_DIR/{image,audio,video}
mkdir -p $VISUALIZR_RESULTS_DIR
mkdir -p $VISUALIZR_CKPTS_DIR
```

### Model Management

```python
from huggingface_hub import snapshot_download

# Download models to custom location
snapshot_download(
    repo_id="taocode/anitalker_ckpts",
    local_dir="./custom_models",
    repo_type="model"
)

# Use custom model directory
os.environ['VISUALIZR_CKPTS_DIR'] = './custom_models'
```

### Multi-Processing Setup

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def process_video(params):
    image_path, audio_path, output_name = params
    # ... generate video
    return output_name

# Process multiple videos in parallel
if __name__ == '__main__':
    params_list = [
        ('img1.jpg', 'audio1.wav', 'output1'),
        ('img2.jpg', 'audio2.wav', 'output2'),
        # ... more combinations
    ]
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = executor.map(process_video, params_list)
```

### Integration with Other Tools

#### FFmpeg Integration

```bash
# Post-process generated videos
ffmpeg -i results/output.mp4 -vf "scale=1080:1080" output_hd.mp4

# Combine multiple videos
ffmpeg -f concat -i filelist.txt -c copy combined.mp4

# Extract audio for analysis
ffmpeg -i results/output.mp4 -vn -acodec copy audio.wav
```

#### Batch Conversion

```python
import subprocess
from pathlib import Path

def enhance_video(input_path, output_path):
    """Enhance video quality using FFmpeg"""
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-vf', 'unsharp=5:5:1.0:5:5:0.0',
        '-c:a', 'copy',
        str(output_path)
    ]
    subprocess.run(cmd, check=True)

# Process all generated videos
results_dir = Path('results')
for video_file in results_dir.glob('*.mp4'):
    enhanced_file = results_dir / f"{video_file.stem}_enhanced.mp4"
    enhance_video(video_file, enhanced_file)
```

## Best Practices

### Image Guidelines

#### ✅ Optimal Images

**Portrait Characteristics**:

- Front-facing or slight 3/4 angle (up to 30 degrees)
- Clear facial features and expressions
- Good lighting without harsh shadows
- Minimal occlusion (no sunglasses, hats covering face)
- Single person in frame

**Technical Specifications**:

- Resolution: 512x512 pixels or higher
- Format: High-quality JPEG or PNG
- Aspect ratio: Square or portrait orientation
- File size: Under 10MB for optimal processing

**Lighting and Quality**:

- Even, soft lighting
- Avoid strong directional shadows
- Clear focus, minimal blur
- Natural colors, minimal filters

#### ❌ Problematic Images

**Avoid These**:

- Profile shots (side view)
- Multiple people in frame
- Heavily shadowed or backlit images
- Blurry or low-resolution photos
- Extreme facial expressions or poses
- Heavy makeup or face paint that obscures features

### Audio Guidelines

#### ✅ Optimal Audio

**Recording Quality**:

- Clear, noise-free speech
- Consistent volume levels
- Single speaker only
- Natural speech pace

**Technical Specifications**:

- Sample rate: 16kHz (preferred) or 22kHz/44kHz
- Format: WAV (uncompressed) preferred
- Duration: Any length (video matches audio length)
- Bit depth: 16-bit or higher

**Content Guidelines**:

- Natural speech patterns
- Minimal background noise
- Clear pronunciation
- Moderate speaking pace

#### ❌ Problematic Audio

**Avoid These**:

- Multiple speakers talking simultaneously
- Heavy background music or noise
- Very quiet or very loud recordings
- Heavily processed or auto-tuned vocals
- Extremely fast or slow speech
- Music-only content (no speech)

### Performance Optimization

#### GPU Usage

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Monitor GPU during generation
nvidia-smi -l 1  # Update every second
```

#### Memory Management

```python
# For systems with limited VRAM
import torch

# Clear GPU cache between generations
torch.cuda.empty_cache()

# Use CPU fallback if needed
os.environ['VISUALIZR_DEVICE'] = 'cpu'
```

#### Batch Size Optimization

```python
# Adjust batch size based on available memory
# Default is 1 for safety
os.environ['VISUALIZR_BATCH_SIZE'] = '1'  # Conservative
os.environ['VISUALIZR_BATCH_SIZE'] = '2'  # More memory needed
```

### Quality vs Speed Trade-offs

#### Speed-Optimized Settings

```python
speed_config = {
    'infer_type': 'mfcc_full_control',    # Fastest model
    'face_sr': False,                     # Skip super-resolution
    'step_t': 25,                         # Fewer diffusion steps
    # Pose parameters as needed
}
```

#### Quality-Optimized Settings

```python
quality_config = {
    'infer_type': 'hubert_full_control',  # Best quality model
    'face_sr': True,                      # Enable super-resolution
    'step_t': 100,                        # More diffusion steps
    # Fine-tune pose parameters
}
```

#### Balanced Settings

```python
balanced_config = {
    'infer_type': 'hubert_audio_only',    # Good balance
    'face_sr': False,                     # Standard resolution
    'step_t': 50,                         # Standard steps
    # Default pose parameters
}
```

## Performance Optimization

### System Requirements by Use Case

#### Development/Testing

- **GPU**: GTX 1060 6GB or equivalent
- **RAM**: 8GB system memory
- **Storage**: 5GB for models
- **Settings**: Standard quality, no super-resolution

#### Content Creation

- **GPU**: RTX 2070 8GB or equivalent
- **RAM**: 16GB system memory
- **Storage**: 10GB+ for models and cache
- **Settings**: High quality with super-resolution

#### Production/Batch Processing

- **GPU**: RTX 3080 10GB or higher
- **RAM**: 32GB system memory
- **Storage**: SSD with 20GB+ free space
- **Settings**: Optimized for throughput

### Monitoring and Debugging

#### Resource Monitoring

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor system resources
htop

# Check disk space
df -h

# Monitor Python memory usage
python -m memory_profiler script.py
```

#### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model loading
from visualizr.app.runner import app
print(f"Models loaded in: {app.settings.directory.checkpoint}")
```

### Scaling Considerations

#### Single Instance Optimization

- Use GPU acceleration when available
- Optimize batch size for available VRAM
- Cache models between generations
- Use SSD storage for faster I/O

#### Multi-Instance Deployment

- One instance per GPU for parallel processing
- Load balance requests across instances
- Use shared model storage to reduce memory
- Implement request queuing for high load

---

*Ready for production deployment? Check out the [Deployment Guide](deployment.md)!*
