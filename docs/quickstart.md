# Quick Start Guide

Get up and running with Visualizr in minutes! This guide shows you how to create your
first talking avatar video.

## Prerequisites

Make sure you have Visualizr installed. If not, see
the [Installation Guide](installation.md).

```bash
# Quick install with uv
uvx --python 3.10 visualizr

# Or with Docker
curl -o compose.yaml https://raw.githubusercontent.com/alphaspheredotai/visualizr/main/compose.yaml
docker compose up
```

## Your First Video

### Option 1: Web Interface (Recommended for Beginners)

The easiest way to start is with the web interface:

#### 1. Start Visualizr

```bash
visualizr
```

#### 2. Open Your Browser

Navigate to `http://localhost:7860` in your web browser.

#### 3. Upload Files

- **Reference Image**: Upload a portrait image (JPG, PNG)
- **Input Audio**: Upload an audio file (WAV, MP3)

#### 4. Generate Video

- Click the **"Generate"** button
- Wait for processing (usually 30-120 seconds)
- Download your generated video!

### Option 2: Command Line Interface

For programmatic use or automation:

#### 1. Prepare Your Files

```bash
# Create directories
mkdir -p assets/image assets/audio results

# Add your files
cp your-portrait.jpg assets/image/
cp your-audio.wav assets/audio/
```

#### 2. Run Generation

```bash
# Using Python
python -c "
from visualizr.app.runner import app
result = app.generate_video(
    infer_type='hubert_audio_only',
    image_path='assets/image/your-portrait.jpg',
    audio_path='assets/audio/your-audio.wav',
    face_sr=False,
    pose_yaw=0.0,
    pose_pitch=0.0,
    pose_roll=0.0,
    face_location=0.5,
    face_scale=0.5,
    step_t=50,
    seed=0
)
print(f'Video generated: {result[0]}')
"
```

## Example Walkthrough

Let's create a talking avatar with a sample image and audio:

### Step 1: Download Sample Files

```bash
# Create directories
mkdir -p assets/image assets/audio

# Download sample image (Napoleon portrait)
curl -o assets/image/napoleon.jpg \
  https://github.com/AlphaSphereDotAI/chattr/raw/main/assets/image/Napoleon.jpg

# Download sample audio
curl -o assets/audio/sample.wav \
  https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav
```

### Step 2: Start Visualizr

```bash
visualizr
```

### Step 3: Generate Your Video

1. Open `http://localhost:7860`
2. Upload the downloaded files:
    - **Reference Image**: `assets/image/napoleon.jpg`
    - **Input Audio**: `assets/audio/sample.wav`
3. Click **Generate**
4. Your video will be saved in the `results/` directory!

## Understanding the Interface

### Main Tabs

#### 1. **Generate from Paths**

- Upload image and audio files directly
- Full control over input files
- Best for one-off generations

#### 2. **Generate from Name**

- Select from pre-loaded character images
- Faster for repeated use with same characters
- Add images to `assets/image/` directory

#### 3. **Configuration**

- Adjust generation parameters
- Control pose, face position, and quality
- Fine-tune output for specific needs

### Key Parameters

| Parameter                 | Description                 | Range                         | Default             |
|---------------------------|-----------------------------|-------------------------------|---------------------|
| **Inference Type**        | Audio processing model      | See [types](#inference-types) | `hubert_audio_only` |
| **Face Super-Resolution** | Enhance output to 512x512   | On/Off                        | Off                 |
| **Pose Yaw**              | Left/right head rotation    | -1 to 1                       | 0                   |
| **Pose Pitch**            | Up/down head tilt           | -1 to 1                       | 0                   |
| **Pose Roll**             | Side head tilt              | -1 to 1                       | 0                   |
| **Face Location**         | Vertical face position      | 0 to 1                        | 0.5                 |
| **Face Scale**            | Face size in frame          | 0 to 1                        | 0.5                 |
| **Steps**                 | Generation quality vs speed | 1+                            | 50                  |
| **Seed**                  | Reproducible results        | Any integer                   | 0                   |

## Inference Types

Choose the right inference type for your needs:

### **hubert_audio_only** (Recommended)

- **Best for**: Most use cases, natural speech
- **Quality**: High lip-sync accuracy
- **Speed**: Fast processing
- **Use when**: You want the best general results

### **mfcc_full_control**

- **Best for**: Fine control over animation
- **Quality**: Good with full parameter control
- **Speed**: Fastest processing
- **Use when**: You need precise pose control

### **hubert_full_control**

- **Best for**: Maximum quality with control
- **Quality**: Highest lip-sync + pose control
- **Speed**: Slower processing
- **Use when**: Quality is most important

### **mfcc_pose_only**

- **Best for**: Simple pose animations
- **Quality**: Basic lip-sync
- **Speed**: Fast
- **Use when**: Simple animations are sufficient

### **hubert_pose_only**

- **Best for**: Natural head movements
- **Quality**: Good pose control
- **Speed**: Medium
- **Use when**: Realistic head movements matter

## Tips for Best Results

### Image Selection

✅ **Good Images**:

- Clear, well-lit portraits
- Front-facing or slight angle
- High resolution (512px+ recommended)
- Minimal background distractions
- Single person in frame

❌ **Avoid**:

- Blurry or low-resolution images
- Extreme angles or profile shots
- Multiple people in frame
- Heavy shadows or poor lighting
- Sunglasses or face obstructions

### Audio Preparation

✅ **Good Audio**:

- Clear speech with minimal noise
- WAV or high-quality MP3
- 16kHz sample rate (recommended)
- Moderate volume levels
- Single speaker

❌ **Avoid**:

- Very noisy or echo-heavy audio
- Multiple speakers talking simultaneously
- Very quiet or very loud audio
- Heavily compressed audio files
- Music-only files (no speech)

### Performance Tips

🚀 **Faster Generation**:

- Use GPU acceleration (CUDA)
- Lower `step_t` values (25-50)
- Disable super-resolution for 256x256
- Use MFCC inference types

🎯 **Better Quality**:

- Higher `step_t` values (50-100)
- Enable super-resolution for 512x512
- Use HuBERT inference types
- High-resolution input images

## Common Use Cases

### 1. Personal Avatar

```bash
# Create a personal talking avatar
# Use: High-quality portrait + your voice recording
# Settings: hubert_audio_only, face_sr=true, step_t=50
```

### 2. Character Animation

```bash
# Animate fictional characters
# Use: Character art + voice acting audio
# Settings: hubert_full_control, custom pose parameters
```

### 3. Educational Content

```bash
# Create educational videos
# Use: Historical figures + educational narration
# Settings: mfcc_full_control, consistent seed for series
```

### 4. Marketing Videos

```bash
# Generate spokesperson videos
# Use: Brand ambassador photo + marketing script
# Settings: hubert_audio_only, face_sr=true for quality
```

## Next Steps

Now that you've created your first video, explore more features:

### 📖 **Learn More**

- **[User Guide](user-guide.md)** - Complete feature overview
- **[Configuration](configuration.md)** - Advanced settings
- **[API Reference](api-reference.md)** - Integrate with applications

### 🔧 **Customize**

- Adjust pose parameters for different expressions
- Try different inference types for various effects
- Enable super-resolution for higher quality output
- Experiment with seeds for consistent character looks

### 🚀 **Scale Up**

- **[Deployment Guide](deployment.md)** - Production setup
- **[Docker Deployment](deployment.md#docker-deployment)** - Containerized scaling
- **[API Integration](api-reference.md)** - Automate generation

## Troubleshooting Quick Issues

### Generation Fails

- Check image and audio file formats
- Ensure sufficient system resources
- Verify GPU/CUDA setup for acceleration
- Check logs in `logs/` directory

### Poor Quality Results

- Use higher resolution input images
- Increase `step_t` parameter
- Enable super-resolution
- Try different inference types

### Slow Processing

- Enable GPU acceleration
- Reduce `step_t` parameter
- Disable super-resolution for speed
- Close other GPU-intensive applications

### Out of Memory

- Reduce batch size in settings
- Switch to CPU processing temporarily
- Use smaller input images
- Close other applications

Need more help? Check our **[Troubleshooting Guide](troubleshooting.md)** or visit our *
*[GitHub Issues](https://github.com/AlphaSphereDotAI/visualizr/issues)**.

---

*Ready to explore advanced features? Continue with the [User Guide](user-guide.md)!*
