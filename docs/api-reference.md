# API Reference

Complete reference for all Visualizr APIs, including the Gradio web interface, Python
API, and REST endpoints.

## Table of Contents

- [Overview](#overview)
- [REST API](#rest-api)
- [Python API](#python-api)
- [Gradio Interface API](#gradio-interface-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)

## Overview

Visualizr provides multiple API interfaces:

1. **REST API**: HTTP-based endpoints for web integration
2. **Python API**: Direct Python module integration
3. **Gradio Interface**: Web UI with API access
4. **Command Line**: Shell command interface

### Base URL

When Visualizr is running, APIs are available at:

- **Default**: `http://localhost:7860`
- **Custom**: `http://HOST:PORT` (based on configuration)

### Authentication

Currently, Visualizr operates without authentication for local development. For
production deployments, consider implementing reverse proxy authentication.

## REST API

### API Documentation UI

Access the interactive API documentation at:

- **Swagger UI**: `http://localhost:7860/docs`
- **ReDoc**: `http://localhost:7860/redoc`

### Endpoints

#### Health Check

Check if the service is running.

**Endpoint**: `GET /heartbeat`

**Response**:

```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

**Example**:

```bash
curl http://localhost:7860/heartbeat
```

#### Generate Video from Paths

Generate a video using uploaded image and audio files.

**Endpoint**: `POST /api/generate_video`

**Parameters**:

| Parameter       | Type    | Required | Description                                     |
|-----------------|---------|----------|-------------------------------------------------|
| `infer_type`    | string  | Yes      | Inference model type                            |
| `image_path`    | file    | Yes      | Portrait image file                             |
| `audio_path`    | file    | Yes      | Audio file for lip-sync                         |
| `face_sr`       | boolean | No       | Enable super-resolution (default: false)        |
| `pose_yaw`      | float   | No       | Left/right rotation (-1.0 to 1.0, default: 0.0) |
| `pose_pitch`    | float   | No       | Up/down tilt (-1.0 to 1.0, default: 0.0)        |
| `pose_roll`     | float   | No       | Side tilt (-1.0 to 1.0, default: 0.0)           |
| `face_location` | float   | No       | Vertical position (0.0 to 1.0, default: 0.5)    |
| `face_scale`    | float   | No       | Face size (0.0 to 1.0, default: 0.5)            |
| `step_t`        | integer | No       | Generation steps (default: 50)                  |
| `seed`          | integer | No       | Random seed (default: 0)                        |

**Request Example**:

```bash
curl -X POST "http://localhost:7860/api/generate_video" \
  -F "infer_type=hubert_audio_only" \
  -F "image_path=@portrait.jpg" \
  -F "audio_path=@speech.wav" \
  -F "face_sr=false" \
  -F "pose_yaw=0.1" \
  -F "pose_pitch=-0.05" \
  -F "step_t=50"
```

**Response**:

```json
{
  "success": true,
  "data": {
    "video_256": {
      "path": "/results/portrait-speech.mp4",
      "url": "http://localhost:7860/file=/results/portrait-speech.mp4"
    },
    "video_512": null,
    "message": "Video (256 ✕ 256 only) generated successfully!"
  },
  "processing_time": 45.2
}
```

**Error Response**:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Image file not found or invalid format",
    "details": "Supported formats: JPG, JPEG, PNG"
  }
}
```

#### Generate Video from Character Name

Generate video using pre-loaded character images.

**Endpoint**: `POST /api/generate_video_from_name`

**Parameters**:

| Parameter        | Type   | Required | Description                                  |
|------------------|--------|----------|----------------------------------------------|
| `name`           | string | Yes      | Character name (from assets/image/)          |
| `audio_path`     | file   | Yes      | Audio file for lip-sync                      |
| `infer_type`     | string | No       | Inference model (default: hubert_audio_only) |
| Other parameters | -      | No       | Same as generate_video endpoint              |

**Request Example**:

```bash
curl -X POST "http://localhost:7860/api/generate_video_from_name" \
  -F "name=napoleon" \
  -F "audio_path=@speech.wav" \
  -F "infer_type=hubert_full_control" \
  -F "face_sr=true"
```

#### List Available Characters

Get list of available character images.

**Endpoint**: `GET /api/characters`

**Response**:

```json
{
  "characters": [
    "napoleon",
    "character1",
    "portrait_sample"
  ]
}
```

#### Upload File

Upload files to the server.

**Endpoint**: `POST /upload`

**Parameters**:

- `files`: Multi-part file upload

**Response**:

```json
{
  "filename": "uploaded_file.jpg",
  "path": "/tmp/gradio/uploaded_file.jpg",
  "size": 1024000
}
```

#### Download File

Download generated videos or other files.

**Endpoint**: `GET /file={filepath}`

**Example**:

```bash
# Download generated video
curl -O http://localhost:7860/file=/results/output.mp4
```

## Python API

### Core Classes

#### App Class

Main application class for video generation.

```python
from visualizr.app.runner import app
from visualizr.app.builder import App
from visualizr.settings import Settings

# Use default app instance
result = app.generate_video(...)

# Or create custom instance
settings = Settings()
custom_app = App(settings)
result = custom_app.generate_video(...)
```

#### Settings Class

Configuration management.

```python
from visualizr.settings import Settings, ModelSettings, DirectorySettings

# Default settings
settings = Settings()

# Custom settings
settings = Settings(
    model=ModelSettings(
        device='cuda',
        infer_type='hubert_audio_only',
        face_sr=True
    ),
    directory=DirectorySettings(
        results=Path('./custom_results')
    )
)
```

### Methods

#### generate_video()

Generate video from image and audio paths.

**Signature**:

```python
def generate_video(
        self,
        infer_type: Literal[
            "mfcc_full_control",
            "mfcc_pose_only",
            "hubert_pose_only",
            "hubert_audio_only",
            "hubert_full_control"
        ],
        image_path: str | Path,
        audio_path: str | Path,
        face_sr: bool,
        pose_yaw: float,
        pose_pitch: float,
        pose_roll: float,
        face_location: float,
        face_scale: float,
        step_t: int,
        seed: int
) -> tuple[Video | None, Video | None, Markdown]
```

**Parameters**:

- `infer_type`: Audio processing model type
- `image_path`: Path to portrait image file
- `audio_path`: Path to audio file
- `face_sr`: Enable super-resolution output
- `pose_yaw`: Left/right head rotation (-1.0 to 1.0)
- `pose_pitch`: Up/down head tilt (-1.0 to 1.0)
- `pose_roll`: Side head tilt (-1.0 to 1.0)
- `face_location`: Vertical face position (0.0 to 1.0)
- `face_scale`: Face size in frame (0.0 to 1.0)
- `step_t`: Number of diffusion steps
- `seed`: Random seed for reproducibility

**Returns**:

- `tuple[Video | None, Video | None, Markdown]`
    - `Video`: 256x256 generated video
    - `Video`: 512x512 super-resolution video (if enabled)
    - `Markdown`: Status message

**Example**:

```python
from visualizr.app.runner import app

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

video_256, video_512, message = result
if video_256:
    print(f"Generated: {video_256.value}")
else:
    print(f"Failed: {message.value}")
```

#### generate_video_from_name()

Generate video using pre-loaded character image.

**Signature**:

```python
def generate_video_from_name(
        self,
        name: str,
        infer_type: Literal[...],  # Same as above
        audio_path: str | Path,
        face_sr: bool,
        pose_yaw: float,
        pose_pitch: float,
        pose_roll: float,
        face_location: float,
        face_scale: float,
        step_t: int,
        seed: int
) -> tuple[Video | None, Video | None, Markdown]
```

**Parameters**:

- `name`: Character name (filename without extension in assets/image/)
- Other parameters same as `generate_video()`

**Example**:

```python
result = app.generate_video_from_name(
    name='napoleon',
    infer_type='hubert_audio_only',
    audio_path='speech.wav',
    face_sr=True,
    # ... other parameters
)
```

### Utility Functions

#### get_character_names()

Get list of available character names.

```python
names = app._get_character_names()
print(names)  # ['napoleon', 'character1', ...]
```

#### get_image_path()

Get full path for character image.

```python
path = app._get_image_path('napoleon')
print(path)  # Path('assets/image/napoleon.jpg')
```

### Error Handling

```python
from pathlib import Path
from visualizr.app.runner import app

try:
    result = app.generate_video(
        infer_type='hubert_audio_only',
        image_path='nonexistent.jpg',  # This will cause an error
        audio_path='speech.wav',
        # ... other parameters
    )

    video_256, video_512, message = result

    if video_256 is None:
        print(f"Generation failed: {message.value}")
    else:
        print(f"Success: {message.value}")

except Exception as e:
    print(f"Error: {e}")
```

## Gradio Interface API

### GUI Object Access

Access the Gradio interface programmatically:

```python
from visualizr.app.runner import app

# Get Gradio Blocks object
gui = app.gui()

# Launch with custom settings
gui.queue(api_open=True).launch(
    server_port=8080,
    server_name="0.0.0.0",
    debug=False,
    show_api=True
)
```

### Interface Components

```python
# Access specific components (for advanced customization)
with app.gui() as interface:
    # Components are created within the context
    # Refer to source code for specific component names
    pass
```

### API Integration

```python
from gradio_client import Client

# Connect to running instance
client = Client("http://localhost:7860")

# Use the API
result = client.predict(
    infer_type="hubert_audio_only",
    image_path="/path/to/image.jpg",
    audio_path="/path/to/audio.wav",
    face_sr=False,
    # ... other parameters
    api_name="/generate_video"
)

print(result)
```

## Error Handling

### Error Types

#### Input Validation Errors

```python
# Common validation errors
{
    "error": "INVALID_IMAGE_PATH",
    "message": "Image file not found or invalid format",
    "code": 400
}

{
    "error": "INVALID_AUDIO_PATH",
    "message": "Audio file not found or unsupported format",
    "code": 400
}

{
    "error": "INVALID_PARAMETERS",
    "message": "Parameter values out of valid range",
    "details": {"pose_yaw": "Must be between -1.0 and 1.0"},
    "code": 400
}
```

#### Processing Errors

```python
{
    "error": "GENERATION_FAILED",
    "message": "Video generation process failed",
    "details": "Insufficient GPU memory",
    "code": 500
}

{
    "error": "MODEL_LOAD_ERROR",
    "message": "Failed to load AI models",
    "details": "CUDA out of memory",
    "code": 500
}
```

#### System Errors

```python
{
    "error": "RESOURCE_ERROR",
    "message": "Insufficient system resources",
    "details": "Not enough disk space for output",
    "code": 507
}
```

### Error Response Format

All API errors follow this structure:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_TYPE",
    "message": "Human-readable error message",
    "details": "Additional error details (optional)",
    "timestamp": "2025-01-01T12:00:00Z"
  }
}
```

### Error Handling in Python

```python
from visualizr.app.runner import app
from gradio import Error

try:
    result = app.generate_video(
        # ... parameters
    )

    video_256, video_512, message = result

    # Check for generation failure
    if "Error:" in message.value:
        print(f"Generation failed: {message.value}")
    else:
        print(f"Success: {message.value}")

except Error as e:
    print(f"Gradio Error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Rate Limiting

### Current Implementation

Visualizr currently does not implement rate limiting, but for production use, consider:

### Recommended Limits

- **Per user**: 10 requests per minute
- **Per IP**: 50 requests per hour
- **Concurrent**: 3 simultaneous generations
- **File size**: 10MB max per upload

### Implementation Example

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Example rate limiting (not implemented in Visualizr)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["10 per minute", "50 per hour"]
)


@limiter.limit("5 per minute")
def generate_video_endpoint():
    # ... generation logic
    pass
```

## Examples

### Basic REST API Usage

```bash
#!/bin/bash
# Basic video generation via REST API

# Upload files and generate video
curl -X POST "http://localhost:7860/api/generate_video" \
  -F "infer_type=hubert_audio_only" \
  -F "image_path=@assets/image/portrait.jpg" \
  -F "audio_path=@assets/audio/speech.wav" \
  -F "face_sr=false" \
  -F "step_t=50" \
  -o response.json

# Parse response and download video
video_url=$(jq -r '.data.video_256.url' response.json)
curl -o generated_video.mp4 "$video_url"
```

### Python Integration

```python
#!/usr/bin/env python3
"""
Complete example of Python API integration
"""

import os
import json
from pathlib import Path
from visualizr.app.runner import app


def generate_avatar_video(
        image_file: str,
        audio_file: str,
        output_dir: str = "output",
        quality: str = "balanced"
) -> dict:
    """
    Generate avatar video with different quality presets
    
    Args:
        image_file: Path to portrait image
        audio_file: Path to speech audio
        output_dir: Directory for output files
        quality: 'fast', 'balanced', or 'high'
    
    Returns:
        dict: Generation results
    """

    # Quality presets
    presets = {
        'fast': {
            'infer_type': 'mfcc_full_control',
            'face_sr': False,
            'step_t': 25
        },
        'balanced': {
            'infer_type': 'hubert_audio_only',
            'face_sr': False,
            'step_t': 50
        },
        'high': {
            'infer_type': 'hubert_full_control',
            'face_sr': True,
            'step_t': 100
        }
    }

    config = presets.get(quality, presets['balanced'])

    try:
        # Generate video
        result = app.generate_video(
            infer_type=config['infer_type'],
            image_path=image_file,
            audio_path=audio_file,
            face_sr=config['face_sr'],
            pose_yaw=0.0,
            pose_pitch=0.0,
            pose_roll=0.0,
            face_location=0.5,
            face_scale=0.5,
            step_t=config['step_t'],
            seed=42
        )

        video_256, video_512, message = result

        if video_256:
            return {
                'success': True,
                'video_256': video_256.value,
                'video_512': video_512.value if video_512 else None,
                'message': message.value
            }
        else:
            return {
                'success': False,
                'error': message.value
            }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# Usage example
if __name__ == "__main__":
    result = generate_avatar_video(
        image_file="assets/image/portrait.jpg",
        audio_file="assets/audio/speech.wav",
        quality="balanced"
    )

    if result['success']:
        print(f"Video generated: {result['video_256']}")
    else:
        print(f"Generation failed: {result['error']}")
```

### Batch Processing

```python
#!/usr/bin/env python3
"""
Batch process multiple image-audio combinations
"""

import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from visualizr.app.runner import app


def process_single_video(params):
    """Process a single video generation"""
    image_path, audio_path, output_name = params

    print(f"Processing: {output_name}")
    start_time = time.time()

    try:
        result = app.generate_video(
            infer_type='hubert_audio_only',
            image_path=str(image_path),
            audio_path=str(audio_path),
            face_sr=False,
            pose_yaw=0.0,
            pose_pitch=0.0,
            pose_roll=0.0,
            face_location=0.5,
            face_scale=0.5,
            step_t=50,
            seed=42
        )

        processing_time = time.time() - start_time
        video_256, video_512, message = result

        return {
            'name': output_name,
            'success': video_256 is not None,
            'video_path': video_256.value if video_256 else None,
            'message': message.value,
            'processing_time': processing_time
        }

    except Exception as e:
        return {
            'name': output_name,
            'success': False,
            'error': str(e),
            'processing_time': time.time() - start_time
        }


def batch_process_videos(image_dir, audio_dir, max_workers=2):
    """Process all image-audio combinations"""

    image_dir = Path(image_dir)
    audio_dir = Path(audio_dir)

    # Find all image-audio combinations
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.mp3"))

    # Create parameter combinations
    tasks = []
    for img in image_files:
        for audio in audio_files:
            output_name = f"{img.stem}_{audio.stem}"
            tasks.append((img, audio, output_name))

    print(f"Processing {len(tasks)} video combinations...")

    # Process with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_video, tasks))

    # Save results
    with open('batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = sum(1 for r in results if r['success'])
    total_time = sum(r['processing_time'] for r in results)

    print(f"\nBatch Processing Complete:")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time: {total_time / len(results):.1f}s per video")

    return results


# Usage
if __name__ == "__main__":
    results = batch_process_videos(
        image_dir="assets/image",
        audio_dir="assets/audio",
        max_workers=1  # Adjust based on GPU memory
    )
```

### Web Service Integration

```python
#!/usr/bin/env python3
"""
FastAPI wrapper for Visualizr
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
from pathlib import Path
from visualizr.app.runner import app

api = FastAPI(title="Visualizr API", version="1.0.0")


@api.post("/generate")
async def generate_video(
        image: UploadFile = File(...),
        audio: UploadFile = File(...),
        infer_type: str = Form("hubert_audio_only"),
        face_sr: bool = Form(False),
        step_t: int = Form(50)
):
    """Generate video from uploaded files"""

    # Save uploaded files temporarily
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Save image file
        image_path = tmp_path / f"image{Path(image.filename).suffix}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

        # Save audio file  
        audio_path = tmp_path / f"audio{Path(audio.filename).suffix}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        try:
            # Generate video
            result = app.generate_video(
                infer_type=infer_type,
                image_path=str(image_path),
                audio_path=str(audio_path),
                face_sr=face_sr,
                pose_yaw=0.0,
                pose_pitch=0.0,
                pose_roll=0.0,
                face_location=0.5,
                face_scale=0.5,
                step_t=step_t,
                seed=0
            )

            video_256, video_512, message = result

            if video_256:
                # Return the generated video file
                return FileResponse(
                    video_256.value,
                    media_type="video/mp4",
                    filename="generated_video.mp4"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Generation failed: {message.value}"
                )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Processing error: {str(e)}"
            )


@api.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "visualizr"}

# Run with: uvicorn api:api --host 0.0.0.0 --port 8000
```

---

*For more advanced integrations, see the [Deployment Guide](deployment.md)
and [User Guide](user-guide.md).*
