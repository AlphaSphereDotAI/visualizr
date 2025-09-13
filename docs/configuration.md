# Configuration Guide

Complete guide to configuring Visualizr for different environments and use cases.

## Table of Contents

- [Overview](#overview)
- [Environment Variables](#environment-variables)
- [Settings Classes](#settings-classes)
- [Directory Configuration](#directory-configuration)
- [Model Configuration](#model-configuration)
- [Performance Tuning](#performance-tuning)
- [Production Settings](#production-settings)
- [Custom Configurations](#custom-configurations)

## Overview

Visualizr can be configured in several ways:

1. **Environment Variables**: System-level configuration
2. **Settings Classes**: Python object configuration
3. **Configuration Files**: `.env` file support
4. **Runtime Parameters**: Dynamic configuration during generation

### Configuration Priority

Settings are applied in this order (highest to lowest priority):

1. **Runtime Parameters**: Function call arguments
2. **Environment Variables**: System environment
3. **Settings Objects**: Python configuration
4. **Default Values**: Built-in defaults

## Environment Variables

### Server Configuration

Control the web interface and API server:

```bash
# Server binding
export VISUALIZR_HOST=0.0.0.0          # Default: 127.0.0.1 (localhost)
export VISUALIZR_PORT=7860             # Default: 7860

# Server features
export VISUALIZR_DEBUG=true            # Default: false
export VISUALIZR_ENABLE_API=true       # Default: true
export VISUALIZR_ENABLE_MONITORING=true # Default: true
export VISUALIZR_SHOW_ERROR=true       # Default: true
export VISUALIZR_SHOW_API=true         # Default: true

# MCP (Model Context Protocol)
export VISUALIZR_MCP_SERVER=true       # Default: true
```

### Device and Hardware

Configure GPU/CPU usage and hardware optimization:

```bash
# Device selection
export VISUALIZR_DEVICE=cuda           # Options: 'cuda', 'cpu'
export VISUALIZR_DEVICE_ID=0           # GPU device ID (default: 0)

# Memory management
export VISUALIZR_BATCH_SIZE=1          # Batch size (default: 1)
export VISUALIZR_MAX_MEMORY=8192       # Max GPU memory in MB
export VISUALIZR_MEMORY_FRACTION=0.8   # GPU memory fraction to use

# Performance
export VISUALIZR_NUM_WORKERS=2         # Number of worker threads
export VISUALIZR_ENABLE_MIXED_PRECISION=true  # FP16 optimization
```

### Model Configuration

Set default model parameters:

```bash
# Model selection
export VISUALIZR_INFER_TYPE=hubert_audio_only  # Default inference type
export VISUALIZR_MODEL_REPO=taocode/anitalker_ckpts  # HuggingFace model repo

# Generation parameters
export VISUALIZR_STEP_T=50             # Diffusion steps
export VISUALIZR_SEED=0                # Random seed
export VISUALIZR_MOTION_DIM=20         # Motion dimension
export VISUALIZR_DECODER_LAYERS=2      # Decoder layers

# Quality settings
export VISUALIZR_FACE_SR=false         # Enable super-resolution
export VISUALIZR_IMAGE_SIZE=256        # Base image size

# Pose defaults
export VISUALIZR_POSE_YAW=0.0         # Default yaw (-1.0 to 1.0)
export VISUALIZR_POSE_PITCH=0.0       # Default pitch (-1.0 to 1.0)
export VISUALIZR_POSE_ROLL=0.0        # Default roll (-1.0 to 1.0)
export VISUALIZR_FACE_LOCATION=0.5    # Default face location (0.0 to 1.0)
export VISUALIZR_FACE_SCALE=0.5       # Default face scale (0.0 to 1.0)
```

### Directory Configuration

Configure file locations and paths:

```bash
# Base directories
export VISUALIZR_BASE_DIR=/path/to/visualizr     # Base working directory
export VISUALIZR_ASSETS_DIR=/path/to/assets      # Input assets
export VISUALIZR_RESULTS_DIR=/path/to/results    # Output videos
export VISUALIZR_CKPTS_DIR=/path/to/models       # Model checkpoints
export VISUALIZR_LOGS_DIR=/path/to/logs          # Log files

# Asset subdirectories
export VISUALIZR_IMAGE_DIR=/path/to/assets/image   # Input images
export VISUALIZR_AUDIO_DIR=/path/to/assets/audio   # Input audio
export VISUALIZR_VIDEO_DIR=/path/to/assets/video   # Input videos

# Temporary directories
export VISUALIZR_FRAMES_DIR=/tmp/visualizr/frames  # Temporary frames
export VISUALIZR_CACHE_DIR=/tmp/visualizr/cache    # Cache directory
```

### Logging Configuration

Control logging behavior:

```bash
# Logging levels
export VISUALIZR_LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR
export VISUALIZR_LOG_FORMAT=detailed   # simple, detailed, json

# Log destinations
export VISUALIZR_LOG_FILE=visualizr.log      # Log file name
export VISUALIZR_LOG_TO_FILE=true            # Write to file
export VISUALIZR_LOG_TO_CONSOLE=true         # Write to console
export VISUALIZR_LOG_ROTATE=true             # Enable log rotation
export VISUALIZR_LOG_MAX_SIZE=10MB           # Max log file size
export VISUALIZR_LOG_BACKUP_COUNT=5          # Number of backup files
```

## Settings Classes

### Creating Custom Settings

```python
from visualizr.settings import Settings, ModelSettings, DirectorySettings
from pathlib import Path

# Custom settings with Python
custom_settings = Settings(
    model=ModelSettings(
        device='cuda',
        infer_type='hubert_full_control',
        face_sr=True,
        step_t=100,
        seed=42,
        pose_yaw=0.1,
        pose_pitch=-0.05,
        pose_roll=0.0,
        face_location=0.6,
        face_scale=0.7
    ),
    directory=DirectorySettings(
        base=Path('/custom/base'),
        results=Path('/custom/results'),
        assets=Path('/custom/assets'),
        checkpoint=Path('/custom/models')
    )
)

# Use custom settings
from visualizr.app.builder import App
app = App(custom_settings)
```

### Settings Validation

```python
from visualizr.settings import Settings, ModelSettings
from pydantic import ValidationError

try:
    # This will fail validation
    settings = Settings(
        model=ModelSettings(
            pose_yaw=2.0,  # Invalid: outside -1.0 to 1.0 range
            step_t=-10     # Invalid: negative steps
        )
    )
except ValidationError as e:
    print(f"Configuration error: {e}")
    # Handle validation errors appropriately
```

## Directory Configuration

### Default Directory Structure

Visualizr uses this default structure:

```
visualizr/
├── ckpts/              # Model checkpoints and weights
│   ├── stage1.ckpt
│   ├── stage2_*.ckpt
│   └── chinese-hubert-large/
├── assets/             # Input files
│   ├── image/          # Portrait images (.jpg, .png)
│   ├── audio/          # Speech audio (.wav, .mp3)
│   └── video/          # Video files (future use)
├── results/            # Generated output videos
│   ├── output.mp4
│   └── output_SR.mp4
├── logs/               # Application logs
│   └── visualizr.log
└── .cache/             # Cached models and temp files
```

### Custom Directory Setup

```python
from visualizr.settings import DirectorySettings
from pathlib import Path

# Custom directories
custom_dirs = DirectorySettings(
    base=Path('/workspace/visualizr'),
    results=Path('/workspace/output'),
    checkpoint=Path('/shared/models'),
    assets=Path('/workspace/assets'),
    image=Path('/workspace/assets/portraits'),
    audio=Path('/workspace/assets/speech'),
    video=Path('/workspace/assets/clips'),
    frames=Path('/tmp/frames'),
    log=Path('/workspace/logs')
)

# Directories are created automatically
settings = Settings(directory=custom_dirs)
```

### Network Storage Integration

```python
# Example: Using network-attached storage
from pathlib import Path

network_dirs = DirectorySettings(
    # Local fast storage for temp files
    base=Path('/tmp/visualizr'),
    frames=Path('/tmp/visualizr/frames'),
    log=Path('/var/log/visualizr'),
    
    # Network storage for persistent files
    assets=Path('/mnt/nfs/visualizr/assets'),
    results=Path('/mnt/nfs/visualizr/results'),
    checkpoint=Path('/mnt/nfs/shared/models')
)
```

## Model Configuration

### Inference Type Selection

Configure default inference behavior:

```python
from visualizr.settings import ModelSettings

# Speed-optimized configuration
speed_model = ModelSettings(
    infer_type='mfcc_full_control',  # Fastest processing
    device='cuda',                   # Use GPU
    step_t=25,                       # Fewer steps
    face_sr=False,                   # Skip super-resolution
    batch_size=1                     # Conservative memory use
)

# Quality-optimized configuration
quality_model = ModelSettings(
    infer_type='hubert_full_control', # Best quality
    device='cuda',                    # Use GPU
    step_t=100,                      # More steps
    face_sr=True,                    # Enable super-resolution
    batch_size=1                     # Conservative memory use
)

# Balanced configuration
balanced_model = ModelSettings(
    infer_type='hubert_audio_only',  # Good balance
    device='cuda',                   # Use GPU
    step_t=50,                       # Standard steps
    face_sr=False,                   # Standard resolution
    batch_size=1                     # Conservative memory use
)
```

### Checkpoint Management

```python
from visualizr.settings import Checkpoint
from pathlib import Path

# Custom model paths
custom_checkpoints = Checkpoint(
    stage_1=Path('/models/stage1_custom.ckpt'),
    mfcc_full_control=Path('/models/mfcc_full.ckpt'),
    hubert_audio_only=Path('/models/hubert_audio.ckpt'),
    # ... other checkpoint paths
)

model_settings = ModelSettings(
    checkpoint=custom_checkpoints,
    repo_id='custom/model-repo'  # Custom HuggingFace repo
)
```

### Hardware-Specific Settings

```python
# GPU configuration for different hardware
def get_gpu_config(gpu_name: str) -> ModelSettings:
    """Get optimized settings for specific GPU"""
    
    configs = {
        'rtx_3060': ModelSettings(
            device='cuda',
            batch_size=1,
            step_t=50,
            face_sr=False,  # Limited VRAM
        ),
        'rtx_3080': ModelSettings(
            device='cuda',
            batch_size=1,
            step_t=75,
            face_sr=True,   # More VRAM available
        ),
        'rtx_4090': ModelSettings(
            device='cuda',
            batch_size=2,   # Can handle larger batches
            step_t=100,
            face_sr=True,
        ),
        'cpu_only': ModelSettings(
            device='cpu',
            batch_size=1,
            step_t=25,      # Faster processing needed
            face_sr=False,
        )
    }
    
    return configs.get(gpu_name, configs['rtx_3060'])

# Usage
settings = Settings(model=get_gpu_config('rtx_3080'))
```

## Performance Tuning

### Memory Optimization

```bash
# Reduce memory usage
export VISUALIZR_BATCH_SIZE=1           # Minimum batch size
export VISUALIZR_MEMORY_FRACTION=0.7    # Limit GPU memory
export VISUALIZR_ENABLE_MIXED_PRECISION=true  # Use FP16

# Clear cache between generations
export VISUALIZR_CLEAR_CACHE=true

# Use CPU fallback for large models
export VISUALIZR_CPU_FALLBACK=true
```

### Speed Optimization

```python
from visualizr.settings import ModelSettings

speed_settings = ModelSettings(
    # Use fastest inference type
    infer_type='mfcc_full_control',
    
    # Reduce quality for speed
    step_t=25,
    face_sr=False,
    
    # Enable optimizations
    device='cuda',
    fp16=True,  # Mixed precision
    
    # Optimize model parameters
    motion_dim=16,      # Reduce motion complexity
    decoder_layers=1    # Simpler decoder
)
```

### Batch Processing Settings

```python
# Configuration for batch processing
batch_settings = ModelSettings(
    # Consistent results
    seed=42,
    
    # Optimized for throughput
    step_t=50,
    face_sr=False,
    
    # Memory-efficient
    batch_size=1,
    device='cuda'
)

# Use with multiple workers
import multiprocessing
num_workers = min(multiprocessing.cpu_count(), 4)
```

## Production Settings

### High-Availability Configuration

```bash
# Production server settings
export VISUALIZR_HOST=0.0.0.0
export VISUALIZR_PORT=7860
export VISUALIZR_DEBUG=false
export VISUALIZR_ENABLE_MONITORING=true

# Logging for production
export VISUALIZR_LOG_LEVEL=INFO
export VISUALIZR_LOG_FILE=/var/log/visualizr/app.log
export VISUALIZR_LOG_ROTATE=true
export VISUALIZR_LOG_MAX_SIZE=50MB
export VISUALIZR_LOG_BACKUP_COUNT=10

# Performance optimization
export VISUALIZR_ENABLE_MIXED_PRECISION=true
export VISUALIZR_MEMORY_FRACTION=0.8
export VISUALIZR_NUM_WORKERS=4

# Security
export VISUALIZR_SHOW_ERROR=false      # Hide detailed errors
export VISUALIZR_ENABLE_CORS=true      # Enable CORS if needed
```

### Docker Production Configuration

```dockerfile
# Environment variables in Dockerfile
ENV VISUALIZR_HOST=0.0.0.0
ENV VISUALIZR_PORT=7860
ENV VISUALIZR_DEBUG=false
ENV VISUALIZR_LOG_LEVEL=INFO
ENV VISUALIZR_DEVICE=cuda
ENV VISUALIZR_MEMORY_FRACTION=0.8
ENV VISUALIZR_RESULTS_DIR=/app/results
ENV VISUALIZR_CKPTS_DIR=/app/models
ENV VISUALIZR_LOGS_DIR=/app/logs
```

### Load Balancing Settings

```python
# Configuration for load-balanced deployment
production_settings = Settings(
    model=ModelSettings(
        device='cuda',
        infer_type='hubert_audio_only',  # Good balance
        step_t=50,                       # Standard quality
        face_sr=False,                   # Consistent performance
        seed=None                        # Random results per instance
    ),
    directory=DirectorySettings(
        # Shared model storage
        checkpoint=Path('/shared/models'),
        # Instance-specific directories
        results=Path(f'/app/results/{instance_id}'),
        frames=Path(f'/tmp/visualizr/{instance_id}'),
        log=Path(f'/var/log/visualizr/{instance_id}')
    )
)
```

## Custom Configurations

### Environment-Specific Configs

```python
import os
from visualizr.settings import Settings, ModelSettings, DirectorySettings

def get_environment_config(env: str = None) -> Settings:
    """Get configuration for specific environment"""
    
    env = env or os.getenv('VISUALIZR_ENV', 'development')
    
    configs = {
        'development': Settings(
            model=ModelSettings(
                device='cuda' if torch.cuda.is_available() else 'cpu',
                infer_type='hubert_audio_only',
                step_t=25,      # Fast for development
                face_sr=False,
                seed=42         # Reproducible results
            )
        ),
        
        'testing': Settings(
            model=ModelSettings(
                device='cpu',   # Consistent test environment
                infer_type='mfcc_full_control',
                step_t=10,      # Very fast for tests
                face_sr=False,
                seed=12345      # Fixed seed for tests
            ),
            directory=DirectorySettings(
                base=Path('/tmp/visualizr_test')
            )
        ),
        
        'production': Settings(
            model=ModelSettings(
                device='cuda',
                infer_type='hubert_audio_only',
                step_t=50,
                face_sr=True,   # High quality for production
                seed=None       # Random results
            ),
            directory=DirectorySettings(
                results=Path('/data/visualizr/results'),
                checkpoint=Path('/data/models'),
                log=Path('/var/log/visualizr')
            )
        )
    }
    
    return configs.get(env, configs['development'])

# Usage
settings = get_environment_config()
```

### Application-Specific Configs

```python
# Configuration for different use cases
def get_app_config(use_case: str) -> Settings:
    """Get configuration optimized for specific use case"""
    
    configs = {
        'content_creation': Settings(
            model=ModelSettings(
                infer_type='hubert_full_control',
                step_t=100,
                face_sr=True,
                seed=42  # Consistent results for creators
            )
        ),
        
        'real_time_preview': Settings(
            model=ModelSettings(
                infer_type='mfcc_full_control',
                step_t=15,
                face_sr=False,
                device='cuda'
            )
        ),
        
        'batch_processing': Settings(
            model=ModelSettings(
                infer_type='hubert_audio_only',
                step_t=50,
                face_sr=False,
                seed=None  # Random for variety
            )
        ),
        
        'research': Settings(
            model=ModelSettings(
                infer_type='hubert_full_control',
                step_t=200,     # Maximum quality
                face_sr=True,
                seed=12345      # Reproducible
            )
        )
    }
    
    return configs.get(use_case, configs['content_creation'])
```

### Configuration Validation

```python
from pydantic import ValidationError
from visualizr.settings import Settings

def validate_config(settings: Settings) -> tuple[bool, list[str]]:
    """Validate configuration and return issues"""
    issues = []
    
    try:
        # Test model loading
        if not settings.directory.checkpoint.exists():
            issues.append("Checkpoint directory does not exist")
        
        # Test device availability
        if settings.model.device == 'cuda' and not torch.cuda.is_available():
            issues.append("CUDA requested but not available")
        
        # Test directory permissions
        if not os.access(settings.directory.results, os.W_OK):
            issues.append("Results directory is not writable")
        
        # Test parameter ranges
        if not -1.0 <= settings.model.pose_yaw <= 1.0:
            issues.append("pose_yaw outside valid range")
        
        return len(issues) == 0, issues
        
    except ValidationError as e:
        issues.extend([str(error) for error in e.errors()])
        return False, issues

# Usage
settings = get_environment_config('production')
is_valid, issues = validate_config(settings)

if not is_valid:
    print("Configuration issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Dynamic Configuration Updates

```python
class DynamicSettings:
    """Settings that can be updated at runtime"""
    
    def __init__(self, initial_settings: Settings):
        self.settings = initial_settings
        self._observers = []
    
    def update_model_settings(self, **kwargs):
        """Update model settings dynamically"""
        current_model = self.settings.model
        
        # Create new model settings with updates
        new_model = ModelSettings(
            **{**current_model.dict(), **kwargs}
        )
        
        # Update settings
        self.settings = Settings(
            model=new_model,
            directory=self.settings.directory
        )
        
        # Notify observers
        self._notify_observers()
    
    def add_observer(self, callback):
        """Add callback for configuration changes"""
        self._observers.append(callback)
    
    def _notify_observers(self):
        """Notify all observers of changes"""
        for callback in self._observers:
            callback(self.settings)

# Usage
dynamic_config = DynamicSettings(get_environment_config())

def on_config_change(new_settings):
    print(f"Configuration updated: {new_settings.model.infer_type}")

dynamic_config.add_observer(on_config_change)
dynamic_config.update_model_settings(step_t=75, face_sr=True)
```

## Configuration Files

### Environment File (.env)

Create a `.env` file in your working directory:

```bash
# .env file for Visualizr configuration

# Server settings
VISUALIZR_HOST=0.0.0.0
VISUALIZR_PORT=7860
VISUALIZR_DEBUG=false

# Model settings
VISUALIZR_DEVICE=cuda
VISUALIZR_INFER_TYPE=hubert_audio_only
VISUALIZR_STEP_T=50
VISUALIZR_FACE_SR=false

# Directories (use absolute paths)
VISUALIZR_RESULTS_DIR=/workspace/results
VISUALIZR_CKPTS_DIR=/workspace/models
VISUALIZR_ASSETS_DIR=/workspace/assets

# Performance
VISUALIZR_BATCH_SIZE=1
VISUALIZR_MEMORY_FRACTION=0.8

# Logging
VISUALIZR_LOG_LEVEL=INFO
VISUALIZR_LOG_FILE=visualizr.log
```

### Configuration Loading

```python
from dotenv import load_dotenv
from visualizr.settings import Settings

# Load environment variables from .env file
load_dotenv('.env')

# Settings automatically pick up environment variables
settings = Settings()

# Or load from specific file
load_dotenv('/path/to/custom.env')
settings = Settings()
```

### JSON Configuration (Custom Implementation)

```python
import json
from pathlib import Path
from visualizr.settings import Settings, ModelSettings, DirectorySettings

def load_config_from_json(config_path: Path) -> Settings:
    """Load configuration from JSON file"""
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return Settings(
        model=ModelSettings(**config_data.get('model', {})),
        directory=DirectorySettings(**config_data.get('directory', {}))
    )

def save_config_to_json(settings: Settings, config_path: Path):
    """Save configuration to JSON file"""
    
    config_data = {
        'model': settings.model.dict(),
        'directory': {
            k: str(v) for k, v in settings.directory.dict().items()
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

# Usage
config_file = Path('visualizr_config.json')
settings = load_config_from_json(config_file)
```

---

*Ready to deploy in production? Check out the [Deployment Guide](deployment.md)!*
