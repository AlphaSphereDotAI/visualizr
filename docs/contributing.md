# Contributing Guide

Welcome to Visualizr! This guide will help you get started with contributing to the
project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Contributing Workflow](#contributing-workflow)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Guidelines](#issue-guidelines)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Ways to Contribute

We welcome all types of contributions:

- 🐛 **Bug reports** - Help us identify and fix issues
- 💡 **Feature requests** - Suggest new capabilities
- 🔧 **Code contributions** - Implement features and fixes
- 📚 **Documentation** - Improve guides and references
- 🧪 **Testing** - Add tests and improve coverage
- 🎨 **Design** - UI/UX improvements
- 🌍 **Translation** - Localization support
- 💬 **Community** - Help other users

### Prerequisites

Before contributing, ensure you have:

- **Python 3.10** installed
- **Git** for version control
- **GitHub account** for collaboration
- **Basic knowledge** of Python and AI/ML concepts
- **NVIDIA GPU** (optional but recommended for testing)

### First-time Contributors

New to open source? Start here:

1. **Read the Code of Conduct** (see [Community Guidelines](#community-guidelines))
2. **Browse existing issues** labeled `good first issue`
3. **Fork the repository** and clone locally
4. **Set up development environment** (see [Development Setup](#development-setup))
5. **Make a small change** and create your first pull request

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/visualizr.git
cd visualizr

# Add upstream remote
git remote add upstream https://github.com/AlphaSphereDotAI/visualizr.git
```

### 2. Environment Setup

#### Option A: Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install development dependencies
uv sync --dev
```

#### Option B: Using pip

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Test basic functionality
python -c "import visualizr; print('✓ Visualizr imported successfully')"

# Run health check
python -c "
from visualizr.settings import Settings
settings = Settings()
print('✓ Settings loaded successfully')
"

# Test GPU availability (if available)
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

### 4. Development Tools Setup

#### Code Quality Tools

```bash
# Install pre-commit hooks
pre-commit install

# Run code quality checks
ruff check src/
ruff format src/
mypy src/
```

#### IDE Configuration

**VS Code** (`settings.json`):

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests/"
  ]
}
```

**PyCharm**:

- Set interpreter to `.venv/bin/python`
- Enable Ruff for linting
- Configure pytest as test runner
- Set code style to match project standards

### 5. Running Development Server

```bash
# Run with development settings
export VISUALIZR_DEBUG=true
export VISUALIZR_LOG_LEVEL=DEBUG
python -m visualizr

# Or using custom configuration
VISUALIZR_PORT=8080 VISUALIZR_DEBUG=true python -m visualizr
```

## Code Standards

We follow strict code quality standards to maintain a clean, readable codebase.

### Python Style Guide

Based on **PEP 8** with project-specific adaptations:

#### Code Formatting

```python
# Use Ruff for formatting (similar to Black)
# Line length: 88 characters
# Use double quotes for strings
# Use trailing commas in multi-line structures

# Good example
def generate_video(
        image_path: str,
        audio_path: str,
        infer_type: str = "hubert_audio_only",
        face_sr: bool = False,
) -> tuple[Video | None, Video | None, str]:
    """Generate talking avatar video from image and audio."""
    pass
```

#### Naming Conventions

```python
# Variables and functions: snake_case
user_input = "example"


def process_audio_features() -> None:
    pass


# Classes: PascalCase  
class ModelSettings:
    pass


# Constants: UPPER_SNAKE_CASE
DEFAULT_STEP_COUNT = 50
MAX_FILE_SIZE = 10 * 1024 * 1024


# Private attributes: leading underscore
class App:
    def __init__(self):
        self._model_cache = {}
```

#### Type Hints

```python
# Always use type hints for public APIs
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path


def generate_video(
        image_path: Union[str, Path],
        audio_path: Union[str, Path],
        settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Generate video with proper type annotations."""
    pass


# Use modern union syntax in Python 3.10+
def modern_function(value: str | int) -> list[str]:
    """Use modern type syntax."""
    return [str(value)]
```

#### Docstrings

Follow **Google-style** docstrings:

```python
def generate_video(
    image_path: str,
    audio_path: str,
    infer_type: str = "hubert_audio_only",
) -> tuple[Video | None, Video | None, str]:
    """Generate talking avatar video from image and audio.
    
    This function processes a portrait image and speech audio to create
    a synchronized talking avatar video using advanced AI models.
    
    Args:
        image_path: Path to the input portrait image.
        audio_path: Path to the input speech audio file.
        infer_type: Type of inference model to use. Options are:
            - "hubert_audio_only": HuBERT-based audio processing (recommended)
            - "mfcc_full_control": MFCC-based with full control parameters
            - "hubert_full_control": HuBERT with full control (highest quality)
    
    Returns:
        A tuple containing:
            - Video object for 256x256 resolution (or None if failed)
            - Video object for 512x512 resolution (or None if disabled/failed)  
            - Status message describing the result
    
    Raises:
        FileNotFoundError: If image_path or audio_path doesn't exist.
        ValueError: If infer_type is not supported.
        RuntimeError: If video generation fails due to system issues.
    
    Example:
        >>> result = generate_video(
        ...     image_path="portrait.jpg",
        ...     audio_path="speech.wav",
        ...     infer_type="hubert_audio_only"
        ... )
        >>> video_256, video_512, message = result
        >>> print(message)  # "Video generated successfully!"
    """
    pass
```

#### Error Handling

```python
# Use specific exceptions
class VisualizrError(Exception):
    """Base exception for Visualizr."""
    pass

class ModelLoadError(VisualizrError):
    """Raised when model loading fails."""
    pass

class GenerationError(VisualizrError):
    """Raised when video generation fails."""
    pass

# Proper error handling
def load_model(model_path: Path) -> Model:
    """Load model with proper error handling."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = torch.load(model_path)
        return model
        
    except torch.serialization.pickle.UnpicklingError as e:
        raise ModelLoadError(f"Failed to load model from {model_path}: {e}") from e
    except Exception as e:
        logger.exception("Unexpected error loading model")
        raise ModelLoadError(f"Unexpected error: {e}") from e
```

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import torch
import numpy as np
from gradio import Blocks, Button, Video

# Local imports
from visualizr.settings import Settings, ModelSettings
from visualizr.anitalker.liamodel import LiaModel
from visualizr.utils import check_gpu_availability
```

### Configuration Management

```python
# Use Pydantic for configuration
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration with validation."""

    device: str = Field(default="cuda", description="Compute device")
    step_t: int = Field(default=50, ge=1, le=200, description="Generation steps")
    seed: int = Field(default=0, ge=0, description="Random seed")

    @validator("device")
    def validate_device(cls, v):
        if v not in ["cuda", "cpu"]:
            raise ValueError("Device must be 'cuda' or 'cpu'")
        return v
```

## Testing

We maintain high test coverage to ensure reliability.

### Testing Framework

We use **pytest** with additional plugins:

```bash
# Install test dependencies (included in dev dependencies)
pip install pytest pytest-cov pytest-mock pytest-asyncio

# Run all tests
pytest

# Run with coverage
pytest --cov=src/visualizr --cov-report=html

# Run specific test file
pytest tests/test_app.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_generation"
```

### Writing Tests

#### Unit Tests

```python
# tests/test_settings.py
import pytest
from visualizr.settings import Settings, ModelSettings


def test_default_settings():
    """Test default settings initialization."""
    settings = Settings()
    assert settings.model.device in ["cuda", "cpu"]
    assert settings.model.step_t == 50
    assert settings.directory.base.exists()


def test_custom_model_settings():
    """Test custom model configuration."""
    model_config = ModelSettings(
        device="cpu",
        step_t=100,
        seed=42
    )
    settings = Settings(model=model_config)

    assert settings.model.device == "cpu"
    assert settings.model.step_t == 100
    assert settings.model.seed == 42


def test_invalid_device():
    """Test validation of invalid device."""
    with pytest.raises(ValueError, match="Device must be"):
        ModelSettings(device="invalid")
```

#### Integration Tests

```python
# tests/test_generation.py
import pytest
from pathlib import Path
from visualizr.app.runner import app


@pytest.fixture
def sample_files():
    """Provide sample image and audio files."""
    return {
        "image": Path("tests/fixtures/sample_portrait.jpg"),
        "audio": Path("tests/fixtures/sample_audio.wav")
    }


def test_basic_generation(sample_files):
    """Test basic video generation functionality."""
    result = app.generate_video(
        infer_type="mfcc_full_control",  # Fastest for testing
        image_path=str(sample_files["image"]),
        audio_path=str(sample_files["audio"]),
        face_sr=False,
        step_t=10,  # Minimal steps for speed
        # ... other parameters
    )

    video_256, video_512, message = result

    assert video_256 is not None
    assert "successfully" in message.value.lower()
    assert Path(video_256.value).exists()


@pytest.mark.slow
def test_quality_generation(sample_files):
    """Test high-quality generation (marked as slow)."""
    result = app.generate_video(
        infer_type="hubert_full_control",
        image_path=str(sample_files["image"]),
        audio_path=str(sample_files["audio"]),
        face_sr=True,
        step_t=50,
        # ... other parameters
    )

    video_256, video_512, message = result

    assert video_256 is not None
    assert video_512 is not None  # Super-resolution enabled
```

#### Mock Tests

```python
# tests/test_model_loading.py
import pytest
from unittest.mock import Mock, patch
from visualizr.app.builder import App


@patch('visualizr.app.builder.snapshot_download')
@patch('visualizr.anitalker.liamodel.LiaModel.load_lightning_model')
def test_model_loading_failure(mock_load_model, mock_download):
    """Test handling of model loading failures."""

    # Mock download failure
    mock_download.side_effect = ConnectionError("Network unavailable")

    with pytest.raises(ConnectionError):
        app = App(settings)


@patch('torch.cuda.is_available')
def test_cpu_fallback(mock_cuda_available):
    """Test CPU fallback when CUDA unavailable."""
    mock_cuda_available.return_value = False

    settings = Settings()
    assert settings.model.device == "cpu"
```

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Test data files
│   ├── sample_portrait.jpg
│   └── sample_audio.wav
├── unit/                    # Unit tests
│   ├── test_settings.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/             # Integration tests
│   ├── test_generation.py
│   └── test_api.py
└── performance/             # Performance tests
    └── test_benchmarks.py
```

### Test Configuration

```python
# conftest.py
import pytest
import tempfile
from pathlib import Path
from visualizr.settings import Settings, ModelSettings, DirectorySettings


@pytest.fixture(scope="session")
def test_settings():
    """Provide test-specific settings."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        yield Settings(
            model=ModelSettings(
                device="cpu",  # Always use CPU for tests
                step_t=10,  # Fast tests
                seed=42  # Reproducible
            ),
            directory=DirectorySettings(
                base=tmp_path,
                results=tmp_path / "results",
                assets=tmp_path / "assets",
                checkpoint=tmp_path / "ckpts"
            )
        )


@pytest.fixture
def mock_gpu():
    """Mock GPU availability for testing."""
    with patch('torch.cuda.is_available', return_value=True):
        yield


# Skip GPU tests if CUDA unavailable
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
```

## Contributing Workflow

### 1. Planning Your Contribution

Before starting work:

1. **Check existing issues** - Avoid duplicate work
2. **Create or comment on an issue** - Discuss your approach
3. **Get feedback** - Ensure alignment with project goals
4. **Break down large features** - Into manageable pieces

### 2. Development Process

```bash
# 1. Update your fork
git checkout main
git pull upstream main
git push origin main

# 2. Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description

# 3. Make your changes
# ... code, test, commit ...

# 4. Keep branch updated
git fetch upstream
git rebase upstream/main

# 5. Push to your fork
git push origin feature/your-feature-name
```

### 3. Commit Guidelines

Follow **Conventional Commits** format:

```bash
# Format: <type>(<scope>): <description>
# Types: feat, fix, docs, style, refactor, test, chore

# Examples:
git commit -m "feat(models): add support for custom inference types"
git commit -m "fix(api): handle timeout errors in generation endpoint"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(generation): add tests for edge cases"
git commit -m "refactor(settings): simplify configuration validation"

# For breaking changes:
git commit -m "feat(api)!: change response format for better error handling

BREAKING CHANGE: API response format changed from {status, data} to {success, result, error}"
```

### 4. Code Review Process

1. **Self-review** your changes
2. **Run all tests** and quality checks
3. **Update documentation** if needed
4. **Create pull request** with detailed description
5. **Address feedback** from reviewers
6. **Squash commits** if requested before merge

## Pull Request Guidelines

### PR Checklist

Before submitting a pull request, ensure:

- [ ] **Code follows style guidelines** (runs `ruff check` and `ruff format`)
- [ ] **Type hints are present** for public APIs
- [ ] **Tests are added/updated** for new functionality
- [ ] **All tests pass** locally (`pytest`)
- [ ] **Documentation is updated** if needed
- [ ] **Commit messages follow conventions**
- [ ] **PR description is clear and complete**

### PR Template

```markdown
## Description

Brief description of the changes and why they're needed.

Fixes #<issue_number>

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not
  work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup/refactoring

## How Has This Been Tested?

Describe the tests that you ran to verify your changes:

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed
- [ ] Tested on multiple platforms/configurations

## Screenshots (if applicable)

For UI changes, include before/after screenshots.

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

### Review Process

1. **Automated checks** run on every PR
2. **Maintainer review** for code quality and design
3. **Community feedback** welcome on public PRs
4. **Address feedback** through discussion or code changes
5. **Final approval** and merge by maintainers

## Issue Guidelines

### Creating Issues

#### Bug Reports

Use the bug report template:

```markdown
## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened.

## Screenshots

If applicable, add screenshots.

## Environment

- OS: [e.g. iOS]
- Python version: [e.g. 3.10.5]
- Visualizr version: [e.g. 0.0.5]
- GPU: [e.g. RTX 3080]

## Additional Context

Any other context about the problem.
```

#### Feature Requests

```markdown
## Feature Summary

Brief description of the feature you'd like to see.

## Motivation

Why would this feature be useful? What problem does it solve?

## Detailed Description

Detailed explanation of the feature and how it should work.

## Possible Implementation

If you have ideas about how this could be implemented.

## Alternatives Considered

Other solutions you've considered.

## Additional Context

Any other context or screenshots about the feature request.
```

### Issue Labels

We use labels to categorize issues:

- **Type**: `bug`, `feature`, `documentation`, `question`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Difficulty**: `good first issue`, `help wanted`, `complex`
- **Area**: `models`, `api`, `ui`, `deployment`, `testing`
- **Status**: `needs-reproduction`, `blocked`, `wontfix`

## Community Guidelines

### Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all
contributors.

#### Our Standards

**Positive behavior includes:**

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**

- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate

#### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by
contacting the project team. All complaints will be reviewed and investigated promptly
and fairly.

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community chat
- **Pull Requests**: Code review and technical discussion
- **Email**: security@alphasphere.ai for security issues

### Recognition

We appreciate all contributions! Contributors will be:

- **Listed in CONTRIBUTORS.md** for significant contributions
- **Mentioned in release notes** for their contributions
- **Invited to join** the contributor team for ongoing contributors
- **Given credit** in academic papers citing this work

### Getting Help

If you need help with contributing:

1. **Check existing documentation** and issues first
2. **Ask in GitHub Discussions** for general questions
3. **Create an issue** for specific problems
4. **Join community events** and office hours (announced in discussions)

### Mentorship

We provide mentorship for new contributors:

- **Good first issues** are labeled and documented
- **Detailed contributing guides** walk you through the process
- **Code review feedback** helps you learn best practices
- **Pairing sessions** available for complex contributions

---

Thank you for contributing to Visualizr! Your efforts help make AI-powered video
generation accessible to everyone. 🚀

*Questions about contributing? Open an issue or discussion - we're here to help!*
