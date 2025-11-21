# Ultralytics Project Context for AI Agents

## Project Overview

Ultralytics is a state-of-the-art computer vision framework focused on YOLO (You Only Look Once) models. The project provides fast, accurate, and easy-to-use implementations for object detection, tracking, instance segmentation, image classification, pose estimation, and oriented bounding box tasks.

### Key Technologies

- **Language**: Python (>=3.8)
- **Core Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Package Management**: setuptools with pyproject.toml
- **Code Quality**: Ruff for linting and formatting
- **Testing**: pytest with coverage
- **Documentation**: MkDocs with Material theme

### Architecture

The codebase is organized into several key modules:

- `ultralytics/models/`: YOLO model implementations (YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR)
- `ultralytics/nn/`: Neural network components and architectures
- `ultralytics/engine/`: Training, validation, and prediction engines
- `ultralytics/utils/`: Utility functions and helpers
- `ultralytics/data/`: Data loading and processing
- `ultralytics/cfg/`: Configuration management
- `ultralytics/hub/`: Integration with Ultralytics HUB
- `ultralytics/solutions/`: End-to-end solutions
- `ultralytics/trackers/`: Object tracking implementations

## Development Workflow

### Installation

```bash
# For users
pip install ultralytics

# For developers (editable mode)
pip install -e .

# For development with optional dependencies
pip install -e ".[dev]"
```

### Testing

The project uses pytest for testing. Test files are located in the `tests/` directory.

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=ultralytics

# Run specific test file
pytest tests/test_engine.py
```

### Code Quality

The project uses Ruff for both linting and formatting:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix linting issues
ruff check . --fix
```

### Documentation

Documentation is built with MkDocs and Material theme:

```bash
# Build documentation locally
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Development Conventions

### Code Style

- Follow PEP 8 with Ruff formatting (line length: 120)
- Use Google-style docstrings for all public functions and classes
- Include type hints where possible
- Use `isort` for import sorting (configured through Ruff)

### Commit Convention

- Use clear, descriptive commit messages
- Reference issue numbers when applicable (e.g., `Fix #123: Corrected calculation error`)
- Include appropriate tests for new features

### Model Development

- New model implementations should follow the existing pattern in `ultralytics/models/`
- Each model should have its own directory with necessary files
- Include pretrained model configurations in YAML format

### Testing Requirements

- All new features must include tests
- Tests should cover edge cases and error conditions
- Use conftest.py for shared fixtures and configuration

## Key Files and Directories

### Configuration

- `pyproject.toml`: Main project configuration, dependencies, and tool settings
- `ultralytics/cfg/`: Default configurations and settings
- `ultralytics/data/`: Dataset configurations and data loaders

### Model Implementations

- `ultralytics/models/yolo/`: YOLO model family implementations
- `ultralytics/models/sam/`: Segment Anything Model implementation
- `ultralytics/models/rtdetr/`: RT-DETR implementation

### Utilities

- `ultralytics/utils/`: Core utility functions
- `ultralytics/nn/`: Neural network building blocks
- `ultralytics/assets/`: Static assets and example images

## Common Development Tasks

### Adding a New Model

1. Create a new directory under `ultralytics/models/`
2. Implement the model class following existing patterns
3. Add configuration files in YAML format
4. Write tests in the `tests/` directory
5. Update documentation

### Adding Export Formats

1. Implement export logic in the model's export method
2. Add format-specific dependencies to `pyproject.toml` under the `export` optional dependency group
3. Write tests for the new export format
4. Update documentation

### Bug Fixes

1. Create a new branch from `main`
2. Write a test that reproduces the bug
3. Fix the issue
4. Ensure all tests pass
5. Submit a pull request with description and issue reference

## Integration Points

### Ultralytics HUB

- API integration is handled in `ultralytics/hub/`
- Authentication and model management functions
- Dataset and experiment tracking

### Third-party Integrations

- Weights & Biases: Experiment tracking
- Comet ML: Model management and visualization
- Roboflow: Dataset management
- Intel OpenVINO: Model optimization

## Performance Considerations

- Models are optimized for both CPU and GPU inference
- Consider memory usage when implementing new features
- Use vectorized operations where possible (NumPy, PyTorch)
- Profile code changes for performance impact

## Security Notes

- Never commit model weights or sensitive data
- Use environment variables for API keys and secrets
- Follow secure coding practices when handling file uploads/user input

## License

This project is licensed under AGPL-3.0. Any derivative works must also be licensed under AGPL-3.0 and their source code must be made publicly available.
