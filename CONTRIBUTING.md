# Contributing to AI Video Analytics System

Thank you for your interest in contributing to the AI Video Analytics System! This document provides guidelines and instructions for contributing.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Process](#development-process)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professionalism

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- Docker (optional)
- NVIDIA GPU (optional, for GPU features)

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ai-video-analytics-system.git
cd ai-video-analytics-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Copy example configuration
cp .env.example .env
```

## Development Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality

3. **Test your changes**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Check coverage
   pytest tests/ --cov=src --cov-report=html
   
   # Lint code
   flake8 src/ tests/
   
   # Format code
   black src/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Coding Standards

### Python Style Guide
- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all public functions/classes

### Example
```python
def process_frame(frame: np.ndarray, threshold: float = 0.5) -> List[DetectionResult]:
    """
    Process a video frame and return detections.
    
    Args:
        frame: Input video frame in BGR format
        threshold: Confidence threshold for detections (default: 0.5)
        
    Returns:
        List of detection results
        
    Raises:
        ValueError: If frame is invalid
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid frame")
    
    # Implementation
    pass
```

### Naming Conventions
- Classes: `PascalCase`
- Functions/Methods: `snake_case`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### Documentation
- Add docstrings to all public APIs
- Update README.md for user-facing changes
- Update DEPLOYMENT.md for deployment-related changes
- Add comments for complex logic

## Testing

### Writing Tests

```python
import pytest
from src.module import YourClass

class TestYourClass:
    """Tests for YourClass."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return YourClass()
    
    def test_method_name(self, instance):
        """Test method description."""
        result = instance.method()
        assert result == expected_value
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_module.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Coverage Goals
- Minimum 80% coverage for new code
- 100% coverage for critical paths
- Test edge cases and error conditions

## Pull Request Process

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] No merge conflicts with main branch

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] Test coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process
1. Automated tests run on PR
2. Code review by maintainer
3. Address feedback
4. Approval and merge

## Feature Requests and Bug Reports

### Bug Reports
Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, GPU, etc.)
- Logs and error messages

### Feature Requests
Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (optional)
- Potential impact on existing functionality

## Development Tips

### Performance Testing
```python
import time

start = time.time()
# Your code here
elapsed = time.time() - start
print(f"Execution time: {elapsed:.2f}s")
```

### Debugging
```python
# Use loguru for logging
from loguru import logger

logger.debug("Debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
```

### Memory Profiling
```bash
pip install memory_profiler
python -m memory_profiler src/main.py
```

## Release Process

1. Update version in `src/__init__.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and test Docker images
5. Publish release notes

## Questions?

- Open an issue for questions
- Join discussions in GitHub Discussions
- Check existing issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! 🎉
