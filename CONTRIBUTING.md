# Contributing to Socrates Nexus

First, thank you for your interest in contributing! We welcome all contributions.

## Getting Started

### Prerequisites
- Python 3.8+
- pip or poetry
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nireus79/socrates-nexus.git
   cd socrates-nexus
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev,all]"
   ```

4. **Verify installation**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for public functions
- Keep changes focused and minimal

### 3. Add Tests

Every feature should have corresponding tests:
```bash
# Run tests locally
pytest tests/ -v

# With coverage
pytest tests/ --cov=socrates_nexus
```

### 4. Format Your Code

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/socrates_nexus
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "Brief description of changes"
```

Use clear, descriptive commit messages. Reference issues when applicable: `Fix #123`

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference to any related issues
- Test results (run locally first)

## Code Style

### Python Style
- Use black for formatting: `black src/ tests/`
- Use ruff for linting: `ruff check src/ tests/`
- Type hints on all functions
- Docstrings for public APIs

### Example
```python
from typing import Optional

def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: dict,
) -> float:
    """
    Calculate API call cost based on tokens.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        pricing: Pricing dict with 'input' and 'output' keys

    Returns:
        Cost in USD

    Raises:
        ValueError: If tokens or pricing is invalid
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Tokens cannot be negative")

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
```

## Testing Guidelines

### Unit Tests
- Location: `tests/test_*.py`
- Should not require API keys
- Should be fast (<1 second)
- Use `@pytest.mark.unit` for explicit marking

### Integration Tests
- Location: `tests/test_integration.py`
- Marked with `@pytest.mark.requires_api`
- Run with: `pytest tests/test_integration.py --requires_api`
- Require valid API keys

### Test Structure
```python
def test_feature_behavior():
    """Clear description of what is being tested."""
    # Arrange
    client = LLMClient(provider="anthropic", ...)

    # Act
    response = client.chat("Hello")

    # Assert
    assert response.content is not None
```

## Adding a New Provider

1. **Create provider file**: `src/socrates_nexus/providers/new_provider.py`
2. **Implement BaseProvider interface**
3. **Add pricing data** to `PROVIDER_PRICING` in `models.py`
4. **Add provider to factory** in `client.py` and `async_client.py`
5. **Create tests** in `tests/test_new_provider.py`
6. **Update documentation** with setup instructions

## Documentation

- Update README.md for major features
- Add docstrings to all public functions
- Update CHANGELOG.md with changes
- Update docs/ for API changes

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a commit: `git commit -m "Bump version to X.Y.Z"`
4. Tag it: `git tag vX.Y.Z`
5. Push: `git push origin main && git push --tags`
6. GitHub Actions will automatically publish to PyPI

## Getting Help

- **Questions**: Use GitHub Discussions
- **Bugs**: Create an issue with reproduction steps
- **Ideas**: Start a discussion before implementing
- **Docs**: Improvements always welcome!

## Code of Conduct

- Be respectful and inclusive
- Constructive feedback only
- No harassment or discrimination
- Report violations to maintainers

## Acknowledgments

Contributors are listed in CHANGELOG.md and on GitHub.

Thank you for contributing to Socrates Nexus! 🚀
