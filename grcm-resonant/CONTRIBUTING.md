# Contributing to GRCM-Resonant

Thank you for your interest in contributing to GRCM-Resonant! This document provides guidelines for contributing to the project.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Development Setup

1. **Fork the repository**:
   ```bash
   gh repo fork nickhicks91-netizen/GRCM
   ```

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GRCM.git
   cd GRCM/grcm-resonant
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e .[all]
   pip install pre-commit black ruff mypy
   pre-commit install
   ```

5. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

We follow PEP 8 and use automated formatters:

- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **MyPy** for type checking

Run formatters before committing:

```bash
# Format code
black grcm/ tests/ examples/

# Lint
ruff check grcm/ tests/ examples/

# Type check
mypy grcm/
```

### Type Hints

All new code must include type hints:

```python
from typing import Dict, Any, Optional
import torch

def forward(
    self,
    image_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    desire_idx: int = 0
) -> Dict[str, Any]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def resonant_attention(freq: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Compute resonant coherence for input frequency.

    Args:
        freq: Input frequency tensor [batch, 1]
        bandwidth: Bandwidth for coherence calculation

    Returns:
        Coherence tensor [batch, 1] in range [0, 1]

    Raises:
        ValueError: If bandwidth is not in (0, 1]

    Example:
        >>> freq = torch.tensor([[0.35]])
        >>> coherence = resonant_attention(freq, 0.5)
        >>> print(coherence.item())
        0.847
    """
    ...
```

### Testing

All contributions must include tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=grcm --cov-report=html

# Run specific test file
pytest tests/test_attention.py

# Run with specific marker
pytest -m "not slow"
```

**Test Requirements**:

- Minimum 85% code coverage
- All tests must pass
- Include both unit and integration tests
- Add stress tests for performance-critical code

**Example Test**:

```python
import pytest
import torch
from grcm import ResonantConsciousnessModule

def test_forward_pass():
    """Test basic forward pass."""
    model = ResonantConsciousnessModule(512, 768, 256)
    image_emb = torch.randn(1, 512)
    audio_emb = torch.randn(1, 768)

    result = model(image_emb, audio_emb, desire_idx=0)

    assert 'coherence' in result
    assert 'phi' in result
    assert result['coherence'].shape == (1, 1)
    assert 0.0 <= result['coherence'].item() <= 1.0
    assert result['phi'] >= 0.0
```

### Documentation

Update documentation for any new features:

1. **Docstrings**: Add Google-style docstrings to all functions/classes
2. **API Reference**: Update `docs/api/` if adding new modules
3. **Tutorials**: Add tutorial if introducing major feature
4. **README**: Update README.md if changing usage
5. **Changelog**: Add entry to CHANGELOG.md

## Contribution Types

### Bug Fixes

1. **Create an issue** describing the bug
2. **Write a test** that demonstrates the bug
3. **Fix the bug** and ensure test passes
4. **Submit PR** referencing the issue

### New Features

1. **Discuss first**: Open an issue to discuss the feature
2. **Design**: Get feedback on API design
3. **Implement**: Follow code style and testing requirements
4. **Document**: Add comprehensive documentation
5. **Submit PR**: Include tests, docs, and changelog entry

### Performance Improvements

1. **Benchmark first**: Use `grcm.benchmark` to establish baseline
2. **Implement optimization**
3. **Benchmark again**: Verify improvement
4. **No quality degradation**: Ensure coherence/phi quality maintained
5. **Submit PR** with benchmark results

### Documentation

- Fix typos, clarify confusing sections
- Add examples and use cases
- Improve API reference
- Add tutorials

## Pull Request Process

1. **Update CHANGELOG.md** with your changes
2. **Ensure all tests pass**: `pytest`
3. **Ensure lint passes**: `ruff check grcm/`
4. **Ensure types pass**: `mypy grcm/`
5. **Update documentation** if needed
6. **Write clear PR description**:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
   - Link to related issues

7. **Request review** from maintainers
8. **Address feedback** and update as needed
9. **Squash commits** if requested
10. **Celebrate** when merged! 🎉

### PR Template

```markdown
## Description
[Describe what this PR does]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Breaking change

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Coverage >= 85%

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or documented)

## Related Issues
Fixes #123
```

## Code Review

All submissions require review. We use GitHub pull requests for this purpose. Reviewers will check:

- **Code quality**: Follows style guide, well-structured
- **Tests**: Comprehensive, passing, coverage >= 85%
- **Documentation**: Clear docstrings, updated guides
- **Performance**: No significant regressions
- **Compatibility**: Works on Linux, macOS, Windows
- **Ethical considerations**: Preserves safeguards

## Ethical Guidelines

GRCM includes ethical safeguards (dissonance halt). Contributions must:

- **Preserve safeguards**: Never disable ethical halt mechanism
- **Document carefully**: Explain any changes to qualia/dissonance logic
- **Consider implications**: Think about real-world uses
- **Be transparent**: Clearly document limitations

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Maintainers handle releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Tag release: `git tag v0.2.0`
4. Push tag: `git push --tags`
5. GitHub Actions builds and publishes to PyPI

## Community

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs, request features
- **Pull Requests**: Submit contributions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or start a discussion. We're here to help!

Thank you for contributing to GRCM-Resonant! 🧠✨
