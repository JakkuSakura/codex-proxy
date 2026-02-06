# Contributing to codex-proxy

Thank you for your interest in contributing to codex-proxy!

## Setting Up Development Environment

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cornellsh/codex-proxy.git
   cd codex-proxy
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest ruff mypy
   ```

4. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, readable code following existing conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Run linting and tests:**
   ```bash
   ruff check src/ tests/
   mypy src/ --ignore-missing-imports
   pytest tests/ -v
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

5. **Push and create a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for new functions
- Keep functions focused and small
- Add docstrings for public APIs
- Use descriptive variable names

## Testing

- Write unit tests for new features
- Mock external API calls in tests
- Aim for good test coverage
- All tests must pass before submitting PR

## Documentation

- Update README.md if changing user-facing behavior
- Update CHANGELOG.md for breaking changes
- Keep AGENTS.md in sync with implementation

## Issues and Feature Requests

- Check existing issues before creating new ones
- Use issue templates when available
- Provide clear reproduction steps for bugs
- Describe use cases for feature requests

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
