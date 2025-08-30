# AGENTS.md for Visualizr

## Build/Lint/Test Commands

- Build: `uv build`
- Lint: `ruff check .` and `ruff fmt .` and `trunk check`
- Test: `pytest tests/test_app.py` (single test)
- Type check: `mypy src/`

## Code Style Guidelines

- **Imports**: Use isort with `from pyproject.toml` config
- **Formatting**: Black with line-length 88
- **Types**: Add type hints to all public APIs
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Errors**: Use specific exceptions, log errors with `logging` module
- **Ruff**: Follow .trunk/ruff.toml rules (no unused imports, no F841)

## Special Rules

- No `print()` statements in production code
- All public functions must have Google-style docstrings
- Tests must achieve >90% branch coverage
- Use `pathlib` for file paths instead of `os.path`
