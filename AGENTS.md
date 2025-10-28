# AGENTS.md for Visualizr

## Memory

You are given two tools from Byterover MCP server, including:
1. `byterover-store-knowledge`
You `MUST` always use this tool when:

- Learning new patterns, APIs, or architectural decisions from the codebase
- Encountering error solutions or debugging techniques
- Finding reusable code patterns or utility functions
- Completing any significant task or plan implementation

2. `byterover-retrieve-knowledge`
You `MUST` always use this tool when:

- Starting any new task or implementation to gather relevant context
- Before making architectural decisions to understand existing patterns
- When debugging issues to check for previous solutions
- Working with unfamiliar parts of the codebase

## Build/Lint/Test Commands

- Build: `uv build`
- Lint/Format: `ruff check` and `ruff format` and `trunk check`
- Test: `pytest tests/test_app.py` (single test)
- Type check: `mypy src/`

## Code Style Guidelines

- **Imports**: Use isort with `from pyproject.toml` config
- **Formatting**: Black with line-length 88
- **Types**: Add type hints to all public APIs
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Errors**: Use specific exceptions, log errors with `logging` module
- **Ruff**: Follow `.trunk/ruff.toml` rules (no unused imports, no F841)

## Special Rules

- No `print()` statements in production code
- All public functions must have Google-style docstrings
- Tests must achieve >90% branch coverage
- Use `pathlib` for file paths instead of `os.path`
