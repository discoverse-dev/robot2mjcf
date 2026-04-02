# Project Instructions

## Execution Commands

- ALWAYS use `uv run <command>` to execute Python code
- NEVER use `python` or `python3` directly

## Development Workflow

- Install dependencies: `uv sync --all-groups`
- Run tests: `uv run pytest`
- Lint: `uv run ruff check src/robot2mjcf tests`
- Format: `uv run ruff format src/robot2mjcf tests`
- Type check: `uv run mypy src/robot2mjcf tests`
- Local build: `uv build`

## Project Structure

- `src/robot2mjcf/` — Main package source (import name remains `robot2mjcf`)
- `tests/` — pytest suite
- `.github/workflows/` — CI (matrix testing + lint + mypy)
- `pyproject.toml` — Single source of truth for build, dependencies, and tool configs

## Packaging & Release

- PyPI distribution name: `robot2mjcf`
- In-code import name: `robot2mjcf` (kept for backward compatibility)
- Releases are published manually; pushing a tag does not trigger package publishing

## Dependency Management

- All dependencies (including dev) are maintained in `pyproject.toml`
- Production deps: `[project] dependencies`
- Dev deps: `[dependency-groups] dev`
- There is no `requirements.txt`; do not create one

## Code Quality Standards

When writing or modifying code, you MUST:

1. **Minimize code**: Write the absolute minimum code needed. No extra features, no "nice-to-haves"
2. **Delete ruthlessly**: Remove unused code immediately. Never comment out code—delete it
3. **Refactor immediately**: If you spot poor design while working, fix it before adding features
4. **Prioritize clarity**: Code must be self-explanatory. Clarity > brevity > flexibility
5. **Optimize for performance**: This is GPU-intensive research code. Slow code wastes expensive compute time
6. **Test coverage**: Any change must pass `uv run pytest`. Add tests for new features

## Design Priorities (in order)

1. Clear, understandable code structure
2. Minimal, focused implementation
3. Good architecture over quick functionality
4. Performance optimization
5. Testability over debuggability

## What NOT to do

- Do NOT add defensive code for impossible scenarios
- Do NOT create abstractions for single-use cases
- Do NOT add features beyond what's explicitly requested
- Do NOT leave commented-out code
- Do NOT add unnecessary error handling or validation in internal code

## Git

Conventional Commits: `feat:` / `fix:` / `docs:` / `style:` / `refactor:` / `test:` / `chore:`

- Commit often, keep history clean
- Never commit to main — always work on feature branches
- Delete feature branches after PR merge
