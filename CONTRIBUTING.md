# Contributing

## Setup

```bash
uv sync --all-groups
```

## Required Checks

```bash
uv run ruff format --check src/robot2mjcf tests
uv run ruff check src/robot2mjcf tests
uv run mypy src/robot2mjcf tests
uv run pytest
uv build
```

## Workflow

- work on a feature branch, not `main`
- use Conventional Commits
- keep changes focused
- add tests for behavior changes
- prefer extracting helpers over growing large orchestrators inline

## Refactor Safety

When changing converter behavior:

- run the real example regressions in [`tests/test_convert.py`](/Users/jiayufei/ws/robot2mjcf/tests/test_convert.py)
- compare semantic output, not only raw XML text
