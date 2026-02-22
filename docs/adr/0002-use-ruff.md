# ADR 0002: Use ruff for linting and formatting

## Status
Accepted - 2026-02-22

## Context
We need a single tool to enforce style, detect common issues, and keep the
codebase consistent with minimal configuration and fast feedback.

## Decision
Use `ruff` for linting and formatting. Configure it in `pyproject.toml` and
run it via `uv run ruff check` and `uv run ruff format`.

## Consequences
- Linting and formatting are fast and consistent.
- Contributors should run `ruff check --fix` and `ruff format` before commits.
- Some legacy patterns may need refactoring to satisfy modern lint rules.
