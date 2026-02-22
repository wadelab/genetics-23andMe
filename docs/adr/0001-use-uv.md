# ADR 0001: Use uv for environments and dependencies

## Status
Accepted - 2026-02-22

## Context
We need a consistent, fast, and reproducible way to manage Python versions,
virtual environments, and dependency installation for this repository.

## Decision
Use `uv` to create a local `.venv` per repository and to install and run
project dependencies (`uv sync`, `uv run ...`).

## Consequences
- Setup is fast and consistent across environments.
- The project expects a `.venv` managed by `uv` in the repo root.
- Developers should use `uv run` for executing tools and scripts.
