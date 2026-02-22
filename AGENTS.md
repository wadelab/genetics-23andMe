# Agent Context & Handoff Guide

## Project Overview
**Repository:** genetics-23andMe  
**Owner:** mikeoc61  
**Description:** 23andMe genome analysis and fitness data visualization tools

## Current State

### Project Structure
Status: **✅ COMPLETE** - Refactored to standard Python project layout

Legacy files (archived in `legacy/`):
- `foundMyFitness.py` → `src/genetics_23andme/fitness.py`
- `genome_gui.py` → `src/genetics_23andme/gui.py`
- `genome.py` → `src/genetics_23andme/core.py`

Current structure:
```
.
├── pyproject.toml
├── uv.lock
├── AGENTS.md
├── SAMPLE_AGENTS.md
├── README.md
│
├── src/genetics_23andme/
│   ├── __init__.py
│   ├── core.py
│   ├── gui.py
│   └── fitness.py
│
├── tests/
│   └── test_core.py
│
└── docs/
    ├── plan.md
    └── adr/
```

## Development Setup

### Tools & Conventions
- **Package Manager:** `uv` (manages venv & dependencies)
- **Linter/Formatter:** `ruff` (configured in `pyproject.toml`)
- **Virtual Environment:** `.venv` (managed by `uv`)
- **Python Version:** 3.10+ (recommended)

### Key Commands
```bash
# Setup
uv sync                    # Create .venv and install dependencies

# Development
ruff check .              # Run linter
ruff format .             # Format code
ruff check --fix .        # Fix auto-fixable issues
uv run pytest tests/       # Run tests

# Running
uv run python -m genetics_23andme.gui     # Launch GUI (interactive)
uv run python -m genetics_23andme.core     # Run CLI (batch analysis)
```

## Recent Work Log

See [docs/plan.md](docs/plan.md) for:
- Implementation plan & progress tracking
- Architecture Decision Records (docs/adr/)
- Detailed session logs with timestamps

**Latest Session**: GUI analysis now fully functional with thread-safe UI updates
- ✅ Fixed threading issue: All UI updates now wrapped in self.after() for main thread safety
- ✅ Progress bar and status text now properly update during analysis
- ✅ 5-stage progress animation visible (📂→📊→🔍→🌐→🎨)
- ✅ Analysis results render correctly without UI freezing
- ✅ All core functions tested with real 23andMe genome file (643K SNPs)
- ✅ Committed: ac5fbb2 - Fix GUI threading for main thread safety

## For Next Agent

1. **Before starting:** Read [docs/plan.md](docs/plan.md) for session history and architecture
2. **Check status:** Review completed items and blockers in plan.md
3. **Test the code:** `uv run python test_run.py` for quick functional check
4. **Update log:** Add your session summary to docs/plan.md before handing off
5. **Key constraints:** 
   - Use `ruff` for all linting/formatting (line length 100)
   - Use `uv` for all Python dependency management
   - Maintain this 3-section handoff format for continuity
   - GUI and CLI are the two main entrypoints

---
*Last updated: Session 2 (2026-02-22) - Threading fix + GUI fully working*
