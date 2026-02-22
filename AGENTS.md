# Agent Context & Handoff Guide

## Project Overview
**Repository:** genetics-23andMe  
**Owner:** mikeoc61  
**Description:** 23andMe genome analysis and fitness data visualization tools

## Current State

### Project Structure
Status: **Needs refactoring** to standard Python project layout

Current files:
- `foundMyFitness.py` - Fitness data analysis
- `genome_gui.py` - GUI for genome data
- `genome.py` - Core genome analysis
- `README.md` - Project documentation
- `LICENSE` - MIT License

Desired structure:
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

# Running
python -m genetics_23andme.gui     # Launch GUI
```

## Recent Work Log

See [docs/plan.md](docs/plan.md) for:
- Implementation plan & progress
- Architecture Decision Records (docs/adr/)
- Session handoff notes

## For Next Agent

1. **Before starting:** Read [docs/plan.md](docs/plan.md) for current context
2. **Check status:** Review last session's completed items
3. **Update log:** Add session summary before handing off
4. **Key constraints:** 
   - Use `ruff` for all linting/formatting
   - Use `uv` for dependency management
   - Maintain this handoff format for continuity

---
*Last updated: Session [date] - [brief summary]*
