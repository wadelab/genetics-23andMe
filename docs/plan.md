# Project Plan & Session Log

## Project Goals
- [ ] Refactor codebase to standard Python project structure
- [ ] Set up `uv` for dependency management
- [ ] Configure `ruff` for linting and formatting
- [ ] Create comprehensive test suite
- [ ] Improve documentation and type hints

## Architecture

### Module Organization
- **src/genetics_23andme/core.py** - Core genome analysis logic
- **src/genetics_23andme/gui.py** - GUI components (from genome_gui.py)
- **src/genetics_23andme/fitness.py** - Fitness data analysis (from foundMyFitness.py)

### Key Dependencies (to be defined)
- TBD: Genome analysis libraries
- TBD: GUI framework (tkinter, PyQt, etc.)
- TBD: Data processing (pandas, etc.)

## Implementation Phases

### Phase 1: Project Setup
- [ ] Create pyproject.toml with dependencies and tool configs
- [ ] Initialize uv environment (`uv sync`)
- [ ] Configure ruff (linting & formatting rules)
- [ ] Create directory structure (src/, tests/, docs/)

**Status:** In Progress
**Dependencies:** None

### Phase 2: Code Refactoring
- [ ] Move genome.py → src/genetics_23andme/core.py
- [ ] Move genome_gui.py → src/genetics_23andme/gui.py
- [ ] Move foundMyFitness.py → src/genetics_23andme/fitness.py
- [ ] Update imports and ensure code runs
- [ ] Run ruff check and apply fixes

**Status:** Blocked (waiting for Phase 1)
**Dependencies:** Phase 1 complete

### Phase 3: Testing & Documentation
- [ ] Create tests/ directory and test files
- [ ] Write unit tests for core modules
- [ ] Add type hints to functions
- [ ] Update README with setup instructions

**Status:** Blocked (waiting for Phase 2)
**Dependencies:** Phase 2 complete

---

## Session Log

### Session 1 - 2026-02-22
**Completed:**
- ✅ Created AGENTS.md for agent handoff documentation
- ✅ Created docs/ directory structure with plan.md and adr/ subdirectory
- ✅ Established handoff protocol for continuity between sessions
- ✅ Created pyproject.toml with uv and ruff configuration
- ✅ Refactored code to src/genetics_23andme/ structure:
  - fitness.py (from foundMyFitness.py)
  - core.py (from genome.py) 
  - gui.py (from genome_gui.py)
  - __init__.py with proper module exports
- ✅ Created tests/ directory with test_core.py (3 passing tests)
- ✅ Initialized uv environment (.venv) and installed dependencies
- ✅ Ran ruff check --fix and ruff format on all code
- ✅ Configured pytest and ruff in pyproject.toml

**Status:** Phase 1 (Project Setup) ✅ COMPLETE

**Next Steps:**
1. Phase 2: Code Refactoring - Further improvements to imports/structure
2. Create Architecture Decision Records (docs/adr/)
3. Add more comprehensive tests
4. Update README with new setup instructions

**Blockers:** None

---

## Architecture Decision Records

See [docs/adr/](docs/adr/) for architectural decisions:
- 0001-why-uv.md (when created)
- 0002-ruff-configuration.md (when created)
- etc.

## Notes

- Current Python stdlib includes tkinter for GUI (if used)
- Original __pycache__/ can be cleaned up after refactoring
- LICENSE is MIT - preserve in pyproject.toml metadata
