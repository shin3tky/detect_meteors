# Version 1.6 Release Notes

## Version 1.6.0 (2025-12-21)

### ⚡ Development Toolchain Modernization

Version 1.6.0 modernizes the development toolchain by migrating from pip/black/flake8 to uv/Ruff. This change dramatically improves developer experience with faster dependency management and unified linting/formatting.

### Highlights

- **uv**: Rust-based Python package manager (10-100× faster than pip)
- **Ruff**: Rust-based linter and formatter replacing both black and flake8
- **Unified configuration**: All tool settings in `pyproject.toml`
- **Simplified setup**: Single command installs everything

### Why This Change?

| Aspect | Before (v1.5.x) | After (v1.6.0) |
|--------|-----------------|----------------|
| **Package manager** | pip | uv |
| **Formatter** | black | Ruff |
| **Linter** | flake8 | Ruff |
| **Config files** | `pyproject.toml` + `.flake8` | `pyproject.toml` only |
| **Install speed** | ~30-60 seconds | ~2-5 seconds |
| **Lint + format** | Two separate tools | Single unified tool |

### Performance Comparison

**Dependency installation:**
```bash
# Before (pip)
pip install -e ".[dev]"  # ~30-60 seconds

# After (uv)
uv sync --all-extras     # ~2-5 seconds
```

**Linting and formatting:**
```bash
# Before (black + flake8)
black .                  # ~2 seconds
flake8 .                 # ~1 second

# After (Ruff)
ruff check --fix .       # ~0.1 seconds
ruff format .            # ~0.1 seconds
```

### Migration Guide for Contributors

#### New Developer Setup

```bash
# 1. Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and setup
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv sync --all-extras

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Verify setup
uv run pre-commit run --all-files
```

#### For Existing Contributors

If you have an existing development environment:

```bash
# Update your local repository
git pull origin main

# Remove old virtual environment (optional but recommended)
rm -rf .venv

# Create new environment with uv
uv sync --all-extras

# Reinstall pre-commit hooks
uv run pre-commit install
```

### New Development Workflow

#### Running Commands

All Python commands should be prefixed with `uv run`:

```bash
# Run tests
uv run python run_tests.py

# Run the CLI
uv run python detect_meteors_cli.py --help

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

#### Pre-commit Hooks

The pre-commit configuration now uses local Ruff hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: .venv/bin/ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: .venv/bin/ruff format
        language: system
        types: [python]
```

### Configuration Changes

#### pyproject.toml

The `[tool.ruff]` section replaces both black and flake8 configurations:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "C90"]
ignore = ["E203", "E501", "E226"]

[tool.ruff.lint.mccabe]
max-complexity = 40

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Ruff vs black/flake8 Compatibility

Ruff is designed to be a drop-in replacement for black and flake8. Key compatibility notes:

| Feature | black/flake8 | Ruff |
|---------|--------------|------|
| Line length | `max-line-length = 88` | `line-length = 88` |
| Quote style | black default (double) | `quote-style = "double"` |
| Complexity | `max-complexity = 40` | `[tool.ruff.lint.mccabe] max-complexity = 40` |
| Ignored rules | `.flake8` ignore list | `[tool.ruff.lint] ignore` |

### Dependencies Updated

**Development dependencies (optional):**

```toml
[project.optional-dependencies]
dev = [
    "ruff==0.14.10",      # Replaces black + flake8
    "pre-commit>=4.5.0",
    "coverage>=7.6.0",
]
```

**Removed from dev dependencies:**
- `black>=25.12.0`
- `flake8>=7.3.0`
- `flake8-pyproject>=1.2.4`

### FAQ

#### Q: Do I need to install uv?

Yes, uv is now the recommended way to manage dependencies for this project. It's a single binary with no dependencies.

#### Q: Can I still use pip?

While pip still works for basic installation (`pip install -e .`), the development workflow is optimized for uv. Pre-commit hooks expect Ruff to be available via the uv-managed virtual environment.

#### Q: Will Ruff format code differently than black?

Ruff's formatter is designed to be compatible with black. In most cases, the output is identical. Minor differences may occur in edge cases, but these are cosmetic and don't affect functionality.

#### Q: Is the linting stricter or more lenient?

The same rule set is applied. Ruff supports the same error codes as flake8 (E, W, F, C90), and the ignore list has been preserved.

### Files Changed

| File | Changes |
|------|---------|
| `pyproject.toml` | Added `[tool.ruff]` section, updated dev dependencies |
| `.pre-commit-config.yaml` | Replaced black/flake8 with local Ruff hooks |
| `INSTALL_DEV.md` | Rewritten for uv/Ruff workflow |
| `CHANGELOG.md` | Added v1.6.0 entry |
| `README.md` | Updated "What's New" section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.x:

- **CLI**: No changes to command-line interface
- **Runtime**: No changes to detection behavior or output
- **API**: No changes to `meteor_core` module interfaces
- **Configuration**: No changes to detection parameters or sensor presets

This release only affects the development toolchain. Users who install the package without development dependencies will not notice any difference.

### Links

- [uv documentation](https://docs.astral.sh/uv/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [INSTALL_DEV.md](INSTALL_DEV.md) - Updated developer setup guide

---

## Version Information

- **Version**: 1.6.0
- **Release Date**: 2025-12-21
- **Major Changes**:
  - pip → uv (package management)
  - black + flake8 → Ruff (formatting + linting)
  - Unified configuration in pyproject.toml
  - Simplified developer setup workflow

---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.5.x  
**Recommendation**: Contributors should migrate to uv/Ruff workflow

Happy meteor hunting! 🌠
