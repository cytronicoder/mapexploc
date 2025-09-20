# Requirements Files Documentation

This directory contains several requirements files optimized for different use cases and deployment scenarios.

## requirements.txt

**Complete requirements for full functionality**

This file includes all dependencies needed to run mapexploc with complete functionality, including:

- Core machine learning libraries (scikit-learn, numpy, pandas)
- Bioinformatics tools (BioPython)
- Visualization libraries (matplotlib, seaborn, umap-learn)
- Model explainability tools (SHAP)
- API and CLI frameworks (FastAPI, Typer)
- Jupyter notebook support

Use for: Complete installations, research environments, development

```bash
pip install -r requirements.txt
```

## requirements-minimal.txt

**Minimal production requirements**

This file contains only the core dependencies needed for basic functionality:

- Essential machine learning libraries
- Core API and CLI functionality
- No heavy visualization or development tools

Use for: Production deployments, Docker containers, CI/CD pipelines

```bash
pip install -r requirements-minimal.txt
```

## requirements-dev.txt

**Development and testing requirements**

This file extends the base requirements with additional development tools:

- Testing frameworks (pytest, coverage)
- Code quality tools (black, ruff, mypy)
- Documentation generation (mkdocs)
- Pre-commit hooks and linting tools
- Advanced analysis libraries

Use for: Development environments, contributing to the project

```bash
pip install -r requirements-dev.txt
```

## pyproject.toml

**Package configuration with dependencies**

The pyproject.toml file defines:

- Package metadata and build system configuration
- Core dependencies required for basic functionality
- Optional dependency groups for different use cases
- Development tool configurations (black, ruff, mypy, pytest)

Use for: Installing as a package with optional features

```bash
pip install -e .                    # Basic installation
pip install -e .[notebooks]         # With notebook support
pip install -e .[dev]              # With development tools
```

## environment.yml

This file provides a conda environment specification as an alternative to pip-based installation:

- Uses conda-forge channel for optimized package builds
- Includes both conda and pip dependencies
- Provides a complete development environment
- Compatible with both conda and mamba package managers

Use for: Conda-based environments, reproducible scientific computing setups

```bash
conda env create -f environment.yml
conda activate eslp
```

## Installation Recommendations

### For Research and Analysis

Complete functionality with all visualization and analysis tools:

```bash
pip install -r requirements.txt
```

### For Production Deployment

Minimal dependencies for production environments:

```bash
pip install -r requirements-minimal.txt
```

### For Development and Contributing

Full development environment with testing and quality tools:

```bash
pip install -r requirements-dev.txt
```

### As an Installable Package

Install mapexploc as a Python package with optional features:

```bash
pip install -e .                    # Basic installation
pip install -e .[notebooks]         # With notebook support
pip install -e .[dev]              # With development tools
```

### Using Conda

Create a conda environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate eslp
pip install -e .
```

## Version Management Philosophy

All requirements files use minimum version specifications (>=) to ensure:

- Compatibility with newer package versions
- Automatic inclusion of security updates and bug fixes
- Future-proof installations that work with evolving ecosystems
- Flexibility for users with existing package constraints

Critical version constraints are only specified when necessary for API compatibility or known breaking changes.
