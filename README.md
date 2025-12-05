# MAP-ExPLoc

Model-Agnostic Pipeline for Explainable Protein Subcellular Localization

## Overview

MAP-ExPLoc is a comprehensive machine learning pipeline designed for predicting protein subcellular localization from amino acid sequences. The package provides explainable AI capabilities through SHAP analysis and supports multiple machine learning algorithms including k-NN and Random Forest classifiers.

### Key Features

- **Swiss-Prot Data Processing**: Extract and clean protein data from Swiss-Prot databases
- **Feature Engineering**: Generate 31 physicochemical features from protein sequences
- **Multiple ML Models**: k-NN and Random Forest classifiers with hyperparameter optimization
- **Explainable AI**: SHAP-based model interpretation and visualization
- **Production Ready**: REST API and CLI interfaces for deployment
- **Comprehensive Evaluation**: Model performance metrics and visualization tools

## Installation

### Standard Installation

Install the package from PyPI:

```bash
pip install mapexploc
```

### Complete Installation with All Features

For full functionality including visualization and notebook support:

```bash
pip install -r requirements.txt
```

### Minimal Production Installation

For deployment environments with minimal dependencies:

```bash
pip install -r requirements-minimal.txt
```

### Development Installation

For contributors and developers:

```bash
git clone https://github.com/cytronicoder/mapexploc.git
cd mapexploc
pip install -r requirements-dev.txt
pip install -e .
```

### Conda Environment

Using conda for package management:

```bash
conda env create -f environment.yml
conda activate eslp
pip install -e .
```

### Dependency Groups

- **Core**: Essential packages for basic functionality
- **Notebooks**: Jupyter environment and visualization tools
- **Development**: Testing, linting, and code quality tools
- **Documentation**: Tools for generating and serving documentation
- **Bioinformatics**: Specialized tools for protein sequence analysis

## Usage

### Command Line Interface

Train a model using the default configuration:

```bash
mapexploc train --config config/default.yml
```

Make predictions on a protein sequence:

```bash
mapexploc predict MKTIIALSYIFCLVFADYKDDDDK
```

Generate explanations for predictions:

```bash
mapexploc explain MKTIIALSYIFCLVFADYKDDDDK
```

### Python API

Basic usage example:

```python
from pathlib import Path
from mapexploc.config import load_config
from mapexploc.data import load_example_dataset
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest, predict

# Load configuration
cfg = load_config(Path("config/default.yml"))

# Load and prepare data
df = load_example_dataset(Path("examples/data/example_sequences.csv"))
X = build_feature_matrix(df["sequence"])

# Train model
model = train_random_forest(X, df["label"].to_numpy(), cfg.model)

# Make predictions
predictions = predict(model, X)
print(predictions)
```

### Complete Workflow Example

```python
from mapexploc.preprocessing import extract_protein_data
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest
from mapexploc.explainers.shap import ShapExplainer

# 1. Extract protein data from Swiss-Prot
protein_data = extract_protein_data("uniprot_sprot.dat")

# 2. Build feature matrix
features_df = build_feature_matrix("sequences.fasta", "annotations.csv")

# 3. Train Random Forest model
X = features_df.drop(['entry_name', 'localization'], axis=1)
y = features_df['localization']
rf_result = train_random_forest(X, y)

# 4. Generate SHAP explanations
explainer = ShapExplainer(rf_result['model'])
explanations = explainer.generate_all_plots(X.sample(100))
```

## REST API

Start the API server:

```bash
uvicorn mapexploc.api:app --host 0.0.0.0 --port 8000
```

The API provides endpoints for:

- `/predict`: Protein localization prediction
- `/explain`: Model explanations via SHAP
- `/health`: Service health check

## Development

### Setting Up Development Environment

Install the package in development mode with all dependencies:

```bash
git clone https://github.com/cytronicoder/mapexploc.git
cd mapexploc
pip install -r requirements-dev.txt
pip install -e .
```

Install pre-commit hooks for code quality:

```bash
pre-commit install
```

### Running Tests

Execute the test suite:

```bash
pytest
pytest --cov=mapexploc  # With coverage
```

### Code Quality

Run linting and formatting:

```bash
pre-commit run --all-files
black src/ tests/
ruff check src/ tests/
mypy src/
```

### Building Documentation

Serve documentation locally:

```bash
mkdocs serve
```

Build documentation for production:

```bash
mkdocs build
```

## Publishing

### Building the Package

Create distribution packages:

```bash
python -m build
```

### Uploading to PyPI

Upload to PyPI (requires authentication):

```bash
python -m twine upload dist/*
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality (`pre-commit run --all-files`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
