# MAP-ExPLoc

Model-Agnostic Pipeline for Explainable Localization.

## Installation

### Basic Installation

```bash
pip install map-exploc
```

### From Source

```bash
git clone https://github.com/cytronicoder/explainable-localization-predictor.git
cd explainable-localization-predictor
pip install -e .
```

### Development Environment

For development with all dependencies:

```bash
pip install -e .[dev,notebooks,docs]
```

Or using conda/mamba:

```bash
conda env create -f environment.yml
conda activate eslp
pip install -e .
```

#### Optional Dependencies

- **`dev`**: Testing, linting, and code quality tools
- **`notebooks`**: Jupyter, plotting, and analysis tools
- **`docs`**: Documentation generation tools
- **`torch`**: PyTorch for deep learning models
- **`esm`**: ESM protein language models
- **`bio`**: Bioinformatics tools (BioPython)

## Usage

### CLI

```bash
mapexploc train --config config/default.yml
mapexploc predict MKTIIALSYIFCLVFADYKDDDDK
mapexploc explain MKTIIALSYIFCLVFADYKDDDDK
```

### Python

```python
from pathlib import Path
from mapexploc.config import load_config
from mapexploc.data import load_example_dataset
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import train_random_forest, predict

cfg = load_config(Path("config/default.yml"))
df = load_example_dataset(Path("examples/data/example_sequences.csv"))
X = build_feature_matrix(df["sequence"])
model = train_random_forest(X, df["label"].to_numpy(), cfg.model)
print(predict(model, X))
```

## Development

Install additional dependencies and pre-commit hooks:

```bash
pip install -e .[torch,esm]
pre-commit install
```

Run tests and type checking:

```bash
pre-commit run --files $(git ls-files '*.py')
pytest
```

## Documentation

Build the documentation locally:

```bash
mkdocs serve
```

## Publishing

To build and upload a release to PyPI:

```bash
python -m build
python -m twine upload dist/*
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
