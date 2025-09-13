# Explocal

Explainable Subcellular Localization Predictor.

## Installation

```bash
pip install explocal
```

From source:

```bash
git clone https://github.com/cytronicoder/explainable-localization-predictor.git
cd explainable-localization-predictor
pip install -e .
```

## Usage

### CLI

```bash
explocal train --config config/default.yml
explocal predict MKTIIALSYIFCLVFADYKDDDDK
```

### Python

```python
from pathlib import Path
from explocal.config import load_config
from explocal.data import load_example_dataset
from explocal.features import build_feature_matrix
from explocal.models.rf import train_random_forest, predict

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
