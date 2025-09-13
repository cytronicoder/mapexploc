"""MAP-ExPLoc: Explainable Subcellular Localization Predictor.

This package provides utilities for preprocessing protein sequences,
training classical machine-learning models, and interpreting
predictions with SHAP.
"""

from importlib.metadata import PackageNotFoundError, version

from .adapter import BaseModelAdapter, load_adapter
from .api import create_app
from .config import ModelConfig, Settings, load_config
from .data import iter_sequences, load_example_dataset
from .explainers.shap import ShapExplainer
from .features import build_feature_matrix
from .models import (
    KNeighborsClassifier,
    RandomForestClassifier,
    knn_predict,
    rf_predict,
    train_knn,
    train_random_forest,
)
from .preprocessing import amino_acid_composition, featurize

try:
    __version__ = version("mapexploc")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "BaseModelAdapter",
    "load_adapter",
    "ShapExplainer",
    "create_app",
    "Settings",
    "ModelConfig",
    "load_config",
    "load_example_dataset",
    "iter_sequences",
    "build_feature_matrix",
    "amino_acid_composition",
    "featurize",
    "train_knn",
    "knn_predict",
    "KNeighborsClassifier",
    "train_random_forest",
    "rf_predict",
    "RandomForestClassifier",
]
