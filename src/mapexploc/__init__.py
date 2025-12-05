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
    evaluate_knn,
    evaluate_rf,
    knn_predict,
    knn_predict_proba,
    rf_predict,
    rf_predict_proba,
    train_knn,
    train_random_forest,
)
from .preprocessing import ALLOWED_LOCS, _clean_and_primary, extract_protein_data

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
    "extract_protein_data",
    "_clean_and_primary",
    "ALLOWED_LOCS",
    "train_knn",
    "knn_predict",
    "knn_predict_proba",
    "evaluate_knn",
    "train_random_forest",
    "rf_predict",
    "rf_predict_proba",
    "evaluate_rf",
]
