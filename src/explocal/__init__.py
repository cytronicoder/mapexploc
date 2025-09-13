"""Explocal: Explainable Subcellular Localization Predictor.

This package provides utilities for preprocessing protein sequences,
training classical machine-learning models, and interpreting
predictions with SHAP.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("explocal")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
