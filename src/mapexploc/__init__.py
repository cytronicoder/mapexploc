from __future__ import annotations
import importlib, sys

for _name in ["config", "data", "features", "models", "preprocessing"]:
    try:
        sys.modules[f"mapexploc.{_name}"] = importlib.import_module(f"explocal.{_name}")
    except Exception:
        pass

from .adapter import BaseModelAdapter, load_adapter
from .api import create_app
from .explainers.shap import ShapExplainer

__all__ = ["BaseModelAdapter", "load_adapter", "ShapExplainer", "create_app"]
