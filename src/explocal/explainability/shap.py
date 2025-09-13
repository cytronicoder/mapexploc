"""SHAP-based model explainability."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    import shap
except Exception as exc:  # pragma: no cover
    shap = None  # type: ignore[assignment]
    logger.warning("SHAP is not installed: %s", exc)


def explain(model, X: np.ndarray) -> np.ndarray:
    """Return SHAP values for ``X`` using ``model`` if SHAP is available."""
    if shap is None:
        raise RuntimeError("SHAP is not installed")
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


__all__ = ["explain"]
