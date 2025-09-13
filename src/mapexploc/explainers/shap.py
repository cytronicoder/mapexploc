"""Unified SHAP interface with fallbacks and reporting utilities."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from ..adapter import BaseModelAdapter

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import shap
except Exception as exc:  # pragma: no cover
    shap = None  # type: ignore
    logger.warning("SHAP is not installed: %s", exc)


@dataclass
class Explanation:
    shap_values: np.ndarray
    interaction_values: np.ndarray
    expected_value: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "shap_values": self.shap_values.tolist(),
                "interaction_values": self.interaction_values.tolist(),
                "expected_value": self.expected_value,
            }
        )


class ShapExplainer:
    """Wrapper selecting the best available SHAP explainer.

    Parameters
    ----------
    model:
        Model implementing :class:`~mapexploc.adapter.BaseModelAdapter`.
    background:
        Background sample used for estimating expectations; when ``None`` a small
        subset of the input batch is used.
    """

    def __init__(
        self, model: BaseModelAdapter, background: Sequence[str] | None = None
    ):
        if shap is None:  # pragma: no cover - runtime guard
            raise RuntimeError("SHAP is not installed")
        self.model = model
        self.background = background
        self._explainer: Any | None = None

    def _init_explainer(self, X: np.ndarray) -> None:
        if self._explainer is not None:
            return
        if hasattr(self.model, "embed") and self.model.embed(["M"]) is not None:
            logger.info("Using DeepExplainer")
            self._explainer = shap.DeepExplainer(self.model.predict_proba, X)
        else:
            logger.info("Using KernelExplainer")
            background = X if self.background is None else self.background
            self._explainer = shap.KernelExplainer(self.model.predict_proba, background)

    def explain(self, batch: Sequence[str]) -> Explanation:
        X = np.asarray(batch)
        self._init_explainer(X)
        shap_vals = self._explainer.shap_values(X)
        shap_vals = np.asarray(shap_vals)
        try:  # interaction values are not supported by all explainers
            interactions = np.asarray(self._explainer.shap_interaction_values(X))
        except Exception:  # pragma: no cover - optional
            interactions = np.zeros(shap_vals.shape + (shap_vals.shape[-1],))
        expected = (
            self._explainer.expected_value[0]
            if isinstance(self._explainer.expected_value, np.ndarray)
            else float(self._explainer.expected_value)
        )
        return Explanation(shap_vals, interactions, expected)

    def global_summary(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Return mean absolute attribution per feature."""
        agg = np.abs(shap_values).mean(axis=tuple(range(shap_values.ndim - 1)))
        return {str(i): float(val) for i, val in enumerate(agg)}


def explain(model, X: np.ndarray) -> np.ndarray:
    """Return SHAP values for ``X`` using ``model`` if SHAP is available.

    This is a simple compatibility function for the original explocal interface.
    """
    if shap is None:
        raise RuntimeError("SHAP is not installed")
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


__all__ = ["ShapExplainer", "Explanation", "explain"]
