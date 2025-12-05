"""Model adapter interface for MAP-ExPLoc."""

from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class BaseModelAdapter(Protocol):
    """Minimal interface expected from user models.

    Models must implement ``predict`` and ``predict_proba`` operating on batches
    of sequences. Optionally ``embed`` may be provided to expose intermediate
    representations that can accelerate explainers such as Deep SHAP.
    """

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        """Return predicted class labels for ``batch``."""

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        """Return class probabilities for ``batch``."""

    def embed(
        self, batch: Sequence[str]
    ) -> Optional[np.ndarray]:  # pragma: no cover - optional
        """Return embeddings for ``batch`` if available."""
        raise NotImplementedError


class _SimpleAdapter:
    """Wrap objects implementing the required methods into ``BaseModelAdapter``."""

    def __init__(self, model: Any):
        """Initialize the adapter with the given model."""
        self.model = model

    def predict(self, batch: Sequence[str]) -> np.ndarray:
        """Return predicted class labels for batch by delegating to wrapped model."""
        return np.asarray(self.model.predict(batch))

    def predict_proba(self, batch: Sequence[str]) -> np.ndarray:
        """Return class probabilities for batch by delegating to wrapped model."""
        return np.asarray(self.model.predict_proba(batch))

    def embed(self, batch: Sequence[str]) -> Optional[np.ndarray]:
        """Return embeddings for batch if the wrapped model supports it."""
        if hasattr(self.model, "embed"):
            return np.asarray(self.model.embed(batch))
        return None


def load_adapter(model: Any) -> BaseModelAdapter:
    """Return a ``BaseModelAdapter`` for ``model``.

    ``model`` may already satisfy the protocol or provide ``predict`` and
    ``predict_proba`` methods. Otherwise a ``TypeError`` is raised.
    """

    if isinstance(model, BaseModelAdapter):
        return model
    if hasattr(model, "predict") and hasattr(model, "predict_proba"):
        return _SimpleAdapter(model)
    raise TypeError("Model does not expose predict/predict_proba")


__all__ = ["BaseModelAdapter", "load_adapter"]
