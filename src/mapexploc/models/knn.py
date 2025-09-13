"""k-Nearest Neighbours model utilities."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def train_knn(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 5,
) -> KNeighborsClassifier:
    """Train a simple k-NN classifier."""
    logger.info("Training k-NN with k=%s", n_neighbors)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return model.fit(features, labels)


def predict(model: KNeighborsClassifier, features: np.ndarray) -> np.ndarray:
    """Predict labels using a trained k-NN model."""
    logger.debug("Running inference on %s samples", len(features))
    return model.predict(features)


__all__ = ["train_knn", "predict", "KNeighborsClassifier"]
