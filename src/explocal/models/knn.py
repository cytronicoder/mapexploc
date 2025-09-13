"""k-Nearest Neighbours model utilities."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def train_knn(X: np.ndarray, y: np.ndarray, n_neighbors: int = 5) -> KNeighborsClassifier:
    """Train a simple k-NN classifier."""
    logger.info("Training k-NN with k=%s", n_neighbors)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return model.fit(X, y)


def predict(model: KNeighborsClassifier, X: np.ndarray) -> np.ndarray:
    """Predict labels using a trained k-NN model."""
    logger.debug("Running inference on %s samples", len(X))
    return model.predict(X)


__all__ = ["train_knn", "predict", "KNeighborsClassifier"]
