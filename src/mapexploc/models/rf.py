"""Random Forest model utilities."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def train_random_forest(
    features: np.ndarray,
    labels: np.ndarray,
    config: ModelConfig | None = None,
) -> RandomForestClassifier:
    """Train a RandomForest classifier using ``config`` hyperparameters."""
    config = config or ModelConfig()
    logger.info("Training RandomForest with %s trees", config.n_estimators)
    model = RandomForestClassifier(
        n_estimators=config.n_estimators, max_depth=config.max_depth, random_state=0
    )
    return model.fit(features, labels)


def predict(model: RandomForestClassifier, features: np.ndarray) -> np.ndarray:
    """Predict labels using a trained RandomForest model."""
    logger.debug("Running inference on %s samples", len(features))
    return model.predict(features)


__all__ = ["train_random_forest", "predict", "RandomForestClassifier"]
