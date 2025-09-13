"""Random Forest model utilities."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def train_random_forest(
    X: np.ndarray, y: np.ndarray, cfg: ModelConfig | None = None
) -> RandomForestClassifier:
    """Train a RandomForest classifier using ``cfg`` hyperparameters."""
    cfg = cfg or ModelConfig()
    logger.info("Training RandomForest with %s trees", cfg.n_estimators)
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=0
    )
    return model.fit(X, y)


def predict(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """Predict labels using a trained RandomForest model."""
    logger.debug("Running inference on %s samples", len(X))
    return model.predict(X)


__all__ = ["train_random_forest", "predict", "RandomForestClassifier"]
