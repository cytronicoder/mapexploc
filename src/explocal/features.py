"""Feature engineering convenience wrappers."""
from __future__ import annotations

from typing import Iterable
import numpy as np

from .preprocessing import featurize


def build_feature_matrix(sequences: Iterable[str]) -> np.ndarray:
    """Return a feature matrix for ``sequences``."""
    return featurize(list(sequences))


__all__ = ["build_feature_matrix"]
