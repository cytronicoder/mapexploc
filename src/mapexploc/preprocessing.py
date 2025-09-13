"""Sequence preprocessing and feature engineering."""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def amino_acid_composition(sequence: str) -> np.ndarray:
    """Compute normalized amino-acid composition."""
    logger.debug(
        "Computing amino-acid composition for sequence of length %s", len(sequence)
    )
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    counts = np.array([sequence.count(aa) for aa in amino_acids], dtype=float)
    return counts / counts.sum()


def featurize(sequences: List[str]) -> np.ndarray:
    """Featurize a list of sequences into numeric features."""
    feats = np.array([amino_acid_composition(seq) for seq in sequences])
    logger.info("Generated feature matrix of shape %s", feats.shape)
    return feats


__all__ = ["amino_acid_composition", "featurize"]
