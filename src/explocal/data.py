"""Utilities for loading example datasets."""
from __future__ import annotations

from pathlib import Path
import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def load_example_dataset(path: Path) -> pd.DataFrame:
    """Load the example dataset shipped with the package.

    Parameters
    ----------
    path:
        Path to the CSV file containing sequences and labels.
    """
    logger.info("Loading dataset from %s", path)
    return pd.read_csv(path)


def iter_sequences(df: pd.DataFrame) -> Iterable[str]:
    """Yield sequences from a dataframe one by one."""
    for seq in df["sequence"]:
        yield seq


__all__ = ["load_example_dataset", "iter_sequences"]
