"""Integration tests for the basic training pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mapexploc.config import load_config
from mapexploc.data import load_example_dataset
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import rf_predict, train_random_forest


def test_training_pipeline(tmp_path: Path) -> None:
    cfg = load_config(Path("config/default.yml"))
    df = load_example_dataset(Path("examples/data/example_sequences.csv"))
    
    # Duplicate data to ensure enough samples for CV (StratifiedKFold requires >= n_splits samples per class)
    df = pd.concat([df] * 5, ignore_index=True)
    
    X = build_feature_matrix(df["sequence"])
    
    # Construct param_grid from config
    param_grid = {
        "rf__n_estimators": [cfg.model.n_estimators],
        "rf__max_depth": [cfg.model.max_depth],
    }
    
    model = train_random_forest(X, df["label"], param_grid)
    preds = rf_predict(model["model"], X)
    assert len(preds) == len(df)
