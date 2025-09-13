"""Integration tests for the basic training pipeline."""
from __future__ import annotations

from pathlib import Path

from explocal.config import load_config
from explocal.data import load_example_dataset
from explocal.features import build_feature_matrix
from explocal.models.rf import train_random_forest, predict


def test_training_pipeline(tmp_path: Path) -> None:
    cfg = load_config(Path("config/default.yml"))
    df = load_example_dataset(Path("examples/data/example_sequences.csv"))
    X = build_feature_matrix(df["sequence"])
    model = train_random_forest(X, df["label"].to_numpy(), cfg.model)
    preds = predict(model, X)
    assert len(preds) == len(df)
