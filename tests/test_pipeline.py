"""Integration tests for the basic training pipeline."""

from __future__ import annotations

from pathlib import Path

from mapexploc.data import load_example_dataset
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import rf_predict, train_random_forest


def test_training_pipeline() -> None:
    df = load_example_dataset(Path("examples/data/example_sequences.csv"))
    X = build_feature_matrix("examples/data/example_sequences.fasta", None)
    model = train_random_forest(X, df["label"].to_numpy(), None)
    preds = rf_predict(model, X)
    assert len(preds) == len(df)
