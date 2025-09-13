"""Command-line interface for Explocal."""
from __future__ import annotations

import logging
from pathlib import Path
import pickle

logging.basicConfig(level=logging.INFO)
import typer

from .config import load_config
from .data import load_example_dataset
from .features import build_feature_matrix
from .models.rf import train_random_forest, predict as rf_predict

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def train(config: Path = typer.Option(..., help="Path to YAML config")) -> None:
    """Train a RandomForest model from ``config`` and example data."""
    cfg = load_config(config)
    df = load_example_dataset(Path("examples/data/example_sequences.csv"))
    X = build_feature_matrix(df["sequence"])
    model = train_random_forest(X, df["label"].to_numpy(), cfg.model)
    logger.info("Trained RandomForest on %s samples", len(df))
    Path("model.pkl").write_bytes(pickle.dumps(model))


@app.command()
def predict(sequence: str, model_path: Path = Path("model.pkl")) -> str:
    """Predict the subcellular localization for ``sequence``."""

    logger.info("Loading model from %s", model_path)
    model = pickle.loads(model_path.read_bytes())
    X = build_feature_matrix([sequence])
    pred = rf_predict(model, X)[0]
    typer.echo(pred)
    return pred


def main() -> None:  # pragma: no cover - thin wrapper
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

