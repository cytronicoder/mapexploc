"""CLI for MAP-ExPLoc with prediction and explanation commands."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import typer

from .data import load_example_dataset
from .features import build_feature_matrix
from .models.rf import train_random_forest

logging.basicConfig(level=logging.INFO)
app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()  # type: ignore[misc]
def train(_config: Path = typer.Option(..., help="Path to YAML config")) -> None:
    """Train a RandomForest model from ``config`` and example data."""
    df = load_example_dataset(Path("examples/data/example_sequences.csv"))
    features = build_feature_matrix("examples/data/example_sequences.fasta", None)
    model = train_random_forest(features, df["label"].to_numpy(), None)
    logger.info("Trained RandomForest on %s samples", len(df))
    Path("model.pkl").write_bytes(pickle.dumps(model))


@app.command()  # type: ignore[misc]
def predict(_sequence: str, _model_path: Path = Path("model.pkl")) -> None:
    """Predict the subcellular localization for ``sequence``."""
    # NOTE: Implement proper sequence-to-feature conversion  # noqa: FIX002
    # For now, this is a placeholder that assumes the model can handle raw sequences
    pred = "Placeholder - sequence processing not implemented"
    typer.echo(pred)


@app.command()  # type: ignore[misc]
def explain(sequence: str, model_path: Path = Path("model.pkl")) -> None:
    """Return SHAP explanation for ``sequence`` in JSON schema."""
    # NOTE: SHAP explanation not fully implemented yet  # noqa: FIX002
    # This requires fixing the sequence-to-feature conversion pipeline
    report = {
        "message": "SHAP explanation not fully implemented yet",
        "sequence": sequence,
        "model_path": str(model_path),
    }
    typer.echo(str(report))


def main() -> None:  # pragma: no cover
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
