"""Command-line interface for MAP-ExPLoc."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import typer

from mapexploc.config import load_config
from mapexploc.explainers.shap import ShapExplainer
from mapexploc.features import build_feature_matrix
from mapexploc.models.rf import rf_predict, train_random_forest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="MAP-ExPLoc: Explainable Subcellular Localization Predictor")


@app.command()  # type: ignore[misc]
def train(
    config: Path = typer.Option(..., help="Path to configuration file"),
    data_path: Optional[Path] = typer.Option(
        None,
        help=(
            "Path to training data CSV. "
            "Defaults to examples/data/example_sequences.csv"
        ),
    ),
    output_model: Path = typer.Option(
        Path("model.pkl"), help="Path to save the trained model"
    ),
) -> None:
    """Train a Random Forest model."""
    cfg = load_config(config)

    if data_path is None:
        data_path = Path("examples/data/example_sequences.csv")
        if not data_path.exists():
            typer.echo(
                f"Default data file not found at {data_path}. "
                "Please specify --data-path.",
                err=True,
            )
            raise typer.Exit(code=1)

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    logger.info("Building feature matrix...")
    X = build_feature_matrix(df["sequence"])
    y = df["label"]

    logger.info("Training model...")

    # Construct param_grid from config
    # The model config contains single values, so we wrap them in lists
    # and add the 'rf__' prefix required by the pipeline
    param_grid: Dict[str, List[Any]] = {
        "rf__n_estimators": [cfg.model.n_estimators],
        "rf__max_depth": [cfg.model.max_depth],
    }

    # train_random_forest returns a dict with 'model' key
    result = train_random_forest(X, y, param_grid)
    model = result["model"]

    logger.info("Saving model to %s", output_model)
    joblib.dump(model, output_model)


@app.command()  # type: ignore[misc]
def predict(
    sequence: str = typer.Argument(..., help="Protein sequence to predict"),
    model_path: Path = typer.Option(..., help="Path to trained model file"),
) -> None:
    """Predict localization for a single protein sequence."""
    if not model_path.exists():
        typer.echo(f"Model file not found: {model_path}", err=True)
        raise typer.Exit(code=1)

    model = joblib.load(model_path)

    # Build features
    X = build_feature_matrix([sequence])

    # Predict
    pred = rf_predict(model, X)
    typer.echo(f"Prediction: {pred[0]}")


@app.command()  # type: ignore[misc]
def explain(
    sequence: str = typer.Argument(..., help="Protein sequence to explain"),
    model_path: Path = typer.Option(..., help="Path to trained model file"),
    output_dir: Path = typer.Option(
        Path("results/shap"), help="Directory to save SHAP plots"
    ),
) -> None:
    """Explain prediction for a single protein sequence using SHAP."""
    if not model_path.exists():
        typer.echo(f"Model file not found: {model_path}", err=True)
        raise typer.Exit(code=1)

    model = joblib.load(model_path)

    # Initialize explainer
    explainer = ShapExplainer(model, output_dir=str(output_dir))

    # Build features
    X = build_feature_matrix([sequence])

    try:
        # Use sample_size=1 since we only have 1 sample
        _ = explainer.explain_sample(X, sample_size=1)
        typer.echo("Explanation generated.")
        typer.echo(f"SHAP values saved to {output_dir}")
    except Exception as e:
        logger.error("Failed to generate explanation: %s", e)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
