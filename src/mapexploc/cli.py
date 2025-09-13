"""CLI for MAP-ExPLoc with prediction and explanation commands."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import typer

from .adapter import load_adapter
from .explainers.shap import ShapExplainer
from .features import build_feature_matrix
from .models.rf import train_random_forest, predict as rf_predict
from .config import load_config
from .data import load_example_dataset
from .report import ExplanationSet, GlobalReport, LocalReport

logging.basicConfig(level=logging.INFO)
app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def train(config: Path = typer.Option(..., help="Path to YAML config")) -> None:
    """Train a RandomForest model from ``config`` and example data."""
    cfg = load_config(config)
    df = load_example_dataset(Path("examples/data/example_sequences.csv"))
    features = build_feature_matrix(df["sequence"])
    model = train_random_forest(features, df["label"].to_numpy(), cfg.model)
    logger.info("Trained RandomForest on %s samples", len(df))
    Path("model.pkl").write_bytes(pickle.dumps(model))


@app.command()
def predict(sequence: str, model_path: Path = Path("model.pkl")) -> str:
    """Predict the subcellular localization for ``sequence``."""
    logger.info("Loading model from %s", model_path)
    model = pickle.loads(model_path.read_bytes())
    features = build_feature_matrix([sequence])
    pred = rf_predict(model, features)[0]
    typer.echo(pred)
    return pred


@app.command()
def explain(sequence: str, model_path: Path = Path("model.pkl")) -> None:
    """Return SHAP explanation for ``sequence`` in JSON schema."""
    logger.info("Loading model from %s", model_path)
    model = load_adapter(pickle.loads(model_path.read_bytes()))
    features = build_feature_matrix([sequence])
    explainer = ShapExplainer(model, features)
    exp = explainer.explain(features)
    shap_vals = exp.shap_values[0]
    interactions = exp.interaction_values[0]
    report = ExplanationSet(
        local=[
            LocalReport(
                sequence=sequence,
                shap_values=shap_vals.tolist(),
                interaction_values=interactions.tolist(),
            )
        ],
        global_=GlobalReport(mean_abs_shap=explainer.global_summary(exp.shap_values)),
    )
    typer.echo(report.json())


def main() -> None:  # pragma: no cover
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
