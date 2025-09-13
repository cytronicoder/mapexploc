"""Small REST layer exposing prediction and explanation endpoints."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from .adapter import BaseModelAdapter, load_adapter
from .explainers.shap import ShapExplainer


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    sequences: List[str]
    model_path: Path | None = None


class ExplainRequest(PredictRequest):
    """Request model for explanation endpoint, extends PredictRequest."""

    background: List[str] | None = None


def create_app(model: BaseModelAdapter | None = None) -> FastAPI:
    """Create and configure a FastAPI application with prediction and explanation.

    Parameters
    ----------
    model : BaseModelAdapter | None, optional
        Pre-loaded model adapter. If None, models will be loaded from request paths.

    Returns
    -------
    FastAPI
        Configured FastAPI application with /predict and /explain endpoints.
    """
    app = FastAPI(title="MAP-ExPLoc")
    adapter = model
    explainer: ShapExplainer | None = None

    @app.post("/predict")  # type: ignore[misc]
    def predict(req: PredictRequest) -> Dict[str, List[int]]:
        nonlocal adapter
        if adapter is None:
            adapter = load_adapter(_load_model(req.model_path))
        preds = adapter.predict(req.sequences)
        return {"predictions": preds.tolist()}

    @app.post("/explain")  # type: ignore[misc]
    def explain(req: ExplainRequest) -> str:
        nonlocal adapter, explainer
        if adapter is None:
            adapter = load_adapter(_load_model(req.model_path))
        if explainer is None:
            background = req.background if req.background else req.sequences[:10]
            explainer = ShapExplainer(adapter, background)
        result = explainer.explain(req.sequences)
        return result.to_json()

    return app


def _load_model(path: Path | None) -> BaseModelAdapter:
    """Load model from pickle file."""
    if path is None:
        path = Path("model.pkl")
    return pickle.loads(Path(path).read_bytes())  # type: ignore[no-any-return]


__all__ = ["create_app"]
