"""Small REST layer exposing prediction and explanation endpoints."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel

from .adapter import BaseModelAdapter, load_adapter


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

    @app.post("/predict")  # type: ignore[misc]
    def predict(req: PredictRequest) -> Dict[str, List[int]]:
        nonlocal adapter
        if adapter is None:
            adapter = load_adapter(_load_model(req.model_path))
        preds = adapter.predict(req.sequences)
        return {"predictions": preds.tolist()}

    @app.post("/explain")  # type: ignore[misc]
    def explain_endpoint(req: ExplainRequest) -> str:
        nonlocal adapter
        if adapter is None:
            adapter = load_adapter(_load_model(req.model_path))

        # For now, return a placeholder response since full SHAP integration
        # requires proper sequence-to-feature conversion
        return '{"message": "SHAP explanation not fully implemented yet"}'

    return app


def _load_model(path: Path | None) -> BaseModelAdapter:
    """Load model from pickle file."""
    if path is None:
        path = Path("model.pkl")
    return pickle.loads(Path(path).read_bytes())  # type: ignore[no-any-return]


__all__ = ["create_app"]
