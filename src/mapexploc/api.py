"""Small REST layer exposing prediction and explanation endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from .adapter import BaseModelAdapter, load_adapter
from .explainers.shap import ShapExplainer


class PredictRequest(BaseModel):
    sequences: List[str]
    model_path: Path | None = None


class ExplainRequest(PredictRequest):
    background: List[str] | None = None


def create_app(model: BaseModelAdapter | None = None) -> FastAPI:
    app = FastAPI(title="MAP-ExPLoc")
    adapter = model
    explainer: ShapExplainer | None = None

    @app.post("/predict")
    def predict(req: PredictRequest):
        nonlocal adapter
        if adapter is None:
            adapter = load_adapter(_load_model(req.model_path))
        preds = adapter.predict(req.sequences)
        return {"predictions": preds.tolist()}

    @app.post("/explain")
    def explain(req: ExplainRequest):
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
    if path is None:
        raise ValueError("model_path must be provided")
    import pickle

    return pickle.loads(Path(path).read_bytes())


__all__ = ["create_app"]
