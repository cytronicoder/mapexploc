"""Pydantic models defining the reporting schema."""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class LocalReport(BaseModel):
    sequence: str
    shap_values: List[List[float]]
    interaction_values: List[List[List[float]]]


class GlobalReport(BaseModel):
    mean_abs_shap: Dict[str, float]


class ExplanationSet(BaseModel):
    local: List[LocalReport]
    global_: GlobalReport


__all__ = ["LocalReport", "GlobalReport", "ExplanationSet"]
