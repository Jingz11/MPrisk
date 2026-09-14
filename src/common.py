"""Shared constants and small helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MODEL_FEATURES = [
    "Population_Density",
    "Gross Regional Product Per Capita (yuan)",
    "Agricultural_Plastic_Film_Usage",
    "Wastewater Discharge Quantity",
    "Domestic Waste Innocuous Treatment Capacity (tons/day)",
    "Industrial Particulate Matter Emissions (tons)",
    "VIIRS_2023",
    "2024_temp",
    "TEM",
    "Fertilizer_Consumption",
    "Gross_Output_Value_of_Agriculture,Forestry,Animal Husbandry and Fishery",
]

DML_TARGETS = [
    "Population_Density",
    "Gross Regional Product Per Capita (yuan)",
    "Agricultural_Plastic_Film_Usage",
    "Wastewater Discharge Quantity",
    "Domestic Waste Innocuous Treatment Capacity (tons/day)",
    "Industrial Particulate Matter Emissions (tons)",
    "Fertilizer_Consumption",
    "Gross_Output_Value_of_Agriculture,Forestry,Animal Husbandry and Fishery",
]


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def minmax01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)
