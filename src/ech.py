"""Exposure-Characteristics-Hazard (E-C-H) score calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import minmax01


def _row_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=float) / max(float(temperature), 1e-8)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _weighted_component(frame: pd.DataFrame, mapping: dict[str, float]) -> np.ndarray:
    values = np.zeros(len(frame), dtype=float)
    for column, weight in mapping.items():
        if column in frame:
            values += pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy() * weight
    return minmax01(values)


def _reported(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    available = [column for column in columns if column in frame]
    return frame[available].notna().any(axis=1).to_numpy() if available else np.zeros(len(frame), bool)


def compute_ech_scores(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Compute E, C, H and the database-relative 0-100 risk score.

    Run separately by water-body class because min-max normalization is
    system-specific. Missing composition fractions preserve the historical
    calculation (zero contribution) and are separately flagged in the output.
    """
    result = frame.copy()
    abundance = pd.to_numeric(result["MP_abundance"], errors="coerce").fillna(0).to_numpy()
    exposure = minmax01(np.log1p(abundance))
    shape = _weighted_component(result, config["shape"])
    color = _weighted_component(result, config["color"])
    size = _weighted_component(result, config["size"])
    hazard = _weighted_component(result, config["polymer_hazard"])

    cw = config["component_weights"]
    characteristics = minmax01(cw["shape"] * shape + cw["color"] * color + cw["size"] * size)
    shape_color = minmax01(
        cw["shape"] / (cw["shape"] + cw["color"]) * shape
        + cw["color"] / (cw["shape"] + cw["color"]) * color
    )
    weight_e, weight_c, weight_h = _row_softmax(
        np.column_stack([exposure, characteristics, hazard]), config["weight_temp"]
    ).T
    coefficients = config["interactions"]
    interactions = {
        "cont_inter_EC": coefficients["E_C_shape"] * exposure * shape,
        "cont_inter_EH": coefficients["E_H_polymer"] * exposure * hazard,
        "cont_inter_size_color": coefficients["size_color"] * size * color,
        "cont_inter_E_size": coefficients["E_size"] * exposure * size,
        "cont_inter_CH": coefficients["C_H_polymer"] * shape_color * hazard,
        "cont_inter_ECH": coefficients["E_C_H_total"] * exposure * characteristics * hazard,
    }
    result[["E", "C", "H"]] = np.column_stack([exposure, characteristics, hazard])
    result[["w_E", "w_C", "w_H"]] = np.column_stack([weight_e, weight_c, weight_h])
    result["cont_E"] = weight_e * exposure
    result["cont_C"] = weight_c * characteristics
    result["cont_H"] = weight_h * hazard
    for name, values in interactions.items():
        result[name] = values
    contribution_columns = ["cont_E", "cont_C", "cont_H", *interactions]
    result["MP_Risk_raw"] = result[contribution_columns].sum(axis=1)
    result["MP_Risk"] = minmax01(result["MP_Risk_raw"].to_numpy()) * 100
    for label, key in [("polymer", "polymer_hazard"), ("shape", "shape"), ("size", "size"), ("color", "color")]:
        result[f"{label}_reported"] = _reported(result, list(config[key]))
    return result


def compute_scores_by_system(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    return pd.concat(
        [compute_ech_scores(group, config) for _, group in frame.groupby("Class", sort=False)]
    ).sort_index()


def scenario_summary(scores: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "Baseline": (1.0, 1.0, 1.0, 1.0),
        "High Hazard": (1.0, 1.0, 1.3, 1.0),
        "High C": (1.0, 1.3, 1.0, 1.0),
        "High Exposure": (1.3, 1.0, 1.0, 1.0),
        "Worst Case": (1.3, 1.3, 1.3, 1.2),
        "Best Case": (0.7, 0.7, 0.7, 0.8),
        "Increased Interactions": (1.0, 1.0, 1.0, 1.5),
    }
    interaction_columns = [column for column in scores if column.startswith("cont_inter_")]
    rows = []
    for system, group in scores.groupby("Class", sort=False):
        lo, hi = group["MP_Risk_raw"].min(), group["MP_Risk_raw"].max()
        for name, (scale_e, scale_c, scale_h, scale_interaction) in scenarios.items():
            raw = (
                scale_e * group["cont_E"] + scale_c * group["cont_C"] + scale_h * group["cont_H"]
                + scale_interaction * group[interaction_columns].sum(axis=1)
            )
            scenario_score = (raw - lo) / (hi - lo) * 100
            rows.append({"System": system, "Scenario": name, "Mean_MP_Risk": scenario_score.mean()})
    return pd.DataFrame(rows)
