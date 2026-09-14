"""Compact partially-linear DML with five-fold cross-fitting."""

from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .common import DML_TARGETS, MODEL_FEATURES


def _normal_pvalue(z_value: float) -> float:
    return 2 * (1 - 0.5 * (1 + erf(abs(z_value) / sqrt(2))))


def estimate_dml(model_data: pd.DataFrame, n_trees: int = 200) -> pd.DataFrame:
    """Estimate standardized adjusted associations separately by system."""
    rows = []
    for system, data in model_data.groupby("Class", sort=False):
        data = data.dropna(subset=MODEL_FEATURES + ["MP_Risk"]).copy()
        z = data[MODEL_FEATURES + ["MP_Risk"]].apply(
            lambda column: (column - column.mean()) / column.std(ddof=0)
        )
        y = z["MP_Risk"].to_numpy()
        for treatment in DML_TARGETS:
            controls = [feature for feature in MODEL_FEATURES if feature != treatment]
            x, t = z[controls].to_numpy(), z[treatment].to_numpy()
            residual_y, residual_t = np.empty(len(data)), np.empty(len(data))
            for train, test in KFold(5, shuffle=True, random_state=42).split(x):
                outcome_model = RandomForestRegressor(
                    n_estimators=n_trees, min_samples_leaf=2, random_state=42, n_jobs=-1
                )
                treatment_model = Ridge(alpha=1.0)
                outcome_model.fit(x[train], y[train])
                treatment_model.fit(x[train], t[train])
                residual_y[test] = y[test] - outcome_model.predict(x[test])
                residual_t[test] = t[test] - treatment_model.predict(x[test])
            denominator = np.mean(residual_t**2)
            estimate = np.mean(residual_t * residual_y) / denominator
            influence = residual_t * (residual_y - estimate * residual_t) / denominator
            standard_error = influence.std(ddof=1) / sqrt(len(data))
            rows.append({
                "System": system, "Treatment": treatment, "n": len(data),
                "alpha_std": estimate, "se_std": standard_error,
                "ci_low_std": estimate - 1.96 * standard_error,
                "ci_high_std": estimate + 1.96 * standard_error,
                "p_value": _normal_pvalue(estimate / standard_error),
            })
    return pd.DataFrame(rows)
