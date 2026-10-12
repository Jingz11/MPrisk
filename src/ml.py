"""Direct random-forest and cascaded E-C-H reconstruction models."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .common import MODEL_FEATURES


def assemble_model_data(scores: pd.DataFrame, covariates: pd.DataFrame) -> pd.DataFrame:
    return covariates.merge(
        scores[["No.ID", "E", "C", "H", "MP_Risk"]], on="No.ID", how="inner", validate="one_to_one"
    )


def _strata(y: pd.Series) -> pd.Series | None:
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    return bins if bins.value_counts().min() >= 2 else None


def _metrics(system: str, architecture: str, y_true: pd.Series, predicted: np.ndarray) -> dict:
    return {
        "System": system,
        "Architecture": architecture,
        "n_test": len(y_true),
        "R2": r2_score(y_true, predicted),
        "RMSE": sqrt(mean_squared_error(y_true, predicted)),
        "MAE": mean_absolute_error(y_true, predicted),
    }


def _impute_from_train(x: pd.DataFrame, train: np.ndarray) -> pd.DataFrame:
    """Learn medians on training rows only; fail if a feature has no support."""
    medians = x.iloc[train].median()
    empty = medians.index[medians.isna()].tolist()
    if empty:
        raise ValueError(f"Features entirely missing in training data: {empty}")
    return x.fillna(medians)


def fit_direct_rf(model_data: pd.DataFrame, n_trees: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the observation-level direct random-forest model."""
    metrics, importances = [], []
    for system, data in model_data.groupby("Class", sort=False):
        data = data.dropna(subset=["MP_Risk"]).copy()
        x, y = data[MODEL_FEATURES], data["MP_Risk"]
        train, test = train_test_split(
            np.arange(len(data)), test_size=0.2, random_state=42, stratify=_strata(y)
        )
        x = _impute_from_train(x, train)
        model = RandomForestRegressor(
            n_estimators=n_trees, max_depth=20, max_features=0.7,
            min_samples_split=5, min_samples_leaf=1, random_state=20, n_jobs=-1,
        )
        model.fit(x.iloc[train], y.iloc[train])
        metrics.append(_metrics(system, "Direct RF", y.iloc[test], model.predict(x.iloc[test])))
        importances.extend(
            {"System": system, "Feature": feature, "Importance": value}
            for feature, value in zip(MODEL_FEATURES, model.feature_importances_)
        )
    return pd.DataFrame(metrics), pd.DataFrame(importances)


def fit_cascade_rf(
    model_data: pd.DataFrame, n_trees: int = 300, *, evaluation: str = "internal"
) -> pd.DataFrame:
    """Preserve internal reconstruction; opt in to train-isolated holdout.

    ``holdout`` isolates imputation and every RF fit from outer test rows.
    Targets must already be defined; this does not fit ECH score normalization.
    Training meta-features are in-sample predictions, not cross-fitted stacking.
    """
    if evaluation not in {"internal", "holdout"}:
        raise ValueError("evaluation must be 'internal' or 'holdout'")
    metrics = []
    for system, data in model_data.groupby("Class", sort=False):
        data = data.dropna(subset=["E", "C", "H", "MP_Risk"]).copy()
        x, y = data[MODEL_FEATURES], data["MP_Risk"]
        train, test = train_test_split(
            np.arange(len(data)), test_size=0.2, random_state=42, stratify=_strata(y)
        )
        x = (_impute_from_train(x, train) if evaluation == "holdout"
             else x.fillna(x.median()))
        cascade_x = x.copy()
        fitted = []
        for dimension in ["E", "C", "H"]:
            stage = RandomForestRegressor(
                n_estimators=max(100, n_trees // 2), max_features=0.7,
                random_state=42, n_jobs=-1,
            )
            if evaluation == "holdout":
                stage.fit(x.iloc[train], data[dimension].iloc[train])
            else:
                stage.fit(x, data[dimension])
            fitted.append(stage.predict(x))
        cascade_x[["E_fitted", "C_fitted", "H_fitted"]] = np.column_stack(fitted)
        model = RandomForestRegressor(
            n_estimators=n_trees, max_depth=20, max_features=0.7,
            min_samples_split=5, random_state=42, n_jobs=-1,
        )
        model.fit(cascade_x.iloc[train], y.iloc[train])
        metrics.append(
            _metrics(
                system,
                "Cascade holdout RF" if evaluation == "holdout" else "Cascade reconstruction RF",
                y.iloc[test], model.predict(cascade_x.iloc[test]),
            )
        )
    return pd.DataFrame(metrics)
