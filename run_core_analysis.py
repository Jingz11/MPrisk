"""Run the complete workflow on the synthetic demonstration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import MODEL_FEATURES, load_config
from src.dml import estimate_dml
from src.ech import compute_scores_by_system, scenario_summary
from src.ml import assemble_model_data, fit_cascade_rf, fit_direct_rf


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use fewer trees for a smoke test.")
    args = parser.parse_args()

    observations = pd.read_csv(ROOT / "data" / "demo_observations.csv")
    covariates = pd.read_csv(ROOT / "data" / "demo_covariates.csv")
    config = load_config(ROOT / "config" / "ech_parameters.json")
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    scores = compute_scores_by_system(observations, config)
    scenarios = scenario_summary(scores)
    model_data = assemble_model_data(scores, covariates)
    n_trees = 80 if args.quick else 300
    direct_metrics, feature_importance = fit_direct_rf(model_data, n_trees=n_trees)
    cascade_metrics = fit_cascade_rf(model_data, n_trees=n_trees)
    model_metrics = pd.concat([direct_metrics, cascade_metrics], ignore_index=True)
    dml = estimate_dml(model_data, n_trees=50 if args.quick else 200)

    scores.to_csv(output_dir / "ech_scores.csv", index=False)
    scenarios.to_csv(output_dir / "scenario_summary.csv", index=False)
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)
    feature_importance.to_csv(output_dir / "model_feature_importance.csv", index=False)
    dml.to_csv(output_dir / "dml_overall.csv", index=False)

    summary = {
        "mode": "quick" if args.quick else "full",
        "n_observations": int(len(scores)),
        "systems": {
            system: {
                "n": int(len(group)),
                "mean_MP_Risk": float(group["MP_Risk"].mean()),
            }
            for system, group in scores.groupby("Class")
        },
        "complete_model_rows": {
            system: int(len(group.dropna(subset=MODEL_FEATURES)))
            for system, group in model_data.groupby("Class")
        },
    }
    with (output_dir / "core_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nModel metrics")
    print(model_metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
