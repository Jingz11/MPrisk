"""Generate fully synthetic demonstration data; no manuscript records are used."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import MODEL_FEATURES, load_config


ROOT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260914)


def compositions(latent: np.ndarray, mapping: dict[str, float]) -> pd.DataFrame:
    """Create compositions correlated with a synthetic latent pressure gradient."""
    columns = list(mapping)
    weights = np.array([mapping[column] for column in columns], dtype=float)
    high = (weights + 0.05) / (weights + 0.05).sum()
    low = (weights.max() - weights + 0.05)
    low /= low.sum()
    signal = 1 / (1 + np.exp(-latent))
    profile = (1 - signal[:, None]) * low + signal[:, None] * high
    noise = RNG.dirichlet(np.ones(len(columns)), len(latent))
    return pd.DataFrame(0.8 * profile + 0.2 * noise, columns=columns)


def main() -> None:
    n = 160
    config = load_config(ROOT / "config" / "ech_parameters.json")
    system = np.repeat(["river", "lake"], n // 2)
    activity = RNG.normal(size=n) + (system == "lake") * 0.25
    observations = pd.DataFrame({
        "No.ID": [f"DEMO-{index:03d}" for index in range(1, n + 1)],
        "Class": system,
        "MP_abundance": np.exp(2.5 + 1.2 * activity + RNG.normal(0, 0.2, n)),
    })
    for key in ["polymer_hazard", "shape", "size", "color"]:
        observations = pd.concat([observations, compositions(activity, config[key])], axis=1)

    covariates = observations[["No.ID", "Class"]].copy()
    for index, feature in enumerate(MODEL_FEATURES):
        scale = 10 ** (1 + index % 4)
        covariates[feature] = np.maximum(
            0, scale * (2 + (0.5 + index / 50) * activity + RNG.normal(0, 0.2, n))
        )
    output = ROOT / "data"
    observations.to_csv(output / "demo_observations.csv", index=False)
    covariates.to_csv(output / "demo_covariates.csv", index=False)
    print(f"Wrote {n} fully synthetic records to {output}")


if __name__ == "__main__":
    main()
