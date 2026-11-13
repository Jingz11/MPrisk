"""Check station-level spatial dependence in the current Figure 7 PLR residuals."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

import nuisance
from dml import (
    KEYS, RESIDUAL_COLS, choose_sample, estimate, fit_cell, load, multipletests, np, pd, student,
    threadpool_limits,
)


NEIGHBOURS = (4, 8)
PERMUTATIONS = 9999
BANDWIDTH_KM = (200, 500)


def chord_distances(sites):
    lat, lon = np.deg2rad(sites[["latitude_deg", "longitude_deg"]].to_numpy().T)
    xyz = np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    return cdist(xyz, xyz)


def spatial_weights(sites, k):
    # Chord distance on the unit sphere orders neighbours by great-circle distance.
    distance = chord_distances(sites)
    np.fill_diagonal(distance, np.inf)
    neighbours = np.argsort(distance, axis=1, kind="stable")[:, :k]
    graph = csr_matrix((np.ones(len(sites) * k), (np.repeat(np.arange(len(sites)), k),
                                                 neighbours.ravel())), shape=distance.shape)
    return graph.maximum(graph.T)


def diagnose(task):
    workbook, scores, roster_path, *cell = task
    result, residuals = fit_cell((workbook, scores, *cell))
    data = load(workbook, scores).drop_duplicates("observation_id").set_index("observation_id")
    roster = pd.read_csv(roster_path, usecols=["observation_id", "station_block", "coordinate_level"])
    coords = data[["coordinates_usable", "longitude_deg", "latitude_deg"]].join(
        roster.set_index("observation_id"), validate="one_to_one"
    )
    frame = pd.DataFrame()
    sites = pd.DataFrame()
    usable_rows = 0
    slope = result.get("theta_points", np.nan)
    if residuals is not None and np.isfinite(slope):
        frame = residuals.join(coords, on="observation_id", validate="one_to_one")
        eligible = (frame.coordinates_usable & frame.longitude_deg.between(-180, 180)
                    & frame.latitude_deg.between(-90, 90) & frame.station_block.notna()
                    & frame.coordinate_level.str.startswith(("A", "B", "C"), na=False))
        frame = frame.loc[eligible].copy()
        usable_rows = len(frame)
        frame["residual"] = frame.ry - slope * frame.rt
        sites = frame.groupby(["study", "station_block"]).agg(
            longitude_deg=("longitude_deg", "median"), latitude_deg=("latitude_deg", "median"),
            residual=("residual", "mean"),
        ).reset_index()

    n_sites = len(sites)
    n_studies = sites.study.nunique() if n_sites else 0
    n_locations = sites[["longitude_deg", "latitude_deg"]].drop_duplicates().shape[0] if n_sites else 0
    study_blocks = list(sites.groupby("study").indices.values()) if n_sites else []
    centered = sites.residual.to_numpy() - sites.residual.mean() if n_sites else np.array([])
    sum_squares = centered @ centered
    if not np.isfinite(slope):
        status = "PLR slope unavailable"
    elif n_locations < 12 or n_studies < 2:
        status = "Insufficient spatial support"
    elif sum_squares <= 1e-12 or not any(np.ptp(centered[ids]) > 1e-10 for ids in study_blocks):
        status = "No permutable residual variation"
    else:
        status = "Estimated"

    rows = []
    with threadpool_limits(limits=1):
        if status == "Estimated":
            rng = np.random.default_rng(119105)
            permuted = np.broadcast_to(centered, (PERMUTATIONS, n_sites)).copy()
            for ids in study_blocks:
                if len(ids) > 1:
                    permuted[:, ids] = rng.permuted(permuted[:, ids], axis=1)
        for k in NEIGHBOURS:
            row = dict(zip(KEYS, cell), n=result["n"], usable_rows=usable_rows, station_units=n_sites,
                       unique_locations=n_locations, studies=n_studies, k=k, status=status,
                       moran_I=np.nan, null_median=np.nan, p_blocked=np.nan)
            if status == "Estimated":
                weights = spatial_weights(sites, k)
                scale = n_sites / (weights.sum() * sum_squares)
                observed = scale * (centered @ (weights @ centered))
                null = scale * np.einsum("ij,ji->i", permuted, weights @ permuted.T)
                midpoint = np.median(null)
                p = (1 + np.count_nonzero(np.abs(null - midpoint) >= abs(observed - midpoint))) / (1 + len(null))
                row.update(moran_I=observed, null_median=midpoint, p_blocked=p)
            rows.append(row)
        intervals = spatial_intervals(cell, result, frame, n_locations, n_studies)
    return result, rows, intervals


def spatial_intervals(task, main, frame, n_locations, n_studies):
    """Compare cluster and spatial variances on the same coordinate-usable sample."""
    meta = dict(zip(KEYS, task), n_main=main["n"], n_coordinate=len(frame),
                unique_locations=n_locations, studies=n_studies)
    if n_locations < 12 or n_studies < 2:
        return [dict(**meta, bandwidth_km=km, status="Insufficient spatial support") for km in BANDWIDTH_KM]
    match = estimate(frame)
    if match["status"] != "estimated":
        return [dict(**meta, bandwidth_km=km, status=match["status"]) for km in BANDWIDTH_KM]

    weights = nuisance.weight(frame.assignment)
    treatment = frame.rt.to_numpy()
    residual = frame.ry.to_numpy() - match["theta_points"] * treatment
    scale = nuisance.sd(frame["T"], weights) / nuisance.sd(frame.Y, weights)
    influence = scale * weights * treatment * residual / np.dot(weights, treatment**2)
    stations = frame.assign(influence=influence).groupby(["study", "station_block"]).agg(
        longitude_deg=("longitude_deg", "median"), latitude_deg=("latitude_deg", "median"),
        influence=("influence", "sum"),
    )
    distance = 2 * 6371.0088 * np.arcsin(np.clip(chord_distances(stations) / 2, 0, 1))
    theta = match["theta_standardized"]
    df = match["df"]
    critical = student.ppf(.975, df)
    cluster_se = (match["ci_high"] - match["ci_low"]) / (2 * critical)
    score = stations.influence.to_numpy()
    rows = []
    for km in BANDWIDTH_KM:
        kernel = np.maximum(0, 1 - distance / km)
        variance = score @ kernel @ score
        row = dict(**meta, bandwidth_km=km, df=df, theta_standardized=theta,
                   cluster_se=cluster_se, cluster_low=match["ci_low"], cluster_high=match["ci_high"],
                   cluster_p=match["p"], spatial_variance=variance,
                   station_pairs_within_bandwidth=int(np.triu(distance < km, k=1).sum()))
        if not np.isfinite(variance) or variance <= 0:
            rows.append(dict(**row, status="No positive spatial variance"))
            continue
        spatial_se = np.sqrt(variance)
        combined_se = max(cluster_se, spatial_se)
        rows.append(dict(**row, status="Estimated", spatial_se=spatial_se,
                         spatial_low=theta-critical*spatial_se, spatial_high=theta+critical*spatial_se,
                         spatial_p=2*student.sf(abs(theta/spatial_se), df),
                         combined_low=theta-critical*combined_se, combined_high=theta+critical*combined_se,
                         combined_p=2*student.sf(abs(theta/combined_se), df)))
    return rows


def province_sensitivity(task):
    workbook, scores, _, scope, habitat, variable = task
    q = choose_sample(load(workbook, scores), variable, habitat, scope)
    row = dict(zip(KEYS, task[3:]), n=len(q), provinces=q.province_normalized.nunique(),
               study_count=q.study.nunique(), blocked_folds=0, min_train_rows=np.nan,
               min_train_studies=np.nan)
    if row["provinces"] < 3 or q.province_normalized.isna().any():
        return dict(**row, status="Insufficient or missing province support")
    if q["T"].nunique() < 2 or q.Y.nunique() < 2:
        return dict(**row, status="Constant treatment or score")

    blocks = nuisance.group_codes(q, "province_normalized", 5)
    residuals = []
    training = []
    with threadpool_limits(limits=1):
        for fold in sorted(blocks.unique()):
            test = q.loc[blocks.eq(fold)]
            train = q.loc[blocks.ne(fold) & ~q.study.isin(test.study) & ~q.assignment.isin(test.assignment)]
            training.append((len(train), train.study.nunique()))
            if training[-1][1] < 2 or train["T"].nunique() < 2 or train.Y.nunique() < 2:
                continue
            assert not set(train.province_normalized) & set(test.province_normalized)
            py, pt = nuisance.fit_nuisance(train, test, variable, "rf")
            residuals.append(test[RESIDUAL_COLS].assign(ry=test.Y-py, rt=test["T"]-pt))

    row.update(blocked_folds=len(residuals), min_train_rows=min(n for n, _ in training),
               min_train_studies=min(s for _, s in training))
    if len(residuals) != len(training):
        return dict(**row, status="Insufficient training support in at least one fold")
    residual = pd.concat(residuals)
    assert len(residual) == len(q) and residual.observation_id.is_unique
    return dict(**row, **estimate(residual))


def run(workbook, scores, roster, output):
    from itertools import product
    from dml import BASINS
    output.mkdir(parents=True, exist_ok=True)
    variables = sorted(load(workbook, scores).variable.unique())
    cells = [("National", h, v) for h, v in product(("river", "lake"), variables)]
    cells += [(b, "river", v) for b, v in product(BASINS, variables)]
    tasks = [(workbook, scores, roster, *cell) for cell in cells]
    with ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(diagnose, tasks))
    fitted = pd.DataFrame([result for result, _, _ in results]).set_index(KEYS)
    columns = ["theta_standardized", "ci_low", "ci_high", "p"]
    fitted["p_Holm"]=np.nan
    for _, part in fitted.groupby(np.where(fitted.index.get_level_values("scope")=="National","national","basins")):
        fitted.loc[part.index,"p_Holm"]=np.where(part.p.notna(),multipletests(part.p.fillna(1),method="holm")[1],np.nan)
    fitted.to_csv(output/"dml_coefficients.csv")
    old=fitted

    diagnostics = pd.DataFrame([row for _, rows, _ in results for row in rows])
    diagnostics["p_Holm"] = np.nan
    for _, group in diagnostics.groupby(np.where(diagnostics.scope.eq("National"), "national", "basin")):
        adjusted = multipletests(group.p_blocked.fillna(1), method="holm")[1]
        diagnostics.loc[group.index, "p_Holm"] = np.where(group.p_blocked.notna(), adjusted, np.nan)
    diagnostics.to_csv(output/"spatial_diagnostics.csv", index=False)
    intervals = pd.DataFrame([row for _, _, rows in results for row in rows])
    families = np.where(intervals.scope.eq("National"), "national", "basin")
    for _, group in intervals.groupby([families, "bandwidth_km"]):
        for field in ("cluster_p", "spatial_p", "combined_p"):
            adjusted = multipletests(group[field].fillna(1), method="holm")[1]
            intervals.loc[group.index, f"{field}_Holm"] = np.where(group[field].notna(), adjusted, np.nan)
    intervals.to_csv(output/"spatial_intervals.csv", index=False)
    with ProcessPoolExecutor(max_workers=2) as pool:
        blocked = pd.DataFrame(pool.map(province_sensitivity, tasks)).set_index(KEYS)
    blocked = blocked.join(old[["theta_standardized", "ci_low", "ci_high", "p_Holm"]].add_prefix("main_"))
    blocked["p_Holm"] = np.nan
    for _, group in blocked.groupby(np.where(blocked.index.get_level_values("scope") == "National",
                                              "national", "basin")):
        adjusted = multipletests(group.p.fillna(1), method="holm")[1]
        blocked.loc[group.index, "p_Holm"] = np.where(group.p.notna(), adjusted, np.nan)
    blocked.to_csv(output/"province_sensitivity.csv")
    print(diagnostics.groupby(["k", "status"]).size().to_string())
    print(blocked.status.value_counts().to_string())
    print(intervals.groupby(["bandwidth_km", "status"]).size().to_string())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workbook", "scores", "roster", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.workbook, args.scores, args.roster, args.output)
