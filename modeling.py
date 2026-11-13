"""Reproduce the fixed Figure 5 learners and the selected Figure 6 RF cascade."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits

SEEDS = (42, 52, 62, 72, 82, 92)
CV_SEEDS = (119105, 119106, 119107)
COMPONENTS = ("E_ln_count", "C_ln_opportunity", "H_ln_multiplier")
FEATURES = [
    "agri_film_prov_t", "VIIRS_valid_3km_nW_cm2_sr", "temp_L8_climatology_degC", "elevation_2p5min_m",
    "fertilizer_prov_10kt", "agri_output_prov_100mCNY", "PopDensity_urban_construction", "GRP_pc",
    "WW_Discharge", "Waste_Treat", "IPME", "mulch_area_prov_kha", "crop_sown_area_prov_kha",
    "irrigated_area_prov_kha", "pig_stock_prov_10khead", "precip_prior_yr_mm", "runoff_L8_climatology_mm_yr",
    "cropland_L8_context_pct", "urban_L8_context_pct",
]
BUILD = {"ElasticNet": ElasticNet, "Ridge": Ridge, "HGB": HistGradientBoostingRegressor,
         "GB": GradientBoostingRegressor, "ExtraTrees": ExtraTreesRegressor, "RF": RandomForestRegressor}
FIXED = {"ElasticNet": {"max_iter": 4000}, "Ridge": {}, "HGB": {"early_stopping": False},
         "GB": {"loss": "squared_error"}, "ExtraTrees": {"n_jobs": 1}, "RF": {"n_jobs": 1}}


def read_inputs(workbook):
    book = pd.read_excel(workbook, sheet_name=["ECH逐条", "建模逐条", "预测逐条", "模型配置", "图5汇总"])
    frame = book["建模逐条"].copy()
    z = frame.observation_id.map(book["ECH逐条"].set_index("id").Z)
    np.testing.assert_allclose(frame.Z_fit, z, rtol=0, atol=2e-13)
    frame["Z"] = frame.Z_fit
    assert len(frame) == 2532 and frame.Z.notna().all()
    return frame, book["模型配置"], book["预测逐条"].set_index("observation_id"), book["图5汇总"]


def split(frame, habitat, seed):
    data = frame.loc[frame.habitat.eq(habitat)]
    role = data[f"role_{seed}"]
    tr, te = data[role.eq("train")], data[role.eq("test")]
    assert len(tr) + len(te) == len(data)
    assert not set(tr.linkage_block) & set(te.linkage_block)
    return tr, te


def study_weights(frame):
    w = 1 / frame.groupby("group").group.transform("size").to_numpy()
    return w / w.mean()


def design(tr, te, feature_set, family):
    columns = FEATURES[:11] if feature_set == "Original11" else FEATURES
    assert feature_set in ("Original11", "Original11_plus8")
    a, b = [], []
    for name in columns:
        u, v = tr[name].to_numpy(float), te[name].to_numpy(float)
        available = np.isfinite(u)
        med = np.median(u[available]) if available.any() else 0
        u, v = np.where(available, u, med), np.where(np.isfinite(v), v, med)
        if u.min() >= 0 and pd.Series(u).skew() > 1:
            u, v = np.log1p(u), np.log1p(np.maximum(v, 0))
        a.append(u)
        b.append(v)
    for name in ("habitat", "medium"):
        u, v = tr[name].fillna("unknown").astype(str), te[name].fillna("unknown").astype(str)
        for level in sorted(u.unique()):
            a.append(u.eq(level).to_numpy())
            b.append(v.eq(level).to_numpy())
    a, b = [np.column_stack(cols).astype("float32") for cols in (a, b)]
    if family in ("Ridge", "ElasticNet"):
        scaler = RobustScaler(quantile_range=(20, 80)).fit(a)
        a, b = [np.clip(scaler.transform(q), -12, 12) for q in (a, b)]
    return a, b


def fit(tr, a, b, family, params, target, recipe, *, return_model=False):
    w = study_weights(tr)
    y = tr[target].to_numpy()
    if recipe == "logit":
        y = logit(y.clip(.001, 99.999) / 100)
    center = np.average(y, weights=w)
    scale = max(np.sqrt(np.average((y-center)**2, weights=w)), 1e-8)
    fixed = FIXED[family].copy()
    if family not in ("Ridge", "ElasticNet"):
        fixed["random_state"] = 119105
    model = BUILD[family](**fixed, **params).fit(a, (y-center)/scale, sample_weight=w)
    def predict(x):
        out = model.predict(x)*scale+center
        if recipe == "logit":
            return 100*expit(out)
        return out.clip(0, 100) if target == "Z" else out
    return (predict, model) if return_model else predict(b)


def direct(tr, te, config):
    a, b = design(tr, te, config.feature_set, config.model)
    return fit(tr, a, b, config.model, json.loads(config.parameters), "Z", config.recipe)


def errors(observed, predicted):
    residual = np.asarray(observed) - predicted
    return {"RMSE": np.sqrt(np.mean(residual**2)), "MAE": np.mean(np.abs(residual))}


def figure5(frame, params, saved, summary, mode):
    result = []
    if mode == "test":
        for config, seed in product(params.loc[params.use.eq("F5_fixed")].itertuples(), SEEDS):
            tr, te = split(frame, config.habitat, seed)
            z = direct(tr, te, config)
            ref = saved.loc[te.observation_id, f"Z_{seed}_{config.model}"]
            np.testing.assert_allclose(z, ref, rtol=0, atol=1e-8)
            result.append(dict(seed=seed, habitat=config.habitat, model=config.model,
                               metric="test_R2", value=r2_score(te.Z, z), **errors(te.Z, z)))
    else:
        cv_params = params.loc[params.use.eq("F5_CV")]
        assert set(cv_params.seed) == set(CV_SEEDS)
        for config in cv_params.itertuples():
            tr, _ = split(frame, config.habitat, int(config.seed))
            predictions = np.empty(len(tr))
            for a, b in GroupKFold(3).split(tr, groups=tr.linkage_block):
                predictions[b] = direct(tr.iloc[a], tr.iloc[b], config)
            result.append(dict(seed=int(config.seed), habitat=config.habitat, model=config.model,
                               metric="CV_R2", value=r2_score(tr.Z, predictions)))
        stats = pd.DataFrame(result).groupby(["habitat", "model"]).value.agg(["mean", "std"])
        ref = summary.set_index(["habitat", "model"]).loc[stats.index]
        np.testing.assert_allclose(stats, ref[["CV_pooled_R2_mean", "CV_pooled_R2_SD"]], rtol=0, atol=1e-8)
    return result


def build_component_oof(tr):
    """Rebuild the strict seed-102 inputs for the second-stage forest."""
    reference = {"n_estimators": 256, "max_depth": 18, "min_samples_leaf": 1, "max_features": .6}
    candidates = (reference, {**reference, "max_depth": 12, "min_samples_leaf": 3})
    out = np.empty((len(tr), len(COMPONENTS)))
    for a, b in GroupKFold(3).split(tr, groups=tr.linkage_block):
        inner = tr.iloc[a]
        costs = np.zeros((len(candidates), len(COMPONENTS)))
        for u, v in GroupKFold(3).split(inner, groups=inner.linkage_block):
            fit_rows, validation = inner.iloc[u], inner.iloc[v]
            x, y = design(fit_rows, validation, "Original11", "RF")
            w = study_weights(validation)
            for k, candidate in enumerate(candidates):
                for j, target in enumerate(COMPONENTS):
                    estimate = fit(fit_rows, x, y, "RF", candidate, target, "score")
                    costs[k, j] += np.average((validation[target].to_numpy()-estimate)**2, weights=w)
        x, y = design(inner, tr.iloc[b], "Original11", "RF")
        for j, target in enumerate(COMPONENTS):
            chosen = candidates[np.argmin(costs[:, j])]
            out[b, j] = fit(inner, x, y, "RF", chosen, target, "score")
    return out


def figure6(frame, params, saved, rebuild_oof=False):
    results = []
    cfg = params.loc[params.use.eq("F6_cascade")]
    for habitat, part in cfg.groupby("habitat"):
        tr, te = split(frame, habitat, 102)
        configs = {row.model: row for row in part.itertuples()}
        a, b = design(tr, te, "Original11", "RF")
        components = []
        for code, target in zip("ECH", COMPONENTS):
            setup = configs[code]
            y = fit(tr, a, b, "RF", json.loads(setup.parameters), target, setup.recipe)
            np.testing.assert_allclose(y, saved.loc[te.observation_id, f"cascade_test_{code}"], atol=1e-8, rtol=0)
            components.append(y)
            results.append(dict(seed=102, habitat=habitat, model=code, metric="test_R2", value=r2_score(te[target], y)))
        stage2 = configs["Z"]
        train_cp = saved.loc[tr.observation_id, [f"cascade_oof_{c}" for c in "ECH"]].to_numpy()
        if rebuild_oof:
            rebuilt = build_component_oof(tr)
            np.testing.assert_allclose(rebuilt, train_cp, rtol=0, atol=1e-8)
            train_cp = rebuilt
        query_cp = np.vstack([train_cp, np.column_stack(components)])
        z = fit(tr, train_cp, query_cp, "RF", json.loads(stage2.parameters), "Z", stage2.recipe)
        np.testing.assert_allclose(z, saved.loc[pd.concat([tr, te]).observation_id, "cascade_Z_102"], atol=1e-8, rtol=0)
        results.append(dict(seed=102, habitat=habitat, model="Cascade Z", metric="test_R2",
                            value=r2_score(te.Z, z[len(tr):]), **errors(te.Z, z[len(tr):])))
    return results


DIRECT_SEED, CASCADE_SEED = 42, 102
PERMUTATIONS = 40
GROUPS = {
    "Environmental context": ("VIIRS_valid_3km_nW_cm2_sr", "temp_L8_climatology_degC", "elevation_2p5min_m"),
    "Activity & affluence": ("PopDensity_urban_construction", "GRP_pc"),
    "Pollution load & control": ("WW_Discharge", "Waste_Treat", "IPME"),
    "Agriculture": ("agri_film_prov_t", "agri_output_prov_100mCNY", "fertilizer_prov_10kt"),
}


def feature_owner(name):
    return name.split("=", 1)[0].removesuffix("__unknown")


def shapley(model, source, target, names, train, test, seed, paths=10):
    rng = np.random.default_rng(seed)
    owners = sorted({feature_owner(n) for n in names})
    columns = [np.array([feature_owner(n)==owner for n in names]) for owner in owners]
    shape = model(target).shape
    result = np.zeros((len(target), len(owners), shape[1]))
    base = np.zeros(shape)
    for _ in range(paths):
        donors = np.empty(len(test), dtype=int)
        for medium in test.medium.unique():
            training = train.medium.eq(medium).to_numpy()
            if not training.any():
                training = np.ones(len(train), dtype=bool)
            weights = study_weights(train.loc[training])
            testing = test.medium.eq(medium).to_numpy()
            donors[testing] = rng.choice(np.flatnonzero(training), testing.sum(), p=weights/weights.sum())
        matrix = source[donors].copy()
        previous = model(matrix)
        base += previous
        for j in rng.permutation(len(columns)):
            col = columns[j]
            matrix[:, col] = target[:, col]
            current = model(matrix)
            result[:, j] += current-previous
            previous = current
    result /= paths
    base /= paths
    np.testing.assert_allclose(base+result.sum(axis=1), model(target), rtol=1e-8, atol=1e-9)
    return owners, result, base


def summarize_shapley(model, source, query, names, tr, te, habitat, split_seed, outputs):
    owners, phi, reference = shapley(model, source, query, names, tr, te, split_seed + int(habitat == "lake"))
    rows = []
    for j, output in enumerate(outputs):
        values = pd.DataFrame(phi[..., j], columns=owners)
        for predictor in owners:
            by_study = values[predictor].abs().groupby(te.group.to_numpy()).mean()
            rows.append(dict(seed=split_seed, habitat=habitat, output=output, predictor=predictor,
                             group_mean_absolute_shapley=by_study.mean(), studies=len(by_study), paths=10,
                             mean_reference=reference[:, j].mean()))
    return pd.DataFrame(rows)


def group_permutation(model, query, names, te, habitat):
    y = te[list(COMPONENTS)].to_numpy()
    w = study_weights(te)
    baseline = r2_score(y, model(query), sample_weight=w, multioutput="raw_values")
    rng = np.random.default_rng(CASCADE_SEED)
    strata = [np.flatnonzero(te.medium.eq(m).to_numpy()) for m in te.medium.unique()]
    rows = []
    for category, variables in GROUPS.items():
        cols = [i for i, col in enumerate(names) if col in variables]
        assert len(cols) == len(variables)
        loss = []
        for _ in range(PERMUTATIONS):
            current = query.copy()
            for ids in strata:
                current[np.ix_(ids, cols)] = query[np.ix_(rng.permutation(ids), cols)]
            loss.append(baseline - r2_score(y, model(current), sample_weight=w, multioutput="raw_values"))
        for k, code in enumerate("ECH"):
            rows.append(dict(seed=CASCADE_SEED, habitat=habitat, category=category, component=code,
                             delta_r2=np.mean(loss, axis=0)[k], repeat_sd=np.std(loss, axis=0, ddof=1)[k],
                             baseline_r2=baseline[k], permutations=PERMUTATIONS, variables=" | ".join(variables)))
    return pd.DataFrame(rows)



def attribution(frame, params, saved):
    """Recompute direct and component importance for Figures 5 and 6."""
    individual, shap, perm = [], [], []
    from figures.figure5 import ORDER
    for habitat in ("river", "lake"):
        for seed in (DIRECT_SEED, CASCADE_SEED):
            tr, te = split(frame, habitat, seed)
            source, query = design(tr, te, "Original11", "RF")
            names = list(FEATURES[:11]) + [
                f"{k}={v}" for k in ("habitat", "medium")
                for v in sorted(tr[k].fillna("unknown").astype(str).unique())
            ]
            if seed == DIRECT_SEED:
                c = params.query("use == 'F5_fixed' and model == 'RF' and habitat == @habitat").iloc[0]
                predict, rf = fit(tr, source, query, "RF", json.loads(c.parameters),
                                  "Z", c.recipe, return_model=True)
                np.testing.assert_allclose(predict(query), saved.loc[te.observation_id, f"Z_{seed}_RF"], atol=1e-8, rtol=0)
                baseline = r2_score(te.Z, predict(query))
                rng = np.random.default_rng(seed)
                for variable, _ in ORDER:
                    column = names.index(variable)
                    groups = [np.flatnonzero(te.medium.eq(m).to_numpy() & te[variable].notna().to_numpy())
                              for m in te.medium.unique()]
                    loss = []
                    for _ in range(PERMUTATIONS):
                        shuffled = query.copy()
                        for ids in groups:
                            shuffled[ids, column] = query[rng.permutation(ids), column]
                        loss.append(baseline - r2_score(te.Z, predict(shuffled)))
                    individual.append(dict(habitat=habitat, predictor=variable, PI_mean=np.mean(loss),
                                           PI_SD=np.std(loss, ddof=1), MDI=rf.feature_importances_[column]))
                shap.append(summarize_shapley(lambda x: predict(x)[:, None], source, query,
                                              names, tr, te, habitat, seed, ["Z"]))
            else:
                configs = params.query("use == 'F6_cascade' and habitat == @habitat").set_index("model")
                models = []
                for code, target in zip("ECH", COMPONENTS):
                    c = configs.loc[code]
                    predictor, _ = fit(tr, source, query, "RF", json.loads(c.parameters),
                                       target, c.recipe, return_model=True)
                    models.append(predictor)
                def predict_components(x):
                    return np.column_stack([fn(x) for fn in models])
                shap.append(summarize_shapley(predict_components, source, query, names,
                                              tr, te, habitat, seed, COMPONENTS))
                perm.append(group_permutation(predict_components, query, names, te, habitat))
    return pd.DataFrame(individual), pd.concat(shap, ignore_index=True), pd.concat(perm, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path, help="Analysis workbook with predictors, configurations and reference predictions")
    parser.add_argument("--mode", choices=("test", "cv", "cascade", "all"), default="all")
    parser.add_argument("--rebuild-oof", action="store_true", help="Rebuild the cascade's strict OOF inputs")
    parser.add_argument("--output", type=Path, help="Optional CSV destination")
    parser.add_argument("--attribution-output", type=Path, help="Optional directory for Figure 5/6 attribution tables")
    args = parser.parse_args()
    frame, params, saved, summary = read_inputs(args.workbook)
    rows = []
    with threadpool_limits(limits=1):
        for mode in ("test", "cv"):
            if args.mode in (mode, "all"):
                rows.extend(figure5(frame, params, saved, summary, mode))
        if args.mode in ("cascade", "all"):
            rows.extend(figure6(frame, params, saved, args.rebuild_oof))
        if args.attribution_output:
            args.attribution_output.mkdir(parents=True, exist_ok=True)
            tables = attribution(frame, params, saved)
            for table, name in zip(tables, ("direct_importance", "shapley_importance", "grouped_permutation")):
                table.to_csv(args.attribution_output / (name + ".csv"), index=False)
    if args.output:
        pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Modeling: verified {len(rows)} model results")


if __name__ == "__main__":
    main()
