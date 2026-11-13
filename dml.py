"""Recompute the current Figure 7 DML coefficients from the DML design and final ECH scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path

import nuisance as current
from nuisance import np, pd, student, threadpool_limits
from sklearn.model_selection import KFold

def multipletests(pvals, alpha=.05, method='holm'):
    assert method=='holm'
    p=np.asarray(pvals,dtype=float)
    order=np.argsort(p,kind='stable')
    adjusted=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    out=np.empty_like(p);out[order]=adjusted
    return out<=alpha,out,1-(1-alpha)**(1/len(p)),alpha/len(p)


BASINS = ("Yangtze River Basin", "Pearl River Basin", "Yellow River Basin")
RESIDUAL_COLS = ["observation_id", "study", "geo", "assignment", "province_normalized", "medium", "Y", "T"]
KEYS = ["scope", "habitat", "variable"]


@lru_cache(maxsize=1)
def load(workbook, scores):
    book = pd.read_excel(workbook, sheet_name="DML逐值")
    frame = book.set_index("source_order", verify_integrity=True)
    assert frame.source_year.eq(frame.sampling_year_used).all()
    assert not frame.duplicated(["observation_id", "variable"]).any()
    labels=pd.read_csv(scores,float_precision="round_trip").set_index("id")
    frame["Y"] = frame.observation_id.map(labels.Z)
    frame["T"] = np.log1p(frame.value)
    assert frame.Y.notna().all()
    return frame


def choose_sample(data, variable, habitat, scope):
    keep = data.variable.eq(variable) & data.waterbody_type.eq(habitat)
    if scope != "National":
        keep &= data.basin.eq(scope)
    return data.loc[keep].sort_index()


def crossfit(q, var):
    """Use the current study folds, with the original record folds for one study."""
    by_study = q.study.nunique() >= 2
    mode = "study" if by_study else "record_one_study"
    if by_study:
        folds = current.split(q, "study")
    else:
        splitter = KFold(n_splits=min(5, len(q)), shuffle=True, random_state=42)
        folds = ((f"r{i}", q.iloc[a], q.iloc[b]) for i, (a, b) in enumerate(splitter.split(q)))
    out = []
    with threadpool_limits(limits=1):
        for fold, tr, te in folds:
            if by_study:
                assert not set(tr.study) & set(te.study)
            py, pt = current.fit_nuisance(tr, te, var, "rf")
            r = te[RESIDUAL_COLS].assign(predY=py, predT=pt, fold=fold)
            out.append(r.assign(ry=r.Y-py, rt=r["T"]-pt))
    residual = pd.concat(out, ignore_index=True)
    assert len(residual) == len(q) and residual.observation_id.is_unique
    return residual, mode

def estimate(r):
    """No support threshold: guard only against undefined slope or interval."""
    w = current.weight(r.assignment)
    y, t = r.ry.to_numpy(), r.rt.to_numpy()
    denom = np.dot(w, t*t)
    if denom <= 0 or not np.isfinite(denom):
        return dict(status="undefined_residual_denominator", reason="Zero or non-finite residualized denominator")
    theta = np.dot(w, t*y) / denom
    y_sd, t_sd = current.sd(r.Y, w), current.sd(r["T"], w)
    if y_sd <= 0 or t_sd <= 0 or not np.isfinite(theta):
        return dict(status="undefined_standardized_slope", reason="Undefined standardized slope")
    scale = t_sd / y_sd
    result = dict(theta_points=theta, theta_standardized=theta*scale,
                  residual_sd_ratio=current.sd(t, w)/t_sd)
    # Replicate the existing conservative clustered variance. An interval
    # needs positive degrees of freedom and a positive variance estimate.
    df = min(r.study.nunique(), r.geo.nunique(), r.province_normalized.nunique(), r.assignment.nunique()) - 1
    result["df"] = df
    if df < 1:
        return dict(**result, status="point_only", reason="No positive cluster degrees of freedom")
    influence = w*t*(y-theta*t)/denom
    v1 = current.cluster_var(influence, r.study)
    v2 = current.cluster_var(influence, r.geo)
    v12 = current.cluster_var(influence, r.study.astype(str)+"|"+r.geo.astype(str))
    va = current.cluster_var(influence, r.assignment)
    vp = current.cluster_var(influence, r.province_normalized.fillna("unknown"))
    se = np.sqrt(max(v1, v2, v1+v2-v12, va, vp))
    if not np.isfinite(se) or se <= 0:
        return dict(**result, status="point_only", reason="No positive finite cluster variance")
    delta = student.ppf(.975, df)*se
    return dict(**result, status="estimated", reason="", ci_low=(theta-delta)*scale,
                ci_high=(theta+delta)*scale, p=2*student.sf(abs(theta/se), df))

def fit_cell(task):
    workbook, scores, scope, habitat, variable = task
    q = choose_sample(load(workbook, scores), variable, habitat, scope)
    meta = dict(zip(KEYS, task[2:]), n=len(q), water=int(q.medium.eq("water").sum()),
                sediment=int(q.medium.eq("sediment").sum()), studies=q.study.nunique(),
                assignments=q.assignment.nunique(), geography=q.geo.nunique(),
                provinces=q.province_normalized.nunique(), distinct_treatment=q["T"].nunique())
    if len(q) < 2:
        return dict(**meta, status="undefined_sample", reason="Fewer than two eligible rows"), None
    if q["T"].nunique() < 2 or q.Y.nunique() < 2:
        return dict(**meta, status="undefined_constant", reason="Constant treatment or score"), None
    r, mode = crossfit(q, variable)
    return dict(**meta, fold_mode=mode, **estimate(r)), r.assign(**dict(zip(KEYS, task[2:])))


def run(workbook, scores):
    variables = sorted(load(workbook, scores).variable.unique())
    tasks = [("National", h, v) for h, v in product(("river", "lake"), variables)]
    tasks += [(b, "river", v) for b, v in product(BASINS, variables)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        table = pd.DataFrame([result for result, _ in pool.map(fit_cell, [(workbook, scores, *task) for task in tasks])])
    table["p_Holm"] = np.nan
    for _, part in table.groupby(np.where(table.scope.eq("National"), "national", "basins")):
        p = multipletests(part.p.fillna(1), method="holm")[1]
        table.loc[part.index, "p_Holm"] = np.where(part.p.notna(), p, np.nan)
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path, help="Workbook containing the DML design sheet")
    parser.add_argument("scores", type=Path, help="CSV containing id and final Z")
    parser.add_argument("--output", type=Path, help="Optional CSV destination")
    args = parser.parse_args()
    scores = run(args.workbook, args.scores)
    if args.output:
        scores.to_csv(args.output, index=False)
    print(f"DML: fitted {len(scores)} coefficient cells")
