"""Nuisance fits and clustered variance for the ECH DML analysis."""
import numpy as np
import pandas as pd
from scipy.stats import t as student
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 119105
X5 = ["elevation_2p5min_m", "temp_L8_climatology_degC", "precip_prior_yr_mm",
      "pop_3km_persons_km2", "GRP_pc_admin_reference_CNY"]


def control_names(var):
    return [c for c in X5 if not (var=='GRP_pc' and c=='GRP_pc_admin_reference_CNY')]

def weight(g):
    return 1/g.map(g.value_counts()).to_numpy()

def sd(x,w):
    return np.sqrt(np.average((x-np.average(x,weights=w))**2,weights=w))

def group_codes(frame,column,k):
    return pd.Series({j:i for i,(_,test) in enumerate(GroupKFold(n_splits=min(k,frame[column].nunique())).split(
        frame,groups=frame[column])) for j in frame.index[test]}).reindex(frame.index).astype(int)

def fit_nuisance(tr,te,var,algorithm,full_x=False):
    numeric=control_names(var)+['sampling_year_used','ln_window','ln_window_upper','longitude_deg','latitude_deg']
    if full_x:
        numeric += [v for v in tr if v.startswith('legacy_') and v!='legacy_'+var and
                    not (v=='legacy_GRP_pc' and 'GRP_pc_admin_reference_CNY' in numeric)]
    categorical=['method_collection','medium','province_normalized']
    xtr=tr[numeric+categorical].copy(); xte=te[numeric+categorical].copy()
    for x, q in [(xtr,tr),(xte,te)]:
        x.loc[~q.coordinates_usable,['longitude_deg','latitude_deg']]=np.nan
        x[categorical]=x[categorical].fillna('unknown')
    p=ColumnTransformer([
        ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
                         ('scale',StandardScaler())]),numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical)])
    u,v=p.fit_transform(xtr),p.transform(xte);w=weight(tr.assignment)
    pred=[]
    for target in ('Y','T'):
        reg=(Ridge(alpha=1.) if algorithm=='ridge' else RandomForestRegressor(
            n_estimators=150,min_samples_leaf=8,max_features=.8,n_jobs=1,random_state=SEED))
        reg.fit(u,tr[target],sample_weight=w)
        pred.append(reg.predict(v))
    return pred

def cluster_var(score,key):
    a=pd.Series(score).groupby(pd.Series(np.asarray(key))).sum()
    return (a@a)*len(a)/max(len(a)-1,1)


def split(q, mode):
    assert mode == "study"
    studies = group_codes(q, "study", 5)
    for i in sorted(studies.unique()):
        test = studies.eq(i)
        yield f"s{i}", q[~test], q[test]
