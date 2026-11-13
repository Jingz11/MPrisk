"""Fixed E-C-H calculation from regenerated compositions, with no interactions."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit, softmax

SIZE_CONTACT = np.array([.99, .90, .50, .03, .01])

def exposure(abundance, reference):
    """Log abundance relative to the reference for its medium."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(abundance)) - np.log(reference)

def jit_mix(p,b,mix):
    return (1-mix)*np.sum(p*b,axis=-1)+mix*np.log(np.maximum(np.sum(p*np.exp(b),axis=-1),1e-15))

def contact_component(st, f, categories):
    modifiers=[]
    for k in ['shape','color']:
        p=st[k]/st[k].sum(-1,keepdims=True)
        b=np.asarray([f[k][c]['beta'] for c in categories[k]])
        modifiers.append(jit_mix(p,b,f['mix'])-jit_mix(np.asarray(f[k+'_ref']),b,f['mix']))
    return (st['size'] @ SIZE_CONTACT) * np.exp(modifiers[0]+modifiers[1])


def hazard_component(st, f, categories):
    multiplier=np.exp([f['polymer'][c]['beta'] for c in categories['polymer']])
    ref=np.asarray(f['polymer_ref'])
    denominator=ref@multiplier
    if ref[-2:].sum()>0:denominator=ref[:-2]@multiplier[:-2]/ref[:-2].sum()
    multiplier[-2:]=denominator
    return np.log((st['polymer'] @ multiplier) / denominator)


def components(states, medium, parameters):
    """Calculate E, C and H for each supplied conditional state."""
    f = parameters["medium"][medium]
    categories = parameters["categories"]
    return np.stack([exposure(states["A"], f["Aref"]),
                     contact_component(states, f, categories),
                     hazard_component(states, f, categories)], axis=-1)


def fuse(v, ref):
    """Apply the exposure-anchored Softmax to (..., 3) component arrays."""
    u=(v-np.asarray(ref['center']))/np.asarray(ref['scale'])
    r=expit(u);w=softmax(r,axis=-1);g=np.sum(r*w,axis=-1)
    neutral=r.copy();neutral[...,1:]=.5
    g0=np.sum(neutral*softmax(neutral,axis=-1),axis=-1)
    eps=np.finfo(float).eps
    z = 100*expit(u[...,0]+logit(np.clip(g,eps,1-eps))-logit(g0))
    return z, r, w

def summarize(states, medium, parameters, references):
    z, _, _ = fuse(components(states, medium, parameters), references[medium])
    center=float(np.quantile(z,.5,method='inverted_cdf'))
    lo,hi=np.quantile(z,[.05,.95])
    return dict(Z=center,conditional_low=float(lo),conditional_high=float(hi),grade=int(np.searchsorted([20,40,60,80],center,side='right')+1))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path, help="NPZ with A, size, shape, color and polymer for one record")
    parser.add_argument("medium", choices=("water", "sediment"))
    parser.add_argument("--parameters", type=Path, default=Path(__file__).with_name("parameters.json"))
    args = parser.parse_args()
    config = json.loads(args.parameters.read_text(encoding="utf-8"))
    with np.load(args.states, allow_pickle=False) as states:
        print(json.dumps(summarize(states, args.medium, config, config["references"])))
